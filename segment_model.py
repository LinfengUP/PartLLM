import argparse
from pointllm.model.pointbert.part_prompt_dataset import PartPromptDataset
from transformers import AutoTokenizer
import torch
import os
from collections import OrderedDict
from pointllm.conversation import conv_templates, SeparatorStyle
from tqdm import tqdm
from pointllm.utils import disable_torch_init
from pointllm.model import *
from pointllm.model.utils import KeywordsStoppingCriteria
import torch.nn.functional as F
from pointllm.data import load_ulip2_objaverse_point_cloud
import copy
import wandb
import os
from pointllm.model.pointbert.point_segmentor import PointSegTransformer
from easydict import EasyDict
import yaml
import torch.nn as nn
from pointllm.model.pointbert.pointnet2_utils import PointNetFeaturePropagation
from pointnet2_ops import pointnet2_utils
from torch.nn import functional as F
import torch.optim as optim
import json

from torch import distributed as dist
import os

def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, 'r') as f:
        new_config = yaml.load(f, Loader=yaml.FullLoader)
    merge_new_config(config=config, new_config=new_config)
    return config
def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                with open(new_config['_base_'], 'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config
def step_lr_after_step(optimizer, base_lr, step, steps):
    for i in range(0,len(steps)-1):
        if step>steps[i] and step<=steps[i+1]:
            lr = base_lr * 0.1**(i)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def calc_iou(masks,target,threshold=0.0):
    masks = masks>threshold
    intersection = torch.sum(masks*target,dim=-1)
    epsilon = 1e-7
    union = torch.sum(masks,dim=-1) + torch.sum(target,dim=-1) - intersection
    batch_iou = intersection / (union + epsilon)

    return batch_iou.unsqueeze(dim=-1)  

def get_dice_loss(masks,iou_pred,target,scale=1000,eps=1e-6):
    masks = masks.sigmoid()
    numerator = 2 * (masks/scale * target).sum(-1)
    denominator = (masks/scale).sum(-1) + (target/scale).sum(-1)
    loss = 1-(numerator+eps)/(denominator+eps)
    return loss.mean()

def get_sigmoid_ce_loss(masks,iou_pred,target):
    target = target.float()
    loss = F.binary_cross_entropy_with_logits(masks,target,reduction="mean")
    return loss

def get_args():
    parser = argparse.ArgumentParser('Segmentor')
    parser.add_argument('--config', type=str, help='path to config file',default="./pointllm/model/pointbert/experiment-1.yaml")
    parser.add_argument('--dist', action='store_true', help='run with distributed parallel')
    parser.add_argument('--experiment_name', type=str, default="experiment-1")
    args = parser.parse_args()
    return args
    

def train(args, run=None):

    do_log = run is not None
    is_master = args.local_rank == 0

    point_bert_config_name = args.config
    point_bert_config = cfg_from_yaml_file(point_bert_config_name)
    point_bert_config.model.point_dims = 6
    point_bert_config.model.mask_threshold = 0.0
    use_max_pool = getattr(point_bert_config.model, "use_max_pool", False)
    total_step = getattr(point_bert_config.train, "total_step", 100000)
    base_lr = getattr(point_bert_config.train, "base_lr", 0.001)
    resume = getattr(point_bert_config.model, "resume", None)
    eval_freq = getattr(point_bert_config.train, "eval_freq", 2000)
    save_freq = getattr(point_bert_config.train, "save_freq", 4000)
    log_freq = getattr(point_bert_config.train, "log_freq", 100)
    # save_dir = getattr(point_bert_config.model, "save_dir", "./segmentor_models")

    save_dir = os.path.join("/data1/linfeng/PartLLM",point_bert_config_name.split("/")[-1][:-5])
    
    isExist = os.path.exists(save_dir)
    if not isExist:
        os.makedirs(save_dir)

    if args.dist:
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)
        dist.init_process_group(backend="nccl")

    # if args.work_dir:
    #     cfg.work_dir = args.work_dir
    # else:
    #     cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    # logger

        
        
    train_dataset = PartPromptDataset(point_bert_config)
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.dist else None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=point_bert_config.data.batch_size,
        shuffle=(sampler is None),
        num_workers=point_bert_config.data.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=True
    )

    segmentor = PointSegTransformer(point_bert_config,use_max_pool).cuda()
    if args.dist:
        segmentor = nn.parallel.DistributedDataParallel(segmentor, device_ids=[torch.cuda.current_device()],find_unused_parameters=True)

        
    # optimizer = optim.SGD(segmentor.parameters(), lr=base_lr)
    optimizer = optim.Adam(segmentor.parameters(), lr=base_lr)

    if resume is not None:
        if hasattr(segmentor,"module"):
            segmentor = segmentor.module
        device = torch.cuda.current_device()
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage.cuda(device))
        segmentor.prompt_encoder.load_state_dict(checkpoint["prompt_encoder"])
        segmentor.mask_decoder.load_state_dict(checkpoint["mask_decoder"])
        if optimizer is not None:
            assert 'optimizer' in checkpoint
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("loading model from {}".format(resume))


    current_step = 0

    if train_loader.sampler is not None and args.dist:
        train_loader.sampler.set_epoch(current_step)


    segmentor.train()
    train_data_iter = iter(train_loader)

    for step in range(1,total_step+1):
        current_step = step
        cur_lr = step_lr_after_step(optimizer, base_lr, step, [0,28000,40000,total_step])
        try:
            batch_data = next(train_data_iter)
        except:
            if train_loader.sampler is not None and args.dist:
                train_loader.sampler.set_epoch(current_step)
            train_data_iter = iter(train_loader)
            batch_data = next(train_data_iter)
        
        point,text_embedding,target,_ = batch_data
        point = point.cuda()
        text_embedding = text_embedding.cuda()
        target = target.cuda()

        optimizer.zero_grad()

        masks,iou_pred = segmentor(point,text_embedding)

        dice_loss = get_dice_loss(masks,iou_pred,target)
        sigmoid_ce_loss = get_sigmoid_ce_loss(masks,iou_pred,target)

        batch_iou = calc_iou(masks,target)
        iou_loss = F.mse_loss(batch_iou,iou_pred,reduction="mean")

        loss = dice_loss + sigmoid_ce_loss + iou_loss

        loss.backward()
        optimizer.step()

        if step % log_freq == 0 and is_master:
            print("step: ",step,"total_step: ", total_step," loss: ",loss.item()," dice_loss: ",dice_loss.item()," sigmoid_ce_loss: ",sigmoid_ce_loss.item()," iou_loss: ",iou_loss.item())
            
        if do_log and step%5==0:
            wandb.log({"loss":loss.item(),"lr":cur_lr,"dice_loss":dice_loss.item(),"sigmoid_ce_loss":sigmoid_ce_loss.item(),"iou_loss":iou_loss.item(),"step":step})

        if step%eval_freq==0 and is_master:
            validate(segmentor,point_bert_config,save_dir,step,run)
            segmentor.train()
        if step%save_freq==0 and is_master:
            checkpoint_save(segmentor,optimizer, os.path.join(save_dir, "model_{}.pth".format(step)))

        torch.cuda.empty_cache()
        
def validate(segmentor,point_bert_config,save_dir,step,run=None):
    do_log = run is not None
    batch_size = getattr(point_bert_config.data, "eval_batch_size", 100)
    dataset = PartPromptDataset(point_bert_config,mode="val")
    # sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        # sampler = sampler,
        shuffle=False,
        num_workers=point_bert_config.data.workers,
        pin_memory=True,
        drop_last=False
    )

    result = {}

    torch.cuda.empty_cache()
    segmentor.eval()
    eval_iou = 0.0
    total_num = 0
    with torch.no_grad():
        for batch_id,batch_data in enumerate(tqdm(dataloader)):
            point,text_embedding,target,text_id = batch_data
            point = point.cuda()
            text_embedding = text_embedding.cuda()
            target = target.cuda()

            masks,iou_pred = segmentor(point,text_embedding)

            batch_iou = calc_iou(masks,target).squeeze()

            batch_ious = batch_iou.cpu().numpy()

            batch_iou = batch_iou.mean().cpu().item()

            cur_len = point.shape[0]
            eval_iou = (eval_iou*total_num + batch_iou*cur_len)/(total_num+cur_len)

            total_num += cur_len

            for idi,i in enumerate(text_id):
                i = i.item()
                if i not in result:
                    result[i] = []
                    result[i].append(0)
                    result[i].append(0)
                # else:
                result[i][0] = result[i][0] + batch_ious[idi]
                result[i][1] = result[i][1] + 1

            # break

    for i in result:
        result[i][0] = result[i][0]/(result[i][1]+1e-5)

    with open(os.path.join(save_dir,"result_{}.json".format(step)), 'w') as f:
        json.dump(result, f)
    
    mIoU = 0.0
    for i in result:
        mIoU += result[i][0]
    mIoU = mIoU/len(result)

    if do_log:
        wandb.log({"eval_iou":eval_iou})
        wandb.log({"mIoU":mIoU})

    print("eval_iou: ",eval_iou)
    print("mIoU: ",mIoU)

def checkpoint_save(model,optimizer,path):
    if hasattr(model,"module"):
        model = model.module
    checkpoint = {
        # "point_encoder":checkpoint_to_cpu(model.point_encoder),
        "prompt_encoder":checkpoint_to_cpu(model.prompt_encoder),
        "mask_decoder":checkpoint_to_cpu(model.mask_decoder),
        "optimizer":optimizer.state_dict(),
    }
    torch.save(checkpoint, path)
    print("saving model to {}".format(path))

def checkpoint_to_cpu(model):
    state_dict_cpu = OrderedDict()
    for key, val in model.state_dict().items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu

if __name__ == "__main__":
    args = get_args()
    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if args.local_rank == 0:  # only on main process
        # # Initialize wandb run
        run = wandb.init(
            project="pointllm",
            name = args.config.split("/")[-1][:-5],
        )
        # Train model with DDP
        # run = None
        train(args, run)
    else:
        train(args)



        

# torchrun --nproc_per_node=4 train_segmentor.py --dist --config ./pointllm/model/pointbert/experiment-1.yaml --experiment_name experiment-1

