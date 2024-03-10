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


def calc_iou(masks,target,threshold=0.0):
    masks = masks>threshold
    intersection = torch.sum(masks*target,dim=-1)
    epsilon = 1e-7
    union = torch.sum(masks,dim=-1) + torch.sum(target,dim=-1) - intersection
    batch_iou = intersection / (union + epsilon)

    return batch_iou.unsqueeze(dim=-1)  


def get_args():
    parser = argparse.ArgumentParser('Segmentor')
    parser.add_argument('--config', type=str, help='path to config file',default="./pointllm/model/pointbert/experiment-5.yaml")
    parser.add_argument('--resume', type=str, help='path to resume from',default="/data1/linfeng/PartLLM/experiment-5/model_34000.pth")
    args = parser.parse_args()
    return args
    

def checkpoint_save(model,optimizer,path):
    if hasattr(model,"module"):
        model = model.module
    checkpoint = {
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
    # args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    point_bert_config_name = args.config
    point_bert_config = cfg_from_yaml_file(point_bert_config_name)
    point_bert_config.model.point_dims = 6
    point_bert_config.model.mask_threshold = 0.0
    use_max_pool = getattr(point_bert_config.model, "use_max_pool", False)
    dataset = PartPromptDataset(point_bert_config,mode="val")
    # sampler = torch.utils.data.distributed.DistributedSampler(dataset)s
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=100,
        # sampler = sampler,
        shuffle=False,
        num_workers=point_bert_config.data.workers,
        pin_memory=True,
        drop_last=False
    )

    result = {}

    segmentor = PointSegTransformer(point_bert_config,use_max_pool).cuda()

    checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
    segmentor = PointSegTransformer(point_bert_config,use_max_pool).cuda()
    segmentor.prompt_encoder.load_state_dict(checkpoint["prompt_encoder"])
    segmentor.mask_decoder.load_state_dict(checkpoint["mask_decoder"])

    segmentor.eval()
    with torch.no_grad():

    
        for batch_id,batch_data in enumerate(tqdm(dataloader)):
            torch.cuda.empty_cache()
            point,text_embedding,target,text_id = batch_data
            point = point.cuda()
            text_embedding = text_embedding.cuda()
            target = target.cuda()

            masks,iou_pred = segmentor(point,text_embedding)

            batch_iou = calc_iou(masks,target).squeeze()

            batch_iou = batch_iou.cpu().numpy()

            for idi,i in enumerate(text_id):
                i = i.item()
                if i not in result:
                    result[i] = []
                    result[i].append(0)
                    result[i].append(0)
                # else:
                result[i][0] = result[i][0] + batch_iou[idi]
                result[i][1] = result[i][1] + 1
            print(result)

        for i in result:
            result[i][0] = result[i][0]/(result[i][1]+1e-5)


    with open('./result_exp_5.json', 'w') as f:
        json.dump(result, f)
        
    print(result)





        

# torchrun --nproc_per_node=4 train_segmentor.py --dist --config ./pointllm/model/pointbert/experiment-1.yaml --experiment_name experiment-1

