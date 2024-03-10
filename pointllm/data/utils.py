from collections import OrderedDict, defaultdict

import transformers
from pointllm import conversation as conversation_lib
from dataclasses import dataclass
from typing import Optional, Dict, Sequence
import torch
import os

from enum import Enum
import numpy as np
import torch.distributed as dist

IGNORE_INDEX = -100
POINT_TOKEN_INDEX = -200
DEFAULT_POINT_TOKEN = "<point>"
DEFAULT_POINT_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

SHORT_QUESTION_LIST = [
    DEFAULT_POINT_TOKEN + "\n" + "Can you segment the {class_name} in this model?",
    DEFAULT_POINT_TOKEN + "\n" + "Please segment the {class_name} in this model.",
    DEFAULT_POINT_TOKEN
    + "\n"
    + "What is {class_name} in this model? Please respond with segmentation mask.",
    DEFAULT_POINT_TOKEN
    + "\n"
    + "What is {class_name} in this model? Please output segmentation mask.",
]

LONG_QUESTION_LIST = [
    DEFAULT_POINT_TOKEN + "\n" + "{sent} Please respond with segmentation mask.",
    DEFAULT_POINT_TOKEN + "\n" + "{sent} Please output segmentation mask.",
]

EXPLANATORY_QUESTION_LIST = [
    "Please output segmentation mask and explain why.",
    "Please output segmentation mask and explain the reason.",
    "Please output segmentation mask and give some explanation.",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]

# AFFORDANCE_QUESTION_LIST = [
#     DEFAULT_POINT_TOKEN +
# ]


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


def pc_norm(pc):
    """ pc: NxC, return NxC """
    xyz = pc[:, :3]
    other_feature = pc[:, 3:]

    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    xyz = xyz / m

    pc = np.concatenate((xyz, other_feature), axis=1)
    return pc

def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def dict_to_cuda(input_dict):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.cuda(non_blocking=True) for ele in v]
    return input_dict


# * Sample Usage:
# * from utils import LRUCache
# * cache = LRUCache(capacity, max_access_count)
# if self.cache is None:
#     info_data = self.multiview_scannet[info_index]
# else:
#     info_data = self.cache.get(info_index)
#     if info_data is None or self.cache.get_access_count(info_index) >= self.cache.max_access_count:
#         # If not in cache, or accessed max_access_count times, load it and put it in cache
#         info_data = self.multiview_scannet[info_index]
#         self.cache.put(info_index, info_data)
#         self.cache.reset_access_count(info_index)

class LRUCache:
    def __init__(self, capacity, max_access_count):
        self.cache = OrderedDict()
        self.access_count = defaultdict(int)
        self.capacity = capacity
        self.max_access_count = max_access_count

    def get(self, key):
        if key not in self.cache:
            return None
        value = self.cache.pop(key)
        self.cache[key] = value  # Put key as the newest one
        self.access_count[key] += 1
        return value

    def put(self, key, value):
        if key in self.cache:  # Update the value and put it as newest
            self.cache.pop(key)
        elif len(self.cache) == self.capacity:  # If cache is full
            oldest_key = next(iter(self.cache))
            self.cache.popitem(last=False)  # Remove oldest item
            del self.access_count[oldest_key]  # Remove the corresponding access count
        self.cache[key] = value
        self.access_count[key] = 1

    def get_access_count(self, key):
        return self.access_count.get(key, 0)

    def reset_access_count(self, key):
        self.access_count[key] = 0


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2: # * can handle padded tokens
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX # * this is necessary for padded tokens

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len: # * unk tokens in the dialogue will cause this.
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_multimodal_point_cloud(
    sources: Sequence[str],
    point_backbone_config: dict,
    point_indicator: str = "<point>",
) -> Dict:
    point_token_len = point_backbone_config['point_token_len']
    default_point_patch_token = point_backbone_config['default_point_patch_token']

    for source in sources:
        for sentence in source:
            replace_token = default_point_patch_token * point_token_len 
            if point_backbone_config['mm_use_point_start_end']:
                replace_token = point_backbone_config['default_point_start_token']+ replace_token + point_backbone_config['default_point_end_token']
            sentence["value"] = sentence["value"].replace(point_indicator, replace_token)

    return sources



@dataclass
class DataCollatorForPointTextDataset(object):
    """Collate examples for mixed dataset with text and point cloud data."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'point_clouds' in instances[0]:
            point_clouds = [instance['point_clouds'] for instance in instances]
            if all(x is not None and x.shape == point_clouds[0].shape for x in point_clouds): # * point_clouds have different shapes
                batch['point_clouds'] = torch.stack(point_clouds)
            else:
                batch['point_clouds'] = point_clouds # * return as lists

        return batch


def collate_fn(
        batch, tokenizer=None, conv_type = "llava_v1", use_mm_start_end = True, local_rank =-1
):
    point_cloud_list = []
    conversation_list = []
    masks_list = []
    # labels_list = []
    questions_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    for(
        point_cloud,
        conversation,
        masks,
        # labels,
        questions,
        offset,
        inference,
    ) in batch:
        point_cloud_list.append(point_cloud)
        conversation_list.append(conversation)
        masks_list.append(masks)
        # labels_list.append(labels)
        questions_list.append(questions)
        offset_list.append(offset)
        inferences.append(inference)
        cnt += 1

    if use_mm_start_end:
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_POINT_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_POINT_TOKEN, replace_token
            )
    
    input_ids = [
        tokenizer_point_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,batch_first=True,padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib.default_conversation.copy()
    taregets = input_ids.clone()

    if conv_type=="llava_v1":
        sep = conv.sep + conv.roles[1] + ": "
    else:
        sep = "[/INST]"

    for conversation, target in zip(conversation_list, taregets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            assert len(parts) == 2, (len(parts), rou)

            parts[0] += sep
            
            if DEFAULT_POINT_TOKEN in conversation:
                round_len = len(tokenizer_point_token(rou,tokenizer))
                instruction_len = len(tokenizer_point_token(parts[0],tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

        target[cur_len:] = IGNORE_INDEX

        if cur_len<tokenizer.model_max_length:
            assert cur_len == total_len

    if inferences[0]==False:
        truncate_len = tokenizer.model_max_length - 255

        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]

    return {
        "point_clouds": torch.stack(point_cloud_list),
        "input_ids": input_ids,
        "labels": targets,
        "attention_mask": attention_masks,
        "masks_list": torch.stack(masks_list),
        "offset": torch.LongTensor(offset_list),
        "questions_list" :questions_list,
        "inferences": inferences[0],
        "conversation_list": conversation_list,
    }


def tokenizer_point_token(input_ids,tokenizer,point_token_index=POINT_TOKEN_INDEX,return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in input_ids.split("<point>")]

    def insert_separator(X,sep):
        return [ele for sublist in zip(X,[sep]*len(X)) for ele in sublist][:-1]
    
    input_ids = []
    offset = 0
    if(
        len(prompt_chunks)>0
        and len(prompt_chunks[0])>0
        and prompt_chunks[0][0]==tokenizer.cls_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks,[point_token_index]*(offset+1)):
        input_ids.append(x)
    
    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids,dtype=torch.long)
        raise ValueError(f"Unsupport return_tensors: {return_tensors}")
    return input_ids