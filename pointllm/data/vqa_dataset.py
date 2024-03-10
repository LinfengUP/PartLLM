import json
import os
import random

import torch
import torch.nn.functional as F

from pointllm import conversation as conversation_lib
from pointllm.model.constants import DEATULT_POINT_TOKEN

from .utils import SHORT_QUESTION_LIST,ANSWER_LIST,pc_norm



class VQADataset(torch.utils.data.Dataset):
    ignore_label =  255
    def __init__(
        self,
        base_point_dir,
        samples_per_epoch,
        tokenizer=None,
        point_number = 8192,
        precision = "fp32",
        exclude_val=False,
        vqa_data="PointLLM_brief_description_660K",
        # vqa_data = "/data2/llf/objaverse"
    ):
        self.samples_per_epoch = samples_per_epoch
        self.exclude_val = exclude_val
        self.base_point_dir = base_point_dir
        self.point_number = point_number
        self.tokenizer = tokenizer
        self.precision = precision
        
        # DATA_DIR = os.path.join(base_point_dir, "pointllm_dataset")
        DATA_DIR = os.path.join(base_point_dir, "objaverse")
        self.vqa_data_root = os.path.join(DATA_DIR,"8192_npy")
        with open(os.path.join(DATA_DIR, "{}.json".format(vqa_data))) as f:
            vqa_data = json.load(f)
        self.vqa_data = vqa_data

        print("vqa_data: ", len(self.vqa_data))

    def __getitem__(self, idx):
        idx = random.randint(0, len(self.vqa_data) - 1)
        item = self.vqa_data[idx]
        point_cloud = self.prepocess(item["object_id"])

        conv = conversation_lib.default_conversation.copy()
        source = item["conversations"]
        
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        conversations = []
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

        questions = conversations
        sampled_classes = conversations

        masks = torch.rand(0, self.point_number)
        labels = torch.ones(self.point_number) * self.ignore_label

        return(
            point_cloud,
            conversations,
            masks,
            questions,
            sampled_classes,
        )

    def prepocess(self, object_id):
        point_cloud = load_objaverse_point_cloud(self.vqa_data_root, object_id, pointnum=8192, use_color=True)
        return point_cloud

import numpy as np
def load_objaverse_point_cloud(data_path, object_id, pointnum=8192, use_color=False):
    filename = f"{object_id}_{pointnum}.npy"
    point_cloud = np.load(os.path.join(data_path, filename))

    # * normalize
    point_cloud = pc_norm(point_cloud)

    if not use_color:
        point_cloud = point_cloud[:, :3]

    return point_cloud