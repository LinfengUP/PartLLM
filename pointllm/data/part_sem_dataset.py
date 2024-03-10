import json
import os
import random
from plyfile import PlyData, PlyElement
import torch
import torch.nn.functional as F
import numpy as np
from pointllm import conversation as conversation_lib
from pointllm.model.constants import DEATULT_POINT_TOKEN
# from esaydict import EasyDict
import json

from .utils import SHORT_QUESTION_LIST,ANSWER_LIST,pc_norm



class PartSegataset(torch.utils.data.Dataset):
    ignore_label =  255
    def __init__(
        self,
        base_point_dir,
        samples_per_epoch= 500 * 8 * 2 * 10,
        tokenizer=None,
        point_number = 8192,
        precision = "fp32",
        exclude_val=False,
        part_seg_data="partnet",
        # dataroot = "/data2/llf/partnet",
    ):
        self.samples_per_epoch = samples_per_epoch
        self.exclude_val = exclude_val
        self.base_point_dir = base_point_dir
        self.point_number = point_number
        self.tokenizer = tokenizer
        self.precision = precision

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        # self.data2list = {}
        # self.data2classes = {}

        # self.part_seg_datas = part_seg_data.split("||")
        # for ds in self.part_seg_datas:
        #     classes, point_clouds, labels = eval("init_{}".format(ds))(base_point_dir)
        #     self.data2list[ds] = (point_clouds, labels)
        #     self.data2classes[ds] = classes


        self.dataroot = os.path.join(base_point_dir, part_seg_data)
        self.data2class = json.load(open(os.path.join(self.data_root,"data2class_train.json")))
        self.class2data = json.load(open(os.path.join(self.data_root,"class2data_train.json")))

        self.valid_data = [i for i in list(self.data2class.keys()) if len(self.data2class[i])>0]
        self.dataindex = []

        self.text2id = json.load(open(os.path.join(self.dataroot,"meta_class.json")))
        self.id2text = {self.text2id[i]:i for i in self.text2id}

        # DATA_DIR = os.path.join(base_point_dir, "pointllm_dataset")
    def __len__(self):
        num = 0
        for i in self.class2data:
            num += len(self.class2data[i])
        for i in self.valid_data:
            for j in self.data2class[i]:
                self.dataindex.append((i,j))
        return num

    def __getitem__(self, idx):
        data_id, text_id = self.dataindex[index]
        # text_embedding = self.text_embiddings[text_id]

        data_path = os.path.join(self.data_root,"data_v0",data_id,"point_sample")
        point = self.read_ply(os.path.join(data_path,"sample-points-all-pts-nor-rgba-10000.ply"))
        label = torch.load(os.path.join(data_path,"label.pth"))
        # label = np.array([int(x[:-1]) for x in label])
        # print(self.config.npoints)
        index = np.random.choice(point.shape[0],self.config.data.npoints, replace=False)
        # index = random.sample(range(point.shape[0]),self.config.npoints)

        point = point[index]
        # if self.mode == "train":
        #     point = self.rotate(point)

        point = pc_norm(point)
        label = label[index]

        point = torch.from_numpy(point).float()
        
        mask = label==text_id
        mask = mask.long()
        mask[mask==0] = self.ignore_label


        questions = []
        answers = []
        class_ids = []
        # for sampled_cls in sampled_classes:
        text = self.id2text[text_id]

        assert len(text.split("||")) == 1
        question_template = random.choice(self.short_question_list)
        questions.append(question_template.format(class_name = text.lower()))

        answers.append(random.choice(self.answer_list))

        conversations = []
        conv = conversation_lib.default_conversation.copy()

        i = 0
        while i<len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i+=1

        point_cloud = self.prepocess(idx)

        return(
            point_cloud,
            conversations,
            mask,
            questions,
            text,
        )

    def prepocess(self, object_id):
        point_cloud = pc_norm(point_cloud)
        return point_cloud
    
    def read_ply(self,filename):
        """ read XYZ point cloud from filename PLY file """
        plydata = PlyData.read(filename)
        x = np.asarray(plydata.elements[0].data['x'])
        y = np.asarray(plydata.elements[0].data['y'])
        z = np.asarray(plydata.elements[0].data['z'])
        r = np.asarray(plydata.elements[0].data['red'])
        g = np.asarray(plydata.elements[0].data['blue'])
        b = np.asarray(plydata.elements[0].data['green'])
        # print(plydata)
        return np.stack([x,y,z,r,g,b], axis=1)