import torch
import torch.nn as nn
from torch.utils.data import Dataset
import json
import random
import os
import numpy as np
import random
from plyfile import PlyData, PlyElement
import argparse
from easydict import EasyDict
import yaml
import math

class PartPromptDataset(Dataset):

    def __init__(self,config,mode="train") -> None:
        super().__init__()

        self.config = config
        self.data_root = config.data.data_root
        self.mode = mode

        self.text_embiddings = torch.load(os.path.join(self.data_root,"text_embeddings.pth"))

        if self.mode=="val":
            self.data2class = json.load(open(os.path.join(self.data_root,"data2class_val.json")))
            self.class2data = json.load(open(os.path.join(self.data_root,"class2data_val.json")))
        elif self.mode == "train":
            self.data2class = json.load(open(os.path.join(self.data_root,"data2class_train.json")))
            self.class2data = json.load(open(os.path.join(self.data_root,"class2data_train.json")))

        self.valid_data = [i for i in list(self.data2class.keys()) if len(self.data2class[i])>0]
        self.dataindex = []

        self.len = self.__len__()

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

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        xyz = pc[:, :3]
        other_feature = pc[:, 3:]

        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
        xyz = xyz / m

        pc = np.concatenate((xyz, other_feature), axis=1)
        return pc


    def __len__(self):
        num = 0
        for i in self.class2data:
            num += len(self.class2data[i])
        for i in self.valid_data:
            for j in self.data2class[i]:
                self.dataindex.append((i,j))
        return num
    
    def rotate(self,point):
        N,C = point.shape
        if C>3:
            data = point[:,0:3]
        m = np.eye(3)
        theta = np.random.rand() * 2 * math.pi
        m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0],
                    [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
        theta = np.random.rand() * 2 * math.pi
        m = np.matmul(m,[[math.cos(theta), 0, -math.sin(theta)],
                         [0, 1, 0], [math.sin(theta), 0, math.cos(theta)]])
        
        new_data = np.matmul(data, m)
        new_points = np.concatenate((new_data,point[:,3:]),axis=-1)

        return new_points


    def __getitem__(self, index):
        data_id, text_id = self.dataindex[index]
        text_embedding = self.text_embiddings[text_id]

        data_path = os.path.join(self.data_root,"data_v0",data_id,"point_sample")
        point = self.read_ply(os.path.join(data_path,"sample-points-all-pts-nor-rgba-10000.ply"))
        label = torch.load(os.path.join(data_path,"label.pth"))
        # label = np.array([int(x[:-1]) for x in label])
        # print(self.config.npoints)
        index = np.random.choice(point.shape[0],self.config.data.npoints, replace=False)
        # index = random.sample(range(point.shape[0]),self.config.npoints)

        point = point[index]
        if self.mode == "train":
            point = self.rotate(point)

        point = self.pc_norm(point)
        label = label[index]

        point = torch.from_numpy(point).float()
        
        target = label==text_id
        target = target.long()

        return point,text_embedding.unsqueeze(dim=0),target,torch.tensor(text_id).long()

# def get_args():
#     parser = argparse.ArgumentParser('Segmentor')
#     parser.add_argument('--config', type=str, help='path to config file',default="./pointllm/model/pointbert/PointTransformer_base_8192point.yaml")
#     parser.add_argument('--dist', action='store_true', help='run with distributed parallel')
#     parser.add_argument('--resume', type=str, help='path to resume from')
#     parser.add_argument('--work_dir', type=str, help='working directory')
#     args = parser.parse_args()
#     return args

# def cfg_from_yaml_file(cfg_file):
#     config = EasyDict()
#     with open(cfg_file, 'r') as f:
#         new_config = yaml.load(f, Loader=yaml.FullLoader)
#     merge_new_config(config=config, new_config=new_config)
#     return config
# def merge_new_config(config, new_config):
#     for key, val in new_config.items():
#         if not isinstance(val, dict):
#             if key == '_base_':
#                 with open(new_config['_base_'], 'r') as f:
#                     try:
#                         val = yaml.load(f, Loader=yaml.FullLoader)
#                     except:
#                         val = yaml.load(f)
#                 config[key] = EasyDict()
#                 merge_new_config(config[key], val)
#             else:
#                 config[key] = val
#                 continue
#         if key not in config:
#             config[key] = EasyDict()
#         merge_new_config(config[key], val)
#     return config

# if __name__=="__main__":
#     args = get_args()

#     point_bert_config_name = args.config
#     point_bert_config = cfg_from_yaml_file(point_bert_config_name)

#     dataset = PartPromptDataset(point_bert_config)

#     print(1)