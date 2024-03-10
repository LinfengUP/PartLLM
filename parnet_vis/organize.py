import json
import numpy as np
import pyviz3d.visualizer as viz
import os
from plyfile import PlyData, PlyElement
import torch
from tqdm import tqdm, trange

def dfs(result,include_list,exclude_list,item_class,meta_class):
    # print(result)
    # return
    for i in result:
        if i['name'] in include_list:
            extract_label(i,meta_data,exclude_list,item_class,meta_class)
            # break
        elif 'children' in i:
            dfs(i['children'],include_list,exclude_list,item_class,meta_class)

def extract_label(result,meta_data,exlude_list,item_class,meta_class):
    # print(result)
    i=result
    if 'children' not in i:
        # print(i)
        if i['name'] not in exclude_list:
            part_class[item_class][meta_class].append(i['id'])
    else:
        for children in i['children']:
            extract_label(children,meta_data,exclude_list,item_class,meta_class)


def read_ply(filename):
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


if __name__=="__main__":

    with open('part_meta.json', 'r') as fcc_file:
        meta_data = json.load(fcc_file)
    
    meta_class_id = {}
    index = 0
    for meta_class in meta_data:
        for i in meta_data[meta_class]:
            temp = i+" of "+meta_class.lower()
            # print(temp)
            meta_class_id[temp] = index
            index+=1

    part_class={}
    out_count = 0
    for i in meta_data:
        part_class[i] = {}
        for mode in ['train','val','test']:
            print("-------------------"+mode+"-------------------"+i+"-------------------")
            with open('/data2/llf/partnet/partnet_dataset/stats/train_val_test_split/'+i+'.'+mode+'.json', 'r') as fcc_file:
                train_data = json.load(fcc_file)
                count = 0
                for data in tqdm(train_data):
                    anno_id = data['anno_id']
                    model_id = data['model_id']
                    result_file = '/data2/llf/partnet/data_v0/'+anno_id+'/result.json'
                    with open(result_file, 'r') as result_json:
                        result = json.load(result_json)
                    # print(result)
                    for meta_class in meta_data[i]:
                        if isinstance(meta_data[i][meta_class][0],list):
                            include_list = meta_data[i][meta_class][0]
                            exclude_list = meta_data[i][meta_class][1]
                        else:
                            include_list = meta_data[i][meta_class]
                            exclude_list = []
                        part_class[i][meta_class] = []
                        dfs(result,include_list,exclude_list,i,meta_class)
                    # print(part_class)
        
                    data_path = os.path.join("/data2/llf/partnet/data_v0/",anno_id,"point_sample")
                    # point = read_ply(os.path.join(data_path,"sample-points-all-pts-label-10000.ply"))
                    point = read_ply(os.path.join(data_path,"sample-points-all-pts-nor-rgba-10000.ply"))
                    label = open(os.path.join(data_path,"sample-points-all-label-10000.txt")).readlines()
                    label = np.array([int(x[:-1]) for x in label])
                    color = np.array([[255,255,255]]*10000)
        
                    # print(part_class)
        
                    re = torch.tensor(label)
                    change = np.array([True]*10000)
                    for color_index,name in enumerate(part_class):
                        real_color = 0
                        for sub_name in part_class[name]:
                            temp = sub_name+" of "+name.lower()
                            # print(temp)
                            for index in part_class[name][sub_name]:
                                # color[label==index] = COLOR_DETECTRON2[real_color]
                                # print(color.shape)
                                change[label==index] = False
                                re[label==index] = meta_class_id[temp]
                            real_color+=1
                    re[change] = 255
                    # print(os.path.join(data_path,"label.pth"))
                    torch.save(re,os.path.join(data_path,"label.pth"))
                    # v.add_points(anno_id+"_"+name+"_ori", point[:,:3], point[:,3:], point_size=10)
                    # v.add_points(anno_id+"_"+name+"_lbl", point[:,:3], np.array(color), point_size=10)
        