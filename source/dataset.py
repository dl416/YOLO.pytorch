import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np 
import xml.etree.ElementTree as ET

classes_dic = {
    "aeroplane": 0,
    "bicycle": 1,
    "bird": 2,
    "boat": 3,
    "bottle": 4,
    "bus": 5,
    "car": 6,
    "cat": 7,
    "chair": 8,
    "cow": 9,
    "diningtable": 10,
    "dog": 11,
    "horse": 12,
    "motorbike": 13,
    "person": 14,
    "pottedplant": 15,
    "sheep": 16,
    "sofa": 17,
    "train": 18,
    "tvmonitor": 19
}

class Voc17(Dataset):
    trainset_path = "./ImageSets/Main/train.txt"
    testset_path = "./ImageSets/Main/test.txt"
    image_path = "./JPEGImages/"
    label_path = "./Annotations/"
    def __init__(self, voc_dir="/home/l/data/VOCdevkit/VOC2007/", is_train=True, transform=None):
        self.voc_dir = voc_dir
        if is_train:
            self.trainset_path = os.path.join(voc_dir, self.trainset_path)
        else:
            self.trainset_path = os.path.join(voc_dir, self.testset_path)
        with open(self.trainset_path) as f:
            self.data = f.readlines()
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # read image
        idx = self.data[idx].strip()
        image_path = os.path.join(self.voc_dir, self.image_path, idx+".jpg")
        image = Image.open(image_path)
        w, h = image.size
        image = image.resize((224, 224), Image.ANTIALIAS)
        image = np.array(image)
        if (image.shape) == 2:
            image = np.tile(image, (1, 1, 3))
        if self.transform:
            image = self.transform(image)
        #image = image.permute(1, 2, 0)

        # read bbox
        # 前20保存类别， 后5个保存坐标
        label = np.zeros((7, 7, 25))
        xml_path = os.path.join(self.voc_dir, self.label_path, idx+".xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root:
            if obj.tag == "object":
                for att in obj:
                    if att.tag == "name" and att.text not in classes_dic:
                        break
                    if att.tag == "name" and att.text in classes_dic:
                        name = att.text
                    if att.tag == "bndbox":
                        for site in att:
                            if site.tag == "xmin":
                                xmin = site.text
                            elif site.tag == "ymin":
                                ymin = site.text
                            elif site.tag == "xmax":
                                xmax = site.text
                            elif site.tag == "ymax":
                                ymax = site.text
                xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
                local_cell_h, local_cell_w, rate_local_y, rate_local_x, rate_bbox_h, rate_bbox_w = self.compute_local(h, w, xmin, ymin, xmax, ymax)
                clss = np.zeros(20)
                clss[int(classes_dic[name])] = 1
                label[local_cell_h, local_cell_w, :20] = clss
                label[local_cell_h, local_cell_w, 20:] = np.array([1.0, rate_local_y, rate_local_x, rate_bbox_h, rate_bbox_w])
        
        sample = {
            'label': label,
            'image': image,
            #'origin': [Image.open(image_path)]
        }
        return sample
    
    def compute_local(self, h, w, xmin, ymin, xmax, ymax):

        # bbox 的高和宽
        bbox_h = ymax - ymin
        bbox_w = xmax - xmin

        # 按图片大小进行归一化
        rate_bbox_h = bbox_h / h
        rate_bbox_w = bbox_w / w

        # cell 的高和宽
        cell_h = h / 7
        cell_w = w / 7

        # bbox 的中心
        center_x = (xmax + xmin) / 2
        center_y = (ymax + ymin) / 2

        # bbox 的中心属于哪个 cell
        local_cell_h = int(center_y / cell_h)
        local_cell_w = int(center_x / cell_w)

        # 按 cell 的比例进行归一化 
        rate_local_y = center_y / cell_h - local_cell_h
        rate_local_x = center_x / cell_w - local_cell_w

        return local_cell_h, local_cell_w, rate_local_y, rate_local_x, rate_bbox_h, rate_bbox_w


if __name__ == "__main__":
    dataset = Voc17()
    idataset = iter(dataset)
    print(len(dataset))
    data = next(idataset)
    print(data['label'][:,:,20] > 0)


