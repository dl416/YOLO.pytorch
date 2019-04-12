# import os
# import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader

import dataset
from yolo import YOLO
from yolo import yoloLoss

def train():
    yolo = YOLO()
    yolo = yolo.to("cuda")
    train_dataset = dataset.Voc17()
    train_dataLoader = DataLoader(train_dataset, 16, shuffle=True, num_workers=2)
    mse_loss = torch.nn.MSELoss(reduction="mean")
    criterion = yoloLoss()
    optimizer = torch.optim.Adam(yolo.parameters(), lr=0.0001)

    for e in range(30):
        total_loss = 0.0
        for i, batch_data in enumerate(train_dataLoader):
            image = batch_data['image'].type(torch.FloatTensor)
            image = image.to("cuda")
            label = batch_data['label'].to("cuda").type(torch.FloatTensor)
            label = label.to("cuda")
            out = yolo(image)
            criterion(label, out)


if __name__ == "__main__":
    train()