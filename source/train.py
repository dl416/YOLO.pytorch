# import os
# import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader

import dataset
from yolo import YOLO
from yolo import yoloLoss

def adjust_learning_rate(optimizer, epoch):
    lr = 0.01 * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def train():
    yolo = YOLO()
    yolo = yolo.to("cuda")
    train_dataset = dataset.Voc17()
    train_dataLoader = DataLoader(train_dataset, 16, shuffle=True, num_workers=2)
    criterion = yoloLoss()
    optimizer = torch.optim.Adam(yolo.parameters(), lr=0.01)

    for e in range(150):
        adjust_learning_rate(optimizer, e)
        total_loss = 0.0
        for i, batch_data in enumerate(train_dataLoader):
            image = batch_data['image'].type(torch.FloatTensor)
            image = image.to("cuda")
            label = batch_data['label'].to("cuda").type(torch.FloatTensor)
            label = label.to("cuda")
            predict = yolo(image)

            optimizer.zero_grad()
            loss = criterion(label, predict)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print("e: {0} total loss: {1}".format(e, total_loss))
    torch.save(yolo, "./model/model.pkl")


if __name__ == "__main__":
    train()