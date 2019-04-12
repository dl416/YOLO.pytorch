import torch
import torch.nn as nn
import torchvision.models as models

class YOLO(nn.Module):
    def __init__(self):
        super(YOLO, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.fc1 = nn.Linear(2048*7*7, 4096)
        self.fc2 = nn.Linear(4096, 7*7*30)
    
    def forward(self, x):
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)

        x = x.reshape(-1, 2048*7*7)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.reshape(-1, 7, 7, 30)
        return x

class yoloLoss(nn.Module):
    def __init__(self, l_coord=5, l_noobj=0.5, S=7, B=2):
        super(yoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj
    
    def forward(self, label, predict):
        """
            label: batch, 7, 7, 25
            predict: batch, 7, 7, 30 
        """
        label = label.reshape(-1, 25)
        predict = predict.reshape(-1, 30)

        obj_mask = label[:, 20] > 0.8
        p1_mask = predict[:, 20] > 0.8
        p2_mask = predict[:, 25] > 0.8

        mask1 = obj_mask + p1_mask
        mask2 = obj_mask + p2_mask

        label_mask = label[mask1]
        predict_mask = predict[mask1][:, 21:22]

        # p2_mask = label[:, :25] > 0.8

        # x and y
        location_loss_1 = torch.sum(torch.pow(label[mask1][:, 21:23] - predict[mask1][:, 21:23], 2))
        location_loss_2 = torch.sum(torch.pow(label[mask2][:, 21:23] - predict[mask2][:, 26:28], 2))
        location_loss = location_loss_1 + location_loss_2

        # h and w
        # ? miss sqrt
        location_loss_3 = torch.sum(torch.pow(label[mask1][:, 23:25] - predict[mask1][:, 23:25], 2))
        location_loss_4 = torch.sum(torch.pow(label[mask1][:, 23:25] - predict[mask1][:, 28:], 2))
        location_loss += location_loss_3 + location_loss_4
        print(location_loss)



if __name__ == '__main__':
    yolo = YOLO()
    x = torch.randn(10, 3, 224, 224)
    out = yolo(x)
    print(out.shape)