import torch
import torch.nn as nn
import torchvision.models as models

class YOLO(nn.Module):
    def __init__(self):
        super(YOLO, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        # 原文用的是这个，训练困难，启用，改用卷积来做，还节约显存
        # self.fc1 = nn.Linear(2048*7*7, 4096)
        # self.fc2 = nn.Linear(4096, 7*7*30, bias=False)
        self.con = nn.Conv2d(2048, 30, 1)

    
    def forward(self, x):
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)
        x = self.con(x)
        x = x.permute(0, 2, 3, 1) # batch, 7, 7, 30
        #x = x.reshape(-1, 2048*7*7)
        #x = self.relu(self.fc1(x))
        #x = self.fc2(x)
        #x = x.reshape(-1, 7, 7, 30)
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

        obj_mask = label[:, 20] >= 0.9
        noobj_mask = label[:, 20] < 0.9

        mse = nn.MSELoss(reduction='sum')

        # x and y
        location_loss_1 = mse(label[obj_mask][:, 21:23], predict[obj_mask][:, 21:23])
        location_loss_2 = mse(label[obj_mask][:, 21:23], predict[obj_mask][:, 26:28])

        # h and w
        # ? miss sqrt
        location_loss_3 = mse(label[obj_mask][:, 23:25], predict[obj_mask][:, 23:25])
        location_loss_4 = mse(label[obj_mask][:, 23:25], predict[obj_mask][:, 28:])
        location_loss = location_loss_1 + location_loss_2 + location_loss_3 + location_loss_4

        # is object
        object_loss_1 = mse(label[obj_mask][:, 20], predict[obj_mask][:, 20])
        object_loss_2 = mse(label[obj_mask][:, 20], predict[obj_mask][:, 25])
        object_loss = object_loss_1 + object_loss_2

        noobject_loss_1 = mse(label[noobj_mask][:, 20], predict[noobj_mask][:, 20])
        noobject_loss_2 = mse(label[noobj_mask][:, 20], predict[noobj_mask][:, 25])
        noobject_loss = noobject_loss_1 + noobject_loss_2

        class_loss = mse(label[obj_mask][:, :20], predict[obj_mask][:, :20])

        total_loss = self.l_coord * location_loss + object_loss + self.l_noobj * noobject_loss + class_loss
        # total_loss = object_loss  + noobject_loss * self.l_noobj
        return total_loss

if __name__ == '__main__':
    yolo = YOLO()
    x = torch.randn(10, 3, 224, 224)
    out = yolo(x)
    print(out.shape)
