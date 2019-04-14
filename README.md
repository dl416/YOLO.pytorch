# YOLO.pytorch

## YOLO v1
I remove the last two fc layers and use a conv layer to instead.
It is hard to train fc layers, i do not have time to reproduct yolo like original paper.

This code just has two important part, network used resnet50 like a backbone and yolo loss.
I just train code on Voc2007 for a verification. If you want use this code you should modify your Voc2007 dataset location in train code.

## Question
I do not understand why used two bbox to pridict object.
The two bbox will be similar because loss critize them equally.
Need your help!!!
Thank you very much!!!