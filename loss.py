import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import  Variable


class CrossEntropyLoss2d(nn.Module):
    def __init__(self,weight=None,size_average=True,ignore_index=255,reduce=True):
        super(CrossEntropyLoss2d,self).__init__()
        self.nll_loss = nn.NLLLoss(weight,size_average,ignore_index,reduce)
    def forward(self,inputs,targets):
        log_p = F.log_softmax(inputs,dim=1)
        loss = self.nll_loss(log_p,targets)

        return loss


class mIoULoss(nn.Module):
    def __init__(self,n_classes=2):
        super(mIoULoss,self).__init__()
        self.classes = n_classes

    def forward(self,inputs,target_oneHot):

        inputs = F.softmax(inputs,dim=1)
        target_oneHot = torch.cat([1-target_oneHot,target_oneHot],dim=1)
        inter = inputs * target_oneHot
        inter = torch.sum(torch.sum(inter,dim=3),dim=2)
        inputs = torch.sum(torch.sum(inputs,dim=3),dim=2)
        target_oneHot = torch.sum(torch.sum(target_oneHot,dim=3),dim=2)
        union = inputs + target_oneHot -inter
        loss = (inter)/(union + 1e-8)
        loss = torch.mean(loss,dim=0)
        return loss
class F_measure(nn.Module):
    def __init__(self,weight):
        super(F_measure,self).__init__()
        self.weight = weight
    def forward(self,inputs,target_out):
        inputs = F.softmax(inputs, dim=1)[:,1,:,:]
        TP = torch.sum(torch.sum(inputs*target_out,dim=2),dim=1)
        FP = torch.sum(torch.sum(inputs*(1-target_out),dim=2),dim=1)
        FN = torch.sum(torch.sum((1-inputs)*target_out,dim=2),dim=1)
        p  = TP/(TP+FP)
        r = TP/(TP+FN)
        F_measure = (1+self.weight)*p*r/(self.weight*p+r)
        F_measure = torch.mean(F_measure)
        return  F_measure