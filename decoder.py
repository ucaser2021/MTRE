import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint
from memonger import SublinearSequential
nonlinearity = nn.ReLU(inplace=True)
from  encoder import *
import  time
sec = 0.0000000000000001
def conv(in_channels,out_channels,kernel_size):
        if kernel_size==1:
            return SublinearSequential(
                nn.Conv2d(in_channels,out_channels,1),
                nn.BatchNorm2d(out_channels),
                nonlinearity
            )
        else:
            return SublinearSequential(
                nn.Conv2d(in_channels,out_channels,3,padding=1),
                nn.BatchNorm2d(out_channels),
                nonlinearity
            )

class Temporal_Encoder(nn.Module):
    def __init__(self,in_channels,out_channels,T):
        super(Temporal_Encoder,self).__init__()
        self.T = T

        self.c1m1_a = conv(in_channels,out_channels,1)
        self.c1m1_b = conv(out_channels,out_channels,3)
        self.c1k1_a = conv(in_channels,out_channels//4,1)
        self.c1k1_b = conv(out_channels//4,out_channels//4,3)

        self.c1m2_a = conv(in_channels,out_channels,1)
        self.c1m2_b = conv(out_channels,out_channels,3)
        self.c1k2_a = conv(in_channels,out_channels//4,1)
        self.c1k2_b = conv(out_channels//4,out_channels//4,3)

        self.c1m3_a = conv(in_channels,out_channels,1)
        self.c1m3_b = conv(out_channels,out_channels,3)
        self.c1k3_a = conv(in_channels,out_channels//4,1)
        self.c1k3_b = conv(out_channels//4,out_channels//4,3)

        self.c1m4_a = conv(in_channels,out_channels,1)
        self.c1m4_b = conv(out_channels,out_channels,3)
        self.c1k4_a = conv(in_channels,out_channels//4,1)
        self.c1k4_b = conv(out_channels//4,out_channels//4,3)

        self.c1m5_a = conv(in_channels,out_channels,1)
        self.c1m5_b = conv(out_channels,out_channels,3)
        self.c1k5_a = conv(in_channels,out_channels//4,1)
        self.c1k5_b = conv(out_channels//4,out_channels//4,3)

        self.c1m6_a = conv(in_channels,out_channels,1)
        self.c1m6_b = conv(out_channels,out_channels,3)
        self.c1k6_a = conv(in_channels,out_channels//4,1)
        self.c1k6_b = conv(out_channels//4,out_channels//4,3)




        self.c1v_a = conv(in_channels,out_channels,1)
        self.c1v_b = conv(out_channels,out_channels,3)
        self.c1k_a = conv(in_channels,out_channels//4,1)
        self.c1k_b = conv(out_channels//4,out_channels//4,3)



        self.c2m1_a = conv(in_channels,out_channels,1)
        self.c2m1_b = conv(out_channels,out_channels,3)
        self.c2k1_a = conv(in_channels,out_channels//4,1)
        self.c2k1_b = conv(out_channels//4,out_channels//4,3)

        self.c2m2_a = conv(in_channels,out_channels,1)
        self.c2m2_b = conv(out_channels,out_channels,3)
        self.c2k2_a = conv(in_channels,out_channels//4,1)
        self.c2k2_b = conv(out_channels//4,out_channels//4,3)

        self.c2m3_a = conv(in_channels,out_channels,1)
        self.c2m3_b = conv(out_channels,out_channels,3)
        self.c2k3_a = conv(in_channels,out_channels//4,1)
        self.c2k3_b = conv(out_channels//4,out_channels//4,3)

        self.c2m4_a = conv(in_channels,out_channels,1)
        self.c2m4_b = conv(out_channels,out_channels,3)
        self.c2k4_a = conv(in_channels,out_channels//4,1)
        self.c2k4_b = conv(out_channels//4,out_channels//4,3)

        self.c2m5_a = conv(in_channels,out_channels,1)
        self.c2m5_b = conv(out_channels,out_channels,3)
        self.c2k5_a = conv(in_channels,out_channels//4,1)
        self.c2k5_b = conv(out_channels//4,out_channels//4,3)

        self.c2m6_a = conv(in_channels,out_channels,1)
        self.c2m6_b = conv(out_channels,out_channels,3)
        self.c2k6_a = conv(in_channels,out_channels//4,1)
        self.c2k6_b = conv(out_channels//4,out_channels//4,3)



        self.c2v_a = conv(in_channels,out_channels,1)
        self.c2v_b = conv(out_channels,out_channels,3)
        self.c2k_a = conv(in_channels,out_channels//4,1)
        self.c2k_b = conv(out_channels//4,out_channels//4,3)





    def cal_similarity(self,x_1k,x_2k,x_3k,x_4k,x_5k,x_6k,x_k):

        x_1k = torch.sum(x_1k * x_k, dim=1).unsqueeze(1)
        x_2k = torch.sum(x_2k * x_k, dim=1).unsqueeze(1)
        x_3k = torch.sum(x_3k * x_k, dim=1).unsqueeze(1)
        x_4k = torch.sum(x_4k * x_k, dim=1).unsqueeze(1)
        x_5k = torch.sum(x_5k * x_k, dim=1).unsqueeze(1)
        x_6k = torch.sum(x_6k * x_k, dim=1).unsqueeze(1)


        similarity = torch.cat([x_1k,x_2k,x_3k,x_4k,x_5k,x_6k],dim=1)
        del x_1k,x_2k,x_3k,x_4k,x_5k,x_6k
        torch.cuda.empty_cache()
        similarity = F.softmax(similarity,dim=1)
        return similarity



    def forward(self,x_1,x_2,x_3,x_4,x_5,x_6,x):
        N,C,H,W = x.size()



        x_1m = self.c1m1_a(x_1)
        x_1m = self.c1m1_b(x_1m)
        x_1 = self.c1k1_a(x_1)
        x_1 = self.c1k1_b(x_1)



        x_2m = self.c1m2_a(x_2)
        x_2m = self.c1m2_b(x_2m)
        x_2 = self.c1k2_a(x_2)
        x_2 = self.c1k2_b(x_2)



        x_3m = self.c1m3_a(x_3)
        x_3m = self.c1m3_b(x_3m)
        x_3 = self.c1k3_a(x_3)
        x_3 = self.c1k3_b(x_3)



        x_4m = self.c1m4_a(x_4)
        x_4m = self.c1m4_b(x_4m)
        x_4 = self.c1k4_a(x_4)
        x_4 = self.c1k4_b(x_4)



        x_5m = self.c1m5_a(x_5)
        x_5m = self.c1m5_b(x_5m)
        x_5 = self.c1k5_a(x_5)
        x_5 = self.c1k5_b(x_5)



        x_6m = self.c1m6_a(x_6)
        x_6m = self.c1m6_b(x_6m)
        x_6 = self.c1k6_a(x_6)
        x_6 = self.c1k6_b(x_6)



        x_v = self.c1v_a(x)
        x_v = self.c1v_b(x_v)
        x = self.c1k_a(x)
        x = self.c1k_b(x)



        similarity = self.cal_similarity(x_2,x_3,x_4,x_5,x_6,x,x_1)
        similarity = similarity.reshape(N,self.T,H,W,1,1)
        x_c = torch.cat([x_2m.unsqueeze(1),x_3m.unsqueeze(1),x_4m.unsqueeze(1),x_5m.unsqueeze(1),x_6m.unsqueeze(1),x_v.unsqueeze(1)],dim=1)
        x_c = x_c.permute(0,1,3,4,2)
        x_c = x_c.reshape(N,self.T,H,W,C,1)
        x_c = torch.matmul(x_c,similarity)
        x_c = x_c.reshape(N,self.T,H,W,C)
        x_c = torch.sum(x_c,dim=1).permute(0,3,1,2)
        x_1m_c = x_1m + x_c




        similarity = self.cal_similarity(x_1,x_3,x_4,x_5,x_6,x,x_2)
        similarity = similarity.reshape(N,self.T,H,W,1,1)
        x_c = torch.cat([x_1m.unsqueeze(1),x_3m.unsqueeze(1),x_4m.unsqueeze(1),x_5m.unsqueeze(1),x_6m.unsqueeze(1),x_v.unsqueeze(1)],dim=1)
        x_c = x_c.permute(0,1,3,4,2)
        x_c = x_c.reshape(N,self.T,H,W,C,1)
        x_c = torch.matmul(x_c,similarity)
        x_c = x_c.reshape(N,self.T,H,W,C)
        x_c = torch.sum(x_c, dim=1).permute(0, 3, 1, 2)
        x_2m_c = x_2m + x_c



        similarity = self.cal_similarity(x_1,x_2,x_4,x_5,x_6,x,x_3)
        similarity = similarity.reshape(N,self.T,H,W,1,1)
        x_c = torch.cat([x_1m.unsqueeze(1),x_2m.unsqueeze(1),x_4m.unsqueeze(1),x_5m.unsqueeze(1),x_6m.unsqueeze(1),x_v.unsqueeze(1)],dim=1)
        x_c = x_c.permute(0,1,3,4,2)
        x_c = x_c.reshape(N,self.T,H,W,C,1)
        x_c = torch.matmul(x_c,similarity)
        x_c = x_c.reshape(N,self.T,H,W,C)
        x_c = torch.sum(x_c, dim=1).permute(0, 3, 1, 2)
        x_3m_c = x_3m + x_c



        similarity = self.cal_similarity(x_1,x_2,x_3,x_5,x_6,x,x_4)
        similarity = similarity.reshape(N,self.T,H,W,1,1)
        x_c = torch.cat([x_1m.unsqueeze(1),x_2m.unsqueeze(1),x_3m.unsqueeze(1),x_5m.unsqueeze(1),x_6m.unsqueeze(1),x_v.unsqueeze(1)],dim=1)
        x_c = x_c.permute(0,1,3,4,2)
        x_c = x_c.reshape(N,self.T,H,W,C,1)
        x_c = torch.matmul(x_c,similarity)
        x_c = x_c.reshape(N,self.T,H,W,C)
        x_c = torch.sum(x_c, dim=1).permute(0, 3, 1, 2)
        x_4m_c = x_4m + x_c




        similarity = self.cal_similarity(x_1,x_2,x_3,x_4,x_6,x,x_5)
        similarity = similarity.reshape(N,self.T,H,W,1,1)
        x_c = torch.cat([x_1m.unsqueeze(1),x_2m.unsqueeze(1),x_3m.unsqueeze(1),x_4m.unsqueeze(1),x_6m.unsqueeze(1),x_v.unsqueeze(1)],dim=1)
        x_c = x_c.permute(0,1,3,4,2)
        x_c = x_c.reshape(N,self.T,H,W,C,1)
        x_c = torch.matmul(x_c,similarity)
        x_c = x_c.reshape(N,self.T,H,W,C)
        x_c = torch.sum(x_c, dim=1).permute(0, 3, 1, 2)
        x_5m_c = x_5m + x_c




        similarity = self.cal_similarity(x_1 ,x_2 ,x_3 ,x_4 ,x_5 ,x ,x_6)
        similarity = similarity.reshape(N,self.T,H,W,1,1)
        x_c = torch.cat([x_1m.unsqueeze(1),x_2m.unsqueeze(1),x_3m.unsqueeze(1),x_4m.unsqueeze(1),x_5m.unsqueeze(1),x_v.unsqueeze(1)],dim=1)
        x_c = x_c.permute(0,1,3,4,2)
        x_c = x_c.reshape(N,self.T,H,W,C,1)
        x_c = torch.matmul(x_c,similarity)
        x_c = x_c.reshape(N,self.T,H,W,C)
        x_c = torch.sum(x_c, dim=1).permute(0, 3, 1, 2)
        x_6m_c = x_6m + x_c






        similarity = self.cal_similarity(x_1,x_2,x_3,x_4,x_5,x_6,x)
        similarity = similarity.reshape(N,self.T,H,W,1,1)
        x_c = torch.cat([x_1m.unsqueeze(1),x_2m.unsqueeze(1),x_3m.unsqueeze(1),x_4m.unsqueeze(1),x_5m.unsqueeze(1),x_6m.unsqueeze(1)],dim=1)
        x_c = x_c.permute(0,1,3,4,2)
        x_c = x_c.reshape(N,self.T,H,W,C,1)
        x_c = torch.matmul(x_c,similarity)
        x_c = x_c.reshape(N,self.T,H,W,C)
        x_c = torch.sum(x_c, dim=1).permute(0, 3, 1, 2)
        x_v = x_v + x_c

        del x_1,x_2,x_3,x_4,x_5,x_6,x
        torch.cuda.empty_cache()



        x_1m = self.c2m1_a(x_1m_c)
        x_1m = self.c2m1_b(x_1m)
        x_1m_c = self.c2k1_a(x_1m_c)
        x_1m_c = self.c2k1_b(x_1m_c)

        x_2m = self.c2m2_a(x_2m_c)
        x_2m = self.c2m2_b(x_2m)
        x_2m_c = self.c2k2_a(x_2m_c)
        x_2m_c = self.c2k2_b(x_2m_c)


        x_3m = self.c2m3_a(x_3m_c)
        x_3m = self.c2m3_b(x_3m)
        x_3m_c = self.c2k3_a(x_3m_c)
        x_3m_c = self.c2k3_b(x_3m_c)


        x_4m = self.c2m4_a(x_4m_c)
        x_4m = self.c2m4_b(x_4m)
        x_4m_c = self.c2k4_a(x_4m_c)
        x_4m_c = self.c2k4_b(x_4m_c)

        x_5m = self.c2m5_a(x_5m_c)
        x_5m = self.c2m5_b(x_5m)
        x_5m_c = self.c2k5_a(x_5m_c)
        x_5m_c = self.c2k5_b(x_5m_c)


        x_6m = self.c2m6_a(x_6m_c)
        x_6m = self.c2m6_b(x_6m)
        x_6m_c = self.c2k6_a(x_6m_c)
        x_6m_c = self.c2k6_b(x_6m_c)




        x_v2 = self.c2v_a(x_v)
        x_v2 = self.c2v_b(x_v2)
        x_v = self.c2k_a(x_v)
        x_v = self.c2k_b(x_v)



        similarity = self.cal_similarity(x_1m_c,x_2m_c,x_3m_c,x_4m_c,x_5m_c,x_6m_c,x_v)
        similarity = similarity.reshape(N,self.T,H,W,1,1)
        x_c = torch.cat([x_1m.unsqueeze(1),x_2m.unsqueeze(1),x_3m.unsqueeze(1),x_4m.unsqueeze(1),x_5m.unsqueeze(1),x_6m.unsqueeze(1)],dim=1)
        x_c = x_c.permute(0,1,3,4,2)
        x_c = x_c.reshape(N,self.T,H,W,C,1)
        x_c = torch.matmul(x_c,similarity)
        x_c = x_c.reshape(N,self.T,H,W,C)
        x_c = torch.sum(x_c, dim=1).permute(0, 3, 1, 2)
        x_v2 = x_v2 + x_c


        return x_v2


class multi_scale_fusion(nn.Module):
    def __init__(self,in_channel):
        super(multi_scale_fusion, self).__init__()
        self.conv1 = conv(in_channel,in_channel//2,3)
        self.conv2 = conv(in_channel//2,4,1)
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.softmax(x,dim=1)
        return x
class final_layer(nn.Module):
    def __init__(self,in_channel):
        super(final_layer,self).__init__()

        self.last_layer =SublinearSequential(
        nn.Conv2d(in_channels=in_channel,
                              out_channels=in_channel//2,
                              kernel_size=3,
                              padding=1),
        BatchNorm2d(in_channel//2,momentum=BN_MOMENTUM),
        nn.ReLU(inplace=False),
        nn.Conv2d(in_channels=in_channel//2,
                              out_channels=2,
                              kernel_size=1,
                            ),
        )
    def forward(self,x):
        return self.last_layer(x)


class MSMTRD(nn.Module):
    def __init__(self,T):
        super(MSMTRD,self).__init__()
        self.HR = HighResolutionNet()
        in_channels=[32,64,128,256]
        self.MT1 = Temporal_Encoder(in_channels=in_channels[0],out_channels=in_channels[0],T=T-1)
        self.MT2 = Temporal_Encoder(in_channels=in_channels[1], out_channels=in_channels[1], T=T-1)
        self.MT3 = Temporal_Encoder(in_channels=in_channels[2], out_channels=in_channels[2], T=T-1)
        self.MT4 = Temporal_Encoder(in_channels=in_channels[3], out_channels=in_channels[3], T=T-1)
        self.MS = multi_scale_fusion(in_channels[0]+in_channels[1]+in_channels[2]+in_channels[3])
        self.final_layer1 = final_layer(in_channels[0])
        self.final_layer2 = final_layer(in_channels[1])
        self.final_layer3 = final_layer(in_channels[2])
        self.final_layer4 = final_layer(in_channels[3])
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.normal_(m.weight,std=0.001)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

    def forward(self,x_f1,x_f2,x_f3,x,x_b1,x_b2,x_b3):
        HR0,HR1,HR2,HR3 = self.HR(x,x_f1,x_f2,x_f3,x_b1,x_b2,x_b3)
        del x,x_f1,x_f2,x_f3,x_b1,x_b2,x_b3
        torch.cuda.empty_cache()

        HR0 = self.MT1(HR0[1],HR0[2],HR0[3],HR0[4],HR0[5],HR0[6],HR0[0])
        HR1 = self.MT2(HR1[1],HR1[2],HR1[3],HR1[4],HR1[5],HR1[6],HR1[0])
        HR2 = self.MT3(HR2[1],HR2[2],HR2[3],HR2[4],HR2[5],HR2[6],HR2[0])
        HR3 = self.MT4(HR3[1],HR3[2],HR3[3],HR3[4],HR3[5],HR3[6],HR3[0])


        x0_h = HR0.size()[2]
        x0_w = HR0.size()[3]

        HR_F = torch.cat([HR0,F.upsample(HR1, size=(x0_h, x0_w), mode='bilinear'),F.upsample(HR2, size=(x0_h, x0_w), mode='bilinear'),F.upsample(HR3, size=(x0_h, x0_w), mode='bilinear')],dim=1)
        HR_F = self.MS(HR_F)
        HR_F = F.softmax(HR_F,dim=1)


        HR0 = self.final_layer1(HR0)


        HR1 = self.final_layer2(HR1)
        HR1 = F.upsample(HR1,size=(x0_h, x0_w), mode='bilinear')

        HR2 = self.final_layer3(HR2)
        HR2 = F.upsample(HR2, size=(x0_h, x0_w), mode='bilinear')

        HR3 = self.final_layer4(HR3)


        HR3 = F.upsample(HR3, size=(x0_h, x0_w), mode='bilinear')

        HR1 = HR1.unsqueeze(1)
        HR2 = HR2.unsqueeze(1)
        HR3 = HR3.unsqueeze(1)
        HR0 = HR0.unsqueeze(1)

        out = torch.cat([HR0,HR1,HR2,HR3],dim=1)
        HR_F = HR_F.unsqueeze(2)
        HR_F = torch.cat([HR_F,HR_F],dim=2)
        out = out.mul(HR_F)
        out = torch.sum(out,dim=1)
        out = F.upsample(out,size=(1024,1024),mode='bilinear')

        return out





