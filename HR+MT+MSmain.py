from loss import *
from decoder import *
from data import *
from Config import configs
import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel._functions import Scatter
import logging


def get_log(filename):
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fh = logging.FileHandler(filename, mode='a')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def scatter(inputs, target_gpus, chunk_sizes, dim=0):
    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            try:
                return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
            except Exception:
                print('obj', obj.size())
                print('dim', dim)
                print('chunk_sizes', chunk_sizes)
                quit()
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]

    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


def scatter_kwargs(inputs, kwargs, target_gpus, chunk_sizes, dim=0):
    inputs = scatter(inputs, target_gpus, chunk_sizes, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, chunk_sizes, dim) if kwargs else []

    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


class BalancedDataParallel(DataParallel):
    def __init__(self, gpu0_bsz, *args, **kwargs):
        self.gpu0_bsz = gpu0_bsz
        super().__init__(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        if self.gpu0_bsz == 0:
            device_ids = self.device_ids[1:]
        else:
            device_ids = self.device_ids
        inputs, kwargs = self.scatter(inputs, kwargs, device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids)
        if self.gpu0_bsz == 0:
            replicas = replicas[1:]
        outputs = self.parallel_apply(replicas, device_ids, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def parallel_apply(self, replicas, device_ids, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, device_ids)

    def scatter(self, inputs, kwargs, device_ids):
        bsz = inputs[0].size(self.dim)
        num_dev = len(self.device_ids)
        gpu0_bsz = self.gpu0_bsz
        bsz_unit = (bsz - gpu0_bsz) // (num_dev - 1)
        if gpu0_bsz < bsz_unit:
            chunk_sizes = [gpu0_bsz] + [bsz_unit] * (num_dev - 1)
            delta = bsz - sum(chunk_sizes)
            for i in range(delta):
                chunk_sizes[i + 1] += 1
            if gpu0_bsz == 0:
                chunk_sizes = chunk_sizes[1:]
        else:
            return super().scatter(inputs, kwargs, device_ids)

        return scatter_kwargs(inputs, kwargs, device_ids, chunk_sizes, dim=self.dim)




if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
    model = MSMTRD(7)
    device_c=1
    conf = configs()
    lr_now =0.001
    total_epochs = conf.epochs
    train_data,val_data = get_loader(conf)
    train_log =  get_log(conf.base + '/' + 'result' + '/' + 'train_loghrmtms7.txt')
    val_log = get_log(conf.base +'/' + 'result'+'/' +'val_loghrmtms7.txt')
    class_weight = torch.FloatTensor([0.3]).cuda()
    road_loss = F_measure(class_weight)
    iou_loss = mIoULoss()
    for epoch in range(1,total_epochs+1):
        train_loss = 0
        train_iter = 0
        if epoch %10==0:
            model = model.to(device=device_c)
            lr_now = lr_now *0.1
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr_now)

        for sample in train_data:
                gt = sample['gt']
                model.train()
                optimizer.zero_grad()

                img_f1 = sample['img_f1'].to(device=device_c)
                img_f2 = sample['img_f2'].to(device=device_c)
                img_f3 = sample['img_f3'].to(device=device_c)
                img = sample['img'].to(device=device_c)
                img_b1 = sample['img_b1'].to(device=device_c)
                img_b2 = sample['img_b2'].to(device=device_c)
                img_b3 = sample['img_b3'].to(device=device_c)

                output = model( img_f1, img_f2, img_f3, img, img_b1, img_b2, img_b3)
                gt = sample['gt'].long().to(device=device_c)
                class_weight = torch.FloatTensor([torch.sum(gt).float() / (torch.sum(1 - gt) + torch.sum(gt)),
                                                  torch.sum(1 - gt).float() / (torch.sum(1 - gt) + torch.sum(gt))]).to(device=device_c)
                criterion1 = nn.CrossEntropyLoss(weight=class_weight)
                loss = criterion1(output, gt)
                loss.backward()
                optimizer.step()
                train_loss = train_loss + loss
                train_iter += 1
        train_log.info('train' + '\t' + str(epoch) + '\t' + str(train_loss/train_iter))
    torch.save(model.cpu().state_dict(), conf.base + '/' + 'model' + '/' + 'model.pth')
































