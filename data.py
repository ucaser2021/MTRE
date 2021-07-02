import os
import numpy as np
from torch.utils import data
from torchvision import transforms as T
import scipy.io as scio
import torch
import cv2
import cv2
import random
from torch.utils.data.sampler import WeightedRandomSampler

class Numpy2Tensor_img(object):
    def __init__(self, channels):
        self.channels = channels

    def __call__(self, img):

        channels = self.channels
        img_copy = np.zeros([channels, img.shape[0], img.shape[1]])

        for i in range(channels):
            img_copy[i, :, :] = np.reshape(img, [1, img.shape[0], img.shape[1]]).copy()

        if not isinstance(img_copy, np.ndarray) and (img_copy.ndim in {2, 3}):
            raise TypeError('img should be ndarray. Got {}'.format(type(img_copy)))

        if isinstance(img_copy, np.ndarray):
            # handle numpy array
            img_copy = torch.Tensor(img_copy)
            # backward compatibility
            return img_copy.float()
def randomHorizationalFlip(img1,img2,img3,img4,img5,img6,img7,mask,index):
    np.random.seed(index)
    if np.random.random() < 0.5:
        img1 = cv2.flip(img1,1)
        img2 = cv2.flip(img2,1)
        img3 = cv2.flip(img3, 1)
        img4 = cv2.flip(img4, 1)
        img5 = cv2.flip(img5, 1)
        img6 = cv2.flip(img6, 1)
        img7 = cv2.flip(img7, 1)
        mask = cv2.flip(mask,1)
    return img1,img2,img3,img4,img5,img6,img7,mask
def randomVerticleFlip(img1,img2,img3,img4,img5,img6,img7,mask,index):
    np.random.seed(index)
    if np.random.random()<0.5:
        img1 = cv2.flip(img1, 0)
        img2 = cv2.flip(img2, 0)
        img3 = cv2.flip(img3, 0)
        img4 = cv2.flip(img4, 0)
        img5 = cv2.flip(img5, 0)
        img6 = cv2.flip(img6, 0)
        img7 = cv2.flip(img7, 0)
        mask = cv2.flip(mask, 0)
    return img1, img2, img3, img4, img5, img6, img7, mask
def randomRotate90(img1,img2,img3,img4,img5,img6,img7,mask,index):
    np.random.seed(index)
    if np.random.random()<0.5:
        img1 = np.rot90(img1)
        img2 = np.rot90(img2)
        img3 = np.rot90(img3)
        img4 = np.rot90(img4)
        img5 = np.rot90(img5)
        img6 = np.rot90(img6)
        img7 = np.rot90(img7)
        mask = np.rot90(mask)
    return img1, img2, img3, img4, img5, img6, img7, mask

class ImageFolder(data.DataLoader):
    def __init__(self,train_img_path):
        self.img_files = os.listdir(train_img_path)
        self.img_path = train_img_path
        self.image_size = 256
        self.toimage = Numpy2Tensor_img(3)
        self.len = len(self.img_files)

    def __getitem__(self, index):


        image_path = self.img_path +'/' +self.img_files[index]
        img_f1f = image_path + '/' + 'img_f1.mat'
        img_f1 = scio.loadmat(img_f1f)['img_f1']
        img_f2f = image_path + '/' + 'img_f2.mat'
        img_f2 = scio.loadmat(img_f2f)['img_f2']
        img_f3f = image_path + '/' + 'img_f3.mat'
        img_f3 = scio.loadmat(img_f3f)['img_f3']
        img_f = image_path + '/' + 'img.mat'
        img = scio.loadmat(img_f)['img']
        img_b1f = image_path + '/' + 'img_b1.mat'
        img_b1 = scio.loadmat(img_b1f)['img_b1']
        img_b2f = image_path + '/' + 'img_b2.mat'
        img_b2 = scio.loadmat(img_b2f)['img_b2']
        img_b3f = image_path + '/' + 'img_b3.mat'
        img_b3 = scio.loadmat(img_b3f)['img_b3']
        gt_path = image_path + '/' + 'gt.mat'
        gt = scio.loadmat(gt_path)['gt_curr']


        img_f1 = np.log2(abs(img_f1) + 1) / 16.0
        img_f2 = np.log2(abs(img_f2) + 1) / 16.0
        img_f3 = np.log2(abs(img_f3) + 1) / 16.0
        img = np.log2(abs(img) + 1) / 16.0
        img_b1 = np.log2(abs(img_b1) + 1) / 16.0
        img_b2 = np.log2(abs(img_b2) + 1) / 16.0
        img_b3 = np.log2(abs(img_b3) + 1) / 16.0

        img_f1, img_f2, img_f3, img_b1, img_b2, img_b3, img, gt = randomHorizationalFlip(img_f1, img_f2, img_f3,img_b1, img_b2, img_b3,img, gt,index)
        img_f1, img_f2, img_f3, img_b1, img_b2, img_b3, img, gt = randomVerticleFlip(img_f1, img_f2, img_f3, img_b1,img_b2, img_b3, img, gt,index)
        img_f1, img_f2, img_f3, img_b1, img_b2, img_b3, img, gt = randomRotate90(img_f1, img_f2, img_f3, img_b1, img_b2, img_b3, img, gt,index)


        img_f1 = np.array(img_f1, np.float)
        img_f2 = np.array(img_f2, np.float)
        img_f3 = np.array(img_f3, np.float)
        img_b1 = np.array(img_b1, np.float)
        img_b2 = np.array(img_b2, np.float)
        img_b3 = np.array(img_b3, np.float)
        img = np.array(img, np.float)
        gt = np.array(gt,np.float)



        img_f1 = self.toimage(img_f1)
        img_f2 = self.toimage(img_f2)
        img_f3 = self.toimage(img_f3)
        img_b1 = self.toimage(img_b1)
        img_b2 = self.toimage(img_b2)
        img_b3 = self.toimage(img_b3)

        img = self.toimage(img)
        gt = torch.from_numpy(gt)
        Norm_ = T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))

        img_f1 = Norm_(img_f1)
        img_f2 = Norm_(img_f2)
        img_f3 = Norm_(img_f3)
        img = Norm_(img)
        img_b1 = Norm_(img_b1)
        img_b2 = Norm_(img_b2)
        img_b3 = Norm_(img_b3)

        sample = {

                  'img_f1': img_f1,
                  'img_f2': img_f2,
                  'img_f3':img_f3,
                  'img': img,
                  'img_b1':img_b1,
                  'img_b2':img_b2,
                  'img_b3':img_b3,

                  'gt': gt,

                  }
        return sample

    def __len__(self):
        return self.len*8


def get_loader(conf,num_workers=4):
    dataset = ImageFolder(conf.train_img_path)
    '''
    weight,sample_num = dataset.cal_weight()
    weight = torch.from_numpy(weight)
    weight = torch.DoubleTensor(weight)
    sampler = WeightedRandomSampler(weight, num_samples=sample_num, replacement=True)
    '''
    torch.manual_seed(0)
    data_loader_train = data.DataLoader(dataset=dataset,batch_size=conf.batch_size,shuffle=True, num_workers=num_workers,pin_memory=True)
    dataset = ImageFolder(conf.val_img_path)
    torch.manual_seed(0)
    data_loader_val = data.DataLoader(dataset=dataset, batch_size=conf.batch_size, shuffle=True, num_workers=num_workers,pin_memory=True)
    return data_loader_train,data_loader_val


