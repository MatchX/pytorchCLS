# encoding: utf-8
"""
version=0.1
"""

from matchx.matchutils import *
import torch
from .torchtools import torch_distributed_zero_first
import torch.distributed as dist
# import tensorflow as tf
import torchvision
import torchvision.transforms as transforms
import os
import math
import sys
import shutil
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import random
import json
from matchx import torchtools

import skimage.io

"""
##################################################################################################################
在此之后添加需要的数据加载功能
"""


class roadDataSet(torch.utils.data.dataset.Dataset):
    """
    数据加载器
    分类数据标签序号必须从0开始并且连续
    """

    def __init__(self, root_path, dataconfig, istrain=True):  # , imgcache=False
        self.root_path = root_path
        self.dataconfig = dataconfig
        self.torchtransform = None
        self.istrain = istrain
        self.imgmap = {}  # 图片map key=id val=filename
        self.imginfo = []  # 标签列表 内包含对应图片信息
        mmg = multiprocessing.Manager()
        self.imgbufmap = mmg.dict()
        self.bboxextsize = dataconfig.bboxextsize
        self.showimg = False
        self.cacheimg = dataconfig.cacheimg
        self.lbbalance = dataconfig.lbbalance  # 启用标签均衡
        if istrain:
            self.imgprepath = '/images'
            self.jsonname = '/train_sub.json'
        else:
            self.imgprepath = '/images'
            self.jsonname = '/val.json'
            self.lbbalance = False
        # self.seq = iaa.Sequential([
        #     iaa.Affine(
        #         # translate_percent={"x": (-0.25, 0.25), "y": (-0.25, 0.25)},  # translate by -20 to +20 percent (per axis)
        #         rotate=(-5, 5),  # rotate by -45 to +45 degrees
        #         # shear=(-5, 5),  # shear by -16 to +16 degrees
        #         mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        #     ),
        #     iaa.Fliplr(p=0.5),
        #     iaa.GaussianBlur(sigma=(0, 0.2))
        # ])
        self.imgaugseg = None
        self.albutrans = None
        if 'transformpipeline' in dataconfig:
            transkey = 'imgaug'
            if transkey in dataconfig.transformpipeline:
                self.imgaugseg = dataconfig.transformpipeline[transkey]
            transkey = 'albutrans'
            if transkey in dataconfig.transformpipeline:
                self.albutrans = dataconfig.transformpipeline[transkey]
            transkey = 'torchtrans'
            if transkey in dataconfig.transformpipeline:
                self.torchtransform = dataconfig.transformpipeline[transkey]
        self.__premakedata()

    def __premakedata(self):
        traindata = json.load(open(self.root_path + self.jsonname, 'r'))
        for it in traindata['images']:
            self.imgmap[it['id']] = it
            # makecache
            imginx = 0
            if self.cacheimg:
                imgname = it['file_name']
                imgfullpath = self.root_path + self.imgprepath + "/" + imgname
                with open(imgfullpath, 'rb') as imgf:
                    imgbuf = imgf.read()
                # imgbgr = cv2.imread(imgfullpath)
                self.imgbufmap[it['id']] = imgbuf
                imginx += 1

        labelmap = [0] * 8  # 类别统计
        labelmx = [1, 1, 5, 5, 1, 1, 2, 20]
        for it in traindata['annotations']:
            if self.showimg:
                self.__makedata(it)
            clnum = it['category_id'] - 1
            labelmap[clnum] += 1
            if self.lbbalance:
                for i in range(labelmx[clnum]):
                    self.imginfo.append(it)
            else:
                self.imginfo.append(it)
        if self.istrain:
            print("lb_0=", labelmap[0] * labelmx[0])
            print("lb_1=", labelmap[1] * labelmx[1])
            print("lb_2=", labelmap[2] * labelmx[2])
            print("lb_3=", labelmap[3] * labelmx[3])
            print("lb_4=", labelmap[4] * labelmx[4])
            print("lb_5=", labelmap[5] * labelmx[5])
            print("lb_6=", labelmap[6] * labelmx[6])
            print("lb_7=", labelmap[7] * labelmx[7])

    def __makedata(self, dataitem):
        # 生成bbox抖动随机数
        topratio = random.uniform(self.dataconfig.bxtopratio[0], self.dataconfig.bxtopratio[1])
        leftratio = random.uniform(self.dataconfig.bxleftratio[0], self.dataconfig.bxleftratio[1])
        bottomratio = random.uniform(self.dataconfig.bxbottomratio[0], self.dataconfig.bxbottomratio[1])
        rightratio = random.uniform(self.dataconfig.bxrightratio[0], self.dataconfig.bxrightratio[1])
        imgid = dataitem['image_id']
        imgname = self.imgmap[imgid]['file_name']
        bbox = dataitem['bbox']
        label = dataitem['category_id']
        if self.cacheimg:
            imgbgr = skimage.io.imread(self.imgbufmap[imgid], plugin='imageio')
        else:
            imgfullpath = self.root_path + self.imgprepath + "/" + imgname
            # img = skimage.io.imread(imgbuf, plugin='imageio')
            imgbgr = cv2.imread(imgfullpath)
        if self.istrain:
            bxmin = int(bbox[0])
            bxmax = int(bbox[0] + bbox[2])
            bymin = int(bbox[1])
            bymax = int(bbox[1] + bbox[3])
            if self.showimg:
                imgorg = imgbgr.copy()
                cv2.rectangle(imgorg, (bxmin, bymin), (bxmax, bymax), (0, 0, 255), 1)
                cv2.imshow("orgimg", imgorg)
                cv2.waitKey(1)
            imgwidth = imgbgr.shape[1]
            imgheight = imgbgr.shape[0]
            bwidth = bbox[2]
            bheight = bbox[3]
            xmin = bxmin - bwidth * leftratio
            xmin = 0 if xmin < 0 else xmin
            xmax = bxmax + bwidth * rightratio
            xmax = imgwidth if xmax > imgwidth else xmax
            ymin = bymin - bheight * topratio
            ymin = 0 if ymin < 0 else ymin
            ymax = bymax + bheight * bottomratio
            ymax = imgheight if ymax > imgheight else ymax
            # bboxext
            if self.bboxextsize > 1.0:
                curW = xmax - xmin
                curH = ymax - ymin
                xdiff = curW * (self.bboxextsize - 1.0) / 2
                ydiff = curH * (self.bboxextsize - 1.0) / 2
                xmin = xmin - xdiff
                xmin = 0 if xmin < 0 else xmin
                xmax = xmax + xdiff
                xmax = imgwidth if xmax > imgwidth else xmax
                ymin = ymin - ydiff
                ymin = 0 if ymin < 0 else ymin
                ymax = ymax + ydiff
                ymax = imgheight if ymax > imgheight else ymax
            newbox = [xmin, ymin, xmax - xmin, ymax - ymin]
            newbox = [int(round(it, 0)) for it in newbox]
            subimg = imgbgr[newbox[1]:(newbox[1] + newbox[3]), newbox[0]:(newbox[0] + newbox[2])]
            if self.showimg:
                cv2.imshow("subimg", subimg)
                cv2.waitKey(1)
            if self.imgaugseg is not None:
                imageaug = self.imgaugseg(image=subimg)
            else:
                imageaug = subimg
            if self.showimg:
                cv2.imshow("imgaug", imageaug)
                cv2.waitKey(1)
            if self.albutrans is not None:
                imageaug = self.albutrans(image=imageaug)['image']
            if self.showimg:
                cv2.imshow("albutrans", imageaug)
                cv2.waitKey(1)
            # imgrgb = imageaug[:, :, ::-1]
            imgresize = imageaug  # cv2.resize(imageaug, (self.dataconfig.traininputsize[0], self.dataconfig.traininputsize[1]), interpolation=cv2.INTER_CUBIC)
        else:
            subimg = imgbgr[bbox[1]:(bbox[1] + bbox[3]), bbox[0]:(bbox[0] + bbox[2])]
            imgresize = cv2.resize(subimg, (self.dataconfig.testinputsize[0], self.dataconfig.testinputsize[1]), interpolation=cv2.INTER_CUBIC)
        # 转换输入图像
        # if self.transform is not None:
        #     image = self.transform(image)
        # imgout = np.ascontiguousarray(imgout)
        # imgtensor = torch.from_numpy(imgout)

        imgout = imgresize[:, :, ::-1].transpose((2, 0, 1))  # BGR to RGB and HWC to CHW
        imgout = np.ascontiguousarray(imgout).astype(np.float32) / 255.0  # normal to 1
        # imgout[0] = (imgout[0][:] - self.dataconfig.data_mean[0]) / self.dataconfig.std_mean[0]
        # imgout[1] = (imgout[1][:] - self.dataconfig.data_mean[1]) / self.dataconfig.std_mean[0]
        # imgout[2] = (imgout[2][:] - self.dataconfig.data_mean[2]) / self.dataconfig.std_mean[0]
        imgtensor = torch.from_numpy(imgout)
        if self.torchtransform is not None:
            # aa = Image.fromarray(imgresize)
            imgtensor = self.torchtransform(imgtensor)
            if self.showimg:
                # transform2 = transforms.Compose([
                #     transforms.ToTensor(), ]
                # )
                pil_im = torchtools.transform_invert(imgtensor.cpu(), self.torchtransform)
                # a_img = np.asarray(pil_im)
                plt.figure("img")
                plt.imshow(pil_im)
                plt.show()
                cv2.waitKey(0)
        # 生成标签
        # category_id = [0] * self.dataconfig.num_class
        # category_id[label - 1] = 1
        # labeltensor = torch.from_numpy(np.array(category_id))
        return imgtensor, label - 1

    def __getitem__(self, idx):
        info = self.imginfo[idx]
        return self.__makedata(info)

    def __len__(self):
        return len(self.imginfo)


class dataloaderMaker(object):
    """
    数据集生成器
    """

    def __init__(self):
        pass

    @staticmethod
    def makeroad_dataset(modelconfig, rank=-1):
        # 测试集不采用预加载数据
        # dataconfigdict.data_mean, dataconfigdict.std_maen,
        # datamean = (0.4914, 0.4822, 0.4465), datastd = (0.2023, 0.1994, 0.2010),
        dataconfig = modelconfig.datacfg
        hypconfig = modelconfig.hypcfg

        datamean = dataconfig.data_mean
        datastd = dataconfig.std_mean
        workernum = dataconfig.workernum
        droplast = dataconfig.droplast

        with torch_distributed_zero_first(rank):
            trainset = roadDataSet(root_path=dataconfig.traindata, dataconfig=dataconfig)
        trainsampler = torch.utils.data.distributed.DistributedSampler(trainset) if rank != -1 else None
        # trainset = makedataset('./dataset/flower_data/train', transform_train)
        loader_class = torch.utils.data.DataLoader
        if dataconfig.multi_epochs_loader:
            loader_class = torchtools.MultiEpochsDataLoader
        trainloader = loader_class(trainset, batch_size=hypconfig.trainbatchsize, shuffle=True, drop_last=droplast, num_workers=workernum, pin_memory=True, sampler=trainsampler)
        # if hypconfig.preLoader:
        #     trainloader = torch.utils.data.DataLoader(trainset, batch_size=hypconfig.trainbatchsize, shuffle=True, drop_last=droplast, num_workers=workernum, pin_memory=True,
        #                                               sampler=trainsampler, collate_fn=torchtools.data_Prefetcher.fast_collate)
        if dataconfig.preloader:
            # prefetch_re_prob = re_prob if is_training and not no_aug else 0.
            trainloader = torchtools.PrefetchLoader(
                trainloader,
                mean=datamean,
                std=datastd,
                fp16=hypconfig.useamp
            )

        # from torchvision.utils import make_grid
        # aa = time.time()
        # for i, (imngs, targets) in enumerate(trainloader):
        #     # grid = make_grid(imngs).cpu()
        #     # plt.imshow(grid.numpy().transpose(1, 2, 0))
        #     # plt.show()
        #     pass
        # bb = time.time()
        # cc = bb - aa
        # print(cc)

        # 测试集数据
        testloader = None
        if rank in [-1, 0]:
            with torch_distributed_zero_first(rank):
                testset = roadDataSet(dataconfig.valdata, dataconfig, istrain=False)
            testloader = torch.utils.data.DataLoader(testset, batch_size=hypconfig.testbatchsize, shuffle=False, num_workers=workernum, pin_memory=True)

        datasetloader = torchtools.dataSetLoader(trainloader, testloader, False, datamean, datastd)
        return datasetloader


def creatdata(a1, d):
    a = a1
    b = "fdsafas"
    c = time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime(time.time()))
    # time.sleep(0.1)
    return a, b, c, d


def testparallefun():
    t0 = time.process_time()
    datamaker = torchtools.dataParallelMaker(datafun=creatdata, funarg=(7, "fewa"), pnum=4, buffsize=100)
    t1 = time.process_time()
    print("usetime=", (t1 - t0) * 1000, "ms")

    tn = 0
    while tn < 5000:
        outdata = datamaker.getdata()
        print(tn, outdata[0], outdata[1], outdata[2])
        tn += 1
    datamaker.stop()


if __name__ == '__main__':
    testfun()
