# encoding: utf-8
"""
version=0.1
"""

from matchx.matchutils import *
import torch
from matchx.torchtools import torch_distributed_zero_first
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


class kitchen_hatDataSet(torch.utils.data.dataset.Dataset):
    """
    数据加载器
    分类数据标签序号必须从0开始并且连续

    厨师帽分类数据
    """

    def __init__(self, root_path, dataconfig, istrain=True):
        # 配置信息
        self.root_pathlist = root_path
        self.label_map = dataconfig.label_map
        self.num_class = dataconfig.num_class
        self.dataconfig = dataconfig
        self.cacheimg = dataconfig.cacheimg
        self.lbbalance = dataconfig.lbbalance  # 启用标签均衡
        self.bboxextsize = dataconfig.bboxextsize
        self.istrain = istrain
        # 成员数据
        self.torchtransform = None
        self.imginfolist = []  # 图片标签列表 内包含对应图片信息
        # self.imgmap = {}  # 图片map key=id val=filename
        mmg = multiprocessing.Manager()
        self.imgbufmap = mmg.dict()  # 图片缓存map key=id val=缓存数据
        self.IMGAUGSEG_Enable = torch.multiprocessing.Value(ctypes.c_bool, istrain)
        self.showimg = False

        # 数据增强配置
        self.imgaugseg = None
        self.albutrans = None
        if istrain:
            if 'transformpipeline4train' in dataconfig:
                transkey = 'imgaug'
                if transkey in dataconfig.transformpipeline4train:
                    self.imgaugseg = dataconfig.transformpipeline4train[transkey]
                transkey = 'albutrans'
                if transkey in dataconfig.transformpipeline4train:
                    self.albutrans = dataconfig.transformpipeline4train[transkey]
                transkey = 'torchtrans'
                if transkey in dataconfig.transformpipeline4train:
                    self.torchtransform = dataconfig.transformpipeline4train[transkey]
        else:
            if 'transformpipeline4test' in dataconfig:
                transkey = 'imgaug'
                if transkey in dataconfig.transformpipeline4test:
                    self.imgaugseg = dataconfig.transformpipeline4test[transkey]
                transkey = 'albutrans'
                if transkey in dataconfig.transformpipeline4test:
                    self.albutrans = dataconfig.transformpipeline4test[transkey]
                transkey = 'torchtrans'
                if transkey in dataconfig.transformpipeline4test:
                    self.torchtransform = dataconfig.transformpipeline4test[transkey]
        # 数据预加载
        if self.dataconfig.rebuild_label:
            self.__makefilelabel()
        self.__premakedata()

    def __makefilelabel(self):
        for it in self.root_pathlist:
            output_label_path = os.path.join(it, "label.txt")
            walk = os.walk(it)
            # 遍历指定数据目录及其子目录生成便签文件
            with open(output_label_path, "w") as fr:
                for root, dirs, files in walk:
                    for file in files:
                        if ".jpg" in file:
                            dir_name = os.path.split(root)[-1]
                            label = self.label_map[dir_name]
                            relative_path = os.path.join(root, file).replace(it, "")
                            fr.writelines(relative_path + "\t" + str(label) + "\n")

    def __premakedata(self):
        labelcountlist = [0] * self.num_class  # 每个类别数据总数统计

        imginx = 0
        for filedir in self.root_pathlist:
            label_path = os.path.join(filedir, "label.txt")
            with open(label_path) as fr:
                lines = fr.readlines()
            print("scan %s. Find %s samples." % (filedir, len(lines)))
            for line in lines:
                path, label = line.strip().split("\t")
                img_path = os.path.join(filedir, path)
                assert os.path.exists(img_path)
                label = int(label)
                dataitem = (img_path, label)
                self.imginfolist.append(dataitem)
                imginx += 1
                labelcountlist[label] += 1
                # makecache
                if self.cacheimg:
                    with open(img_path, 'rb') as imgf:
                        imgbuf = imgf.read()
                    self.imgbufmap[imginx] = imgbuf
                if self.showimg:
                    self.__makedata(imginx - 1)

        for i in range(len(labelcountlist)):
            print("label_" + str(i) + "=", labelcountlist[i])

    def __makedata(self, dataindex):
        imgpath, label = self.imginfolist[dataindex]
        if self.cacheimg:
            imgbgr = skimage.io.imread(self.imgbufmap[dataindex], plugin='imageio')
        else:
            imgbgr = cv2.imread(imgpath)
            assert imgbgr is not None
        if self.showimg:
            cv2.imshow("imgorg", imgbgr)
            cv2.waitKey(1)
        if self.IMGAUGSEG_Enable.value and self.imgaugseg is not None:
            imageaug = self.imgaugseg(image=imgbgr)
            if self.showimg:
                cv2.imshow("imgaug", imageaug)
                cv2.waitKey(1)
        else:
            imageaug = imgbgr
        if self.albutrans is not None:
            imageaug = self.albutrans(image=imageaug)['image']
            if self.showimg:
                cv2.imshow("albutrans", imageaug)
                cv2.waitKey(1)
        # imgrgb = imageaug[:, :, ::-1].transpose((2, 0, 1)).copy()  # BGR to RGB and HWC to CHW
        # imgtensor = torch.from_numpy(imageaug)
        pilimage = Image.fromarray(cv2.cvtColor(imageaug, cv2.COLOR_BGR2RGB))
        # pilimage.show()
        if self.torchtransform is not None:
            imgtensor = self.torchtransform(pilimage)
            if self.showimg:
                pil_im = torchtools.transform_invert(imgtensor.cpu(), self.torchtransform)
                # plt.figure("img")
                # plt.imshow(pil_im)
                # plt.show()
                print("show pic=" + imgpath)
                imgbgr = cv2.cvtColor(np.asarray(pil_im), cv2.COLOR_RGB2BGR)
                cv2.imshow("imgout", imgbgr)
                cv2.waitKey(0)
        return imgtensor, label

    def __getitem__(self, idx):
        data = self.__makedata(idx)
        return data

    def __len__(self):
        return len(self.imginfolist)


class dataloaderMaker(object):
    """
    数据集生成器
    """

    def __init__(self):
        pass

    @staticmethod
    def makeroad_dataset(modelconfig, rank=-1):
        # 生成数据加载器
        dataconfig = modelconfig.datacfg
        hypconfig = modelconfig.hypcfg

        datamean = dataconfig.data_mean
        datastd = dataconfig.std_mean
        workernum = dataconfig.workernum
        droplast = dataconfig.droplast

        with torch_distributed_zero_first(rank):
            datasets = kitchen_hatDataSet(root_path=dataconfig.traindata, dataconfig=dataconfig, istrain=True)
        trainsampler = torch.utils.data.distributed.DistributedSampler(datasets) if rank != -1 else None
        loader_class = torch.utils.data.DataLoader
        if dataconfig.multi_epochs_loader:
            loader_class = torchtools.MultiEpochsDataLoader
        datashuffle = True if trainsampler is None else False
        trainloader = loader_class(datasets, batch_size=hypconfig.trainbatchsize, shuffle=datashuffle, drop_last=droplast, num_workers=workernum, pin_memory=True, sampler=trainsampler)
        if dataconfig.preloader:
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
            if len(dataconfig.valdata) > 0:
                with torch_distributed_zero_first(rank):
                    testset = kitchen_hatDataSet(dataconfig.valdata, dataconfig, istrain=False)
                testloader = torch.utils.data.DataLoader(testset, batch_size=hypconfig.testbatchsize, shuffle=False, num_workers=workernum, pin_memory=True)

        datasetloader = torchtools.dataSetLoader(trainloader, testloader, False, datamean, datastd)
        return datasetloader, datasets


if __name__ == '__main__':
    testfun()
