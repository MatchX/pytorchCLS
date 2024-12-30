# encoding: utf-8

import torchvision.transforms as transforms
from addict import Dict
import imgaug as ia
import imgaug.augmenters as iaa
import albumentations as albu
import numpy as np
import cv2
from PIL import Image
import platform
from matchx import configbase


class modelconfig(configbase.configBase):
    """
    配置文件
    """
    def __init__(self):
        super(modelconfig, self).__init__()
        self._makemodconfig()
        self._makedataconfig()
        self._makescheduleconfig()
        self._makehypconfig()

    def _makemodconfig(self):
        """
        生成模型配置文件
        """
        self.modcfg.train_type = "kitche_cigar"
        self.modcfg.trainflg = "V1"
        # resnet26d resnet34d resnet50d resnet101d resnet152d resnet200d  vit_base_patch16_224 vit_base_patch16_224_in21k regnety_032 regnety_120 vit_tiny_patch16_224_in21k
        self.modcfg.netname = "resnet34d" # "vit_tiny_patch16_224_in21k"

    def _makedataconfig(self):
        """
        生成数据配置文件
        """
        platformname = platform.system().lower()
        # 训练数据
        if platformname == 'windows':
            predir = "E:/WorkSpace/AIbase/pytorchCLS/dataset/ciga_classification/train/"
        else:  # linux
            predir = "/home/huangjunjie/datasets/chufang/ciga_classification/train/"
        self.datacfg.traindata = [
            predir + "20210803/",
            predir + "20211126/",
            predir + "20211129/",
            predir + "20220112/",
            predir + "20220113/",
            predir + "mpcap01/",
            predir + "v0/",
            predir + "20241227/",
            # predir + "20211126/",
        ]

        # 测试数据
        if platformname == 'windows':
            predir = "E:/WorkSpace/AIbase/pytorchCLS/dataset/ciga_classification/test/"
        else:  # linux
            predir = "/home/huangjunjie/datasets/chufang/ciga_classification/test/"
        self.datacfg.valdata = [
            predir + "20210803/",
        ]

        # 标签类型与对应目录配置
        self.datacfg.label_map = {"no_smoke": 0, "smoke": 1, "uncertain": 0}
        # 是否重新生成标签文本
        self.datacfg.rebuild_label = True

        # mean and std
        self.datacfg.data_mean = [0.485, 0.456, 0.406]
        self.datacfg.std_mean = [0.229, 0.224, 0.225]

        # number of classes
        label_set = set(self.datacfg.label_map.values())
        self.datacfg.num_class = len(label_set)

        # input data size
        self.datacfg.traininputsize = (224, 224)  # W,H
        self.datacfg.testinputsize = (224, 224)

        self.datacfg.bboxextsize = 1.15  # bbox扩张系数
        self.datacfg.lbbalance = True  # 标签均衡
        # bbox 各个方向抖动系数
        self.datacfg.bxtopratio = [-0.3, 0.3]
        self.datacfg.bxleftratio = [-0.3, 0.3]
        self.datacfg.bxbottomratio = [-0.3, 0.3]
        self.datacfg.bxrightratio = [-0.3, 0.3]

        # laoder config
        self.datacfg.droplast = True  # 丢弃最后的batch数据
        self.datacfg.workernum = 32  # 数据加载线程数
        self.datacfg.cacheimg = False  # 将训练图片放入内存中缓存
        self.datacfg.preloader = False  # use preloader load data
        self.datacfg.multi_epochs_loader = False  # use the multi-epochs-loader to save time at the beginning of every epoch

        # 训练数据处理流程
        self.datacfg.transformpipeline4train = Dict()
        # imgaug
        imgaugseg = iaa.Sequential([
            iaa.Affine(rotate=(-5, 5), mode='constant'),
            iaa.Fliplr(p=0.5),
            iaa.Cutout(nb_iterations=(1, 5), size=0.06, squared=False),
            iaa.CoarseDropout(p=0.02, size_percent=0.15, per_channel=0.5),
            iaa.Multiply(mul=(0.5, 1.0), per_channel=0.5),
            iaa.GaussianBlur(sigma=(0.0, 0.2)),
        ])
        self.datacfg.transformpipeline4train.imgaug = imgaugseg
        self.datacfg.imgaugseg_epoch_ratio = 0.8

        # albu
        albutransseq = albu.Compose([
            # albu.RandomRotate90(),
            # 翻转
            # albu.Flip(),
            # albu.Transpose(),
            albu.OneOf([
                # 高斯噪点
                #albu.IAAAdditiveGaussianNoise(),
                albu.GaussNoise(),
            ], p=0.2),
            albu.OneOf([
                # 模糊相关操作
                albu.MotionBlur(p=.2),
                albu.MedianBlur(blur_limit=3, p=0.1),
                albu.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=10, p=0.2),
            albu.OneOf([
                # 畸变相关操作
                albu.OpticalDistortion(p=0.3),
                albu.GridDistortion(p=.1),
                albu.PiecewiseAffine(p=0.3),
            ], p=0.2),
            albu.OneOf([
                # 锐化、浮雕等操作
                albu.CLAHE(clip_limit=2),
                albu.Sharpen(),
                albu.Emboss(),
                albu.RandomBrightnessContrast(),
            ], p=0.3),
            albu.HueSaturationValue(p=0.3),
        ], p=1.0)
        # self.datacfg.transformpipeline.albutrans = albutransseq

        # torchtrans
        # torchtransseq = transforms.Compose([
        #     transforms.Resize(size=self.datacfg.traininputsize),
        #     # transforms.RandomRotation(degrees=10),
        #     # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.3),
        #     # transforms.RandomChoice([
        #     #     transforms.RandomVerticalFlip(p=0.5),
        #     # ]),
        #     transforms.Normalize(mean=self.datacfg.data_mean, std=self.datacfg.std_mean),
        # ])
        colorJitter = (0.2, 0.2, 0.1, 0.1)
        torchtransseq = transforms.Compose([
            configbase.RandomCrop(0.1, 2.0, new_expansion_rate=0.5, to_bgr=False),  # set to_bgr=False
            transforms.Resize(self.datacfg.traininputsize),  # [H,W] format
            transforms.ColorJitter(*colorJitter),
            transforms.RandomRotation((-20, 20)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.datacfg.data_mean, std=self.datacfg.std_mean),
        ])
        self.datacfg.transformpipeline4train.torchtrans = torchtransseq

        # 测试数据处理流程
        self.datacfg.transformpipeline4test = Dict()
        torchtransseq = transforms.Compose([
            configbase.CenterCrop(1.5/3, to_bgr=False),  # set to_bgr=False
            transforms.Resize(self.datacfg.traininputsize),  # [H,W] format
            transforms.ToTensor(),
            transforms.Normalize(mean=self.datacfg.data_mean, std=self.datacfg.std_mean),
        ])
        self.datacfg.transformpipeline4test.torchtrans = torchtransseq

    def _makescheduleconfig(self):
        """
        生成策略配置文件
        """
        self.schedulecfg.optim_type = "sgd"  # adamw
        # torchtools.SoftTargetCrossEntropy  # torchtools.LabelSmoothingCrossEntropy # nn.CrossEntropyLoss
        self.schedulecfg.lossfun = "torchtools.LabelSmoothingCrossEntropy"
        self.schedulecfg.lr_schedule = "CosineLRScheduler"
        self.schedulecfg.base_lr = 0.01  # base learnrate
        self.schedulecfg.warm_up_epoch = 8  # the learning rate warm epoch
        self.schedulecfg.momentum = 0.9  # the optim momentum
        self.schedulecfg.weight_decay = 0.0005  # the optim decay

    def _makehypconfig(self):
        """
        生成超参数配置文件
        """
        self.hypcfg.log_root = "workdir/"  # log root dir
        self.hypcfg.seed = 89  # the init seed
        self.hypcfg.totalepoch = 60  # he tranin total epoch
        self.hypcfg.saverate = 0.5  # 相对于总训练批次 模型保存的比例
        self.hypcfg.trainbatchsize = 32  # the tranin batch size
        self.hypcfg.testbatchsize = 32  # the test batch size
        self.hypcfg.useamp = True  # use Native AMP for mixed precision training
        self.hypcfg.sync_bn = False  # bn层同步
        self.hypcfg.use_ema = True  # 使用指数移动平均
        self.hypcfg.ema_decay = 0.9999
        self.hypcfg.label_smooth = 0.1
        self.hypcfg.baddatafactor = 0.01  # 误报惩罚系数
        self.hypcfg.lossdatafactor = 0.008  # 漏报惩罚系数
        self.hypcfg.trainmarker = "训练说明"

    def __str__(self):
        infostrlist = "************************************ model setting ************************************\n"
        infostrlist += ("* {:10}{}".format("modcfg", self.modcfg))
        infostrlist += ("\n* {:10}{}".format("datacfg", self.datacfg))
        infostrlist += ("\n* {:10}{}".format("schedulecfg", self.schedulecfg))
        infostrlist += ("\n* {:10}{}".format("hypcfg", self.hypcfg))
        return infostrlist


if __name__ == '__main__':
    import importlib
    cfgfile = "kitchen_cigar_configv1"
    modelconfig = None
    if type(cfgfile) is str:
        configpack = importlib.import_module(cfgfile)
        modelconfig = configpack.modelconfig()
    print(modelconfig)
