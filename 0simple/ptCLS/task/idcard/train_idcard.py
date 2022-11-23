#!/usr/bin/env python3
""" ImageNet Training Script
"""
import os
import argparse
import time

import timm

from idcard_trainer import ModelTrainer


def exportmodel(modeltrainer):
    basedir = "./"
    weight_path = basedir + "last.ptcp"
    outpath = "./last.onnx"
    modeltrainer.convert2onnx(weight_path, outpath)


def starttrain():
    if os.path.isabs(__file__):
        premodule = ""
    else:
        premodule = __file__.split(os.path.basename(__file__))[0].replace('.', '').replace('/', '.')
    # 创建训练器
    modeltrainer = ModelTrainer.inittrainer(premodule + "configs.idcard_configv1")
    # modeltrainer.classdemo()
    modeltrainer.train()
    # 导出模型
    # exportmodel(modeltrainer)


def evalmodel():
    import matchx
    gt_val_json = "D:/dataset/road/images/val.json"
    pred_val_json = "./yoloout.json"
    matchx.matchutils.Utiltool.cocoeval(gt_val_json, pred_val_json)


# python task/kitchen_hat/train_hat.py --device 0
# python -m torch.distributed.launch --nproc_per_node 4 trainer/kitchen_cigar/train.py --device 0,1,2,3
if __name__ == '__main__':
    aa = timm.list_models()
    print(aa)
    starttrain()
    print("finish train")
