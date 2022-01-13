#!/usr/bin/env python3
""" ImageNet Training Script
"""
import os
import argparse
import time
from cigar_trainer import ModelTrainer


def exportmodel(modeltrainer):
    basedir = "./"
    weight_path = basedir + "epoch_119_-1.00.ptcp"
    outpath = "./kitchen_cigar_2022.12.13b.onnx"
    modeltrainer.convert2onnx(weight_path, outpath)


def starttrain():
    if os.path.isabs(__file__):
        premodule = ""
    else:
        premodule = __file__.split(os.path.basename(__file__))[0].replace('.', '').replace('/', '.')
    # 创建训练器
    modeltrainer = ModelTrainer.inittrainer(premodule + "configs.kitchen_cigar_configv1")
    # modeltrainer.classdemo()
    # modeltrainer.train()
    # 导出模型
    exportmodel(modeltrainer)


def evalmodel():
    import matchx
    gt_val_json = "D:/dataset/road/images/val.json"
    pred_val_json = "./yoloout.json"
    matchx.matchutils.Utiltool.cocoeval(gt_val_json, pred_val_json)


# python task/kitchen_hat/train_hat.py --device 0
# python -m torch.distributed.launch --nproc_per_node 4 trainer/kitchen_cigar/train.py --device 0,1,2,3
if __name__ == '__main__':
    starttrain()
    print("finish train")
