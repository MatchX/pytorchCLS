#!/usr/bin/env python3
""" ImageNet Training Script
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
import time
from hat_trainer import ModelTrainer


def exportmodel(modeltrainer):
    basedir = "./"
    weight_path = basedir + "epoch_89_97.78.ptcp"
    outpath = "./kitchen_hat_22_01_18a.onnx"
    modeltrainer.convert2onnx(weight_path, outpath)


def starttrain():
    if os.path.isabs(__file__):
        premodule = ""
    else:
        premodule = __file__.split(os.path.basename(__file__))[0].replace('.', '').replace('/', '.')
    # 创建训练器
    modeltrainer = ModelTrainer.inittrainer(premodule + "configs.kitchen_hat_configv3")
    # modeltrainer.classdemo()
    modeltrainer.train()
    # 导出模型
    # exportmodel(modeltrainer)


def evalmodel():
    import matchx
    gt_val_json = "D:/dataset/road/images/val.json"
    pred_val_json = "./yoloout.json"
    matchx.matchutils.Utiltool.cocoeval(gt_val_json, pred_val_json)


"""
python task/kitchen_hat/train_hat.py --device 0
"""
if __name__ == '__main__':
    starttrain()
    print("finish train")
