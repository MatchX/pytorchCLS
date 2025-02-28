#!/usr/bin/env python3
""" ImageNet Training Script
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
import time
from mask_trainer import ModelTrainer


def exportmodel(modeltrainer):
    basedir = "./"
    weight_path = basedir + "epoch_51_89.14.ptcp"
    outpath = "./kitchen_mask_2021.08.24.onnx"
    modeltrainer.convert2onnx(weight_path, outpath)


def starttrain():
    if os.path.isabs(__file__):
        premodule = ""
    else:
        premodule = __file__.split(os.path.basename(__file__))[0].replace('.', '').replace('/', '.')
    # 创建训练器
    modeltrainer = ModelTrainer.inittrainer(premodule + "configs.kitchen_mask_configv1")
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
python -m torch.distributed.launch --nproc_per_node 4 task/kitchen_mask/train_mask.py --device 0,1,2,3
python task/kitchen_mask/train_mask.py --device 0
fuser -v /dev/nvidia*
"""
if __name__ == '__main__':
    # import os
    # import shutil
    #
    # def list_files_in_directory(directory):
    #     filelist = []
    #     for root, dirs, files in os.walk(directory):
    #         for file in files:
    #             filelist.append(file)
    #             # print(os.path.join(root, file))
    #     return filelist
    #
    # dir01 = R"F:\0match\AIdata\mask\gt_2025.2.14\01\no"
    # dir02 = R"F:\0match\AIdata\mask\gt_2025.2.14\01\no_out"
    # outdir = R"F:\0match\AIdata\mask\gt_2025.2.14\01\0out"
    #
    # flist01 = list_files_in_directory(dir01)
    # flist02 = list_files_in_directory(dir02)
    # foutlist = [item for item in flist01 if item not in flist02]
    # for item in foutlist:
    #     imgfile = os.path.join(dir01, item)
    #     outfile = os.path.join(outdir, item)
    #     shutil.copy(imgfile, outfile)
    starttrain()
    print("finish train")
