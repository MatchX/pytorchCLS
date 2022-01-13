# encoding: utf-8

"""
训练配置基础类
"""

from addict import Dict
import torch.nn as nn
import numpy as np
from PIL import Image


class RandomCrop(nn.Module):
    def __init__(self, random_rate=0.15, expansion_rate=2.0,
                 new_expansion_rate=1.0,
                 to_bgr=False):
        super().__init__()
        assert random_rate < expansion_rate
        self.random_rate = random_rate
        self.expansion_rate = expansion_rate
        self.to_bgr = to_bgr
        self.new_expansion_rate = new_expansion_rate

    def forward(self, img):
        img = np.array(img)
        raw_h, raw_w = img.shape[:2]
        bbox_h = raw_h / (1 + self.expansion_rate)
        bbox_w = raw_w / (1 + self.expansion_rate)

        delta_xmin = np.random.uniform(-self.random_rate, self.random_rate) * bbox_w
        delta_xmax = np.random.uniform(-self.random_rate, self.random_rate) * bbox_w

        delta_ymin = np.random.uniform(-self.random_rate, self.random_rate) * bbox_h
        delta_ymax = np.random.uniform(-self.random_rate, self.random_rate) * bbox_h

        ymin = int((raw_h - bbox_h) / 2 + delta_ymin - self.new_expansion_rate * bbox_h / 2)
        ymin = max(0, ymin)

        ymax = int((raw_h + bbox_h) / 2 + delta_ymax + self.new_expansion_rate * bbox_h / 2)

        xmin = int((raw_w - bbox_w) / 2 + delta_xmin - self.new_expansion_rate * bbox_w / 2)
        xmax = int((raw_w + bbox_w) / 2 + delta_xmax + self.new_expansion_rate * bbox_w / 2)
        img = img[ymin:ymax, xmin:xmax, :]
        if self.to_bgr:
            img = img[..., ::-1]
        return Image.fromarray(img)


class CenterCrop(nn.Module):
    """
    scale_rate=裁剪后的保留比例
    """
    def __init__(self, scale_rate, to_bgr=False):
        super().__init__()
        assert scale_rate < 1
        self.scale_rate = scale_rate
        self.diffrate = (1 - self.scale_rate) * 0.5
        self.to_bgr = to_bgr

    def forward(self, img):
        img = np.array(img)
        raw_h, raw_w = img.shape[:2]
        ymin = int(raw_h * self.diffrate)
        ymax = int(raw_h * (1 - self.diffrate))
        xmin = int(raw_w * self.diffrate)
        xmax = int(raw_w * (1 - self.diffrate))
        img = img[ymin:ymax, xmin:xmax, :]
        if self.to_bgr:
            img = img[..., ::-1]
        return Image.fromarray(img)


class configBase(object):
    """
    配置基类
    """
    def __init__(self):
        """
        模型配置文件
        """
        self.modcfg = Dict()
        self.modcfg.train_type = "baseconfig"
        self.modcfg.trainflg = "V1"
        self.modcfg.netname = "modelname"

        # 数据配置
        self.datacfg = Dict()
        # 策略配置
        self.schedulecfg = Dict()
        # 超参数配置
        self.hypcfg = Dict()

        self._makemodconfig()
        self._makedataconfig()
        self._makescheduleconfig()
        self._makehypconfig()

    def _makemodconfig(self):
        pass

    def _makedataconfig(self):
        pass

    def _makescheduleconfig(self):
        pass

    def _makehypconfig(self):
        pass

    def __str__(self):
        pass


if __name__ == '__main__':
    import importlib
    cfgfile = "mode_road_21k"
    modelconfig = None
    if type(cfgfile) is str:
        configpack = importlib.import_module(cfgfile)
        modelconfig = configpack.modelconfig()
    print(modelconfig)
