# encoding: utf-8
"""
version=0.1
Match工具包
Utiltool : 工具类包含各种常用工具 比如日志 状态条等
dataMaker : 数据生成工具 采用多进程方式生成数据
"""

import os
import math
import sys
import shutil
import logging
import time
import sys
import ctypes
import multiprocessing
import numpy as np
import inspect
import matplotlib.pyplot as plt
import imgaug as ia
import imgaug.augmenters as iaa
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

"""
开发版本为 python3.7
"""
assert (sys.version_info.major == 3 and sys.version_info.minor >= 7)


class Utiltool(object):
    """
    工具类
    """

    # static parame
    BAR_TERM_WIDTH = 1
    TOTAL_BAR_LENGTH = 30.
    begin_time = None
    last_time = None
    usetotal_time = 0.
    logfile_handler = None

    def __init__(self):
        pass

    @staticmethod
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    @staticmethod
    def makedir(subdir):
        logdirpath = os.path.abspath(subdir)
        if not os.path.exists(logdirpath):
            os.makedirs(logdirpath)

    @staticmethod
    def initlogfile(logdir, prefixname=None):
        """
        日志初始化函数
        """
        nowtimestr = time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime(time.time()))
        if prefixname is None:
            prefixname = nowtimestr
        logdirpath = os.path.abspath(logdir)
        if not os.path.exists(logdirpath):
            os.makedirs(logdirpath)

        logfilename = prefixname + ".log"
        logfullpath = os.path.join(logdirpath, logfilename)
        # logging.basicConfig(level=logging.NOTSET,  # 输出所有日志
        #                     format='%(levelname)s %(asctime)s %(filename)s[line:%(lineno)d]  %(message)s',
        #                     datefmt='%Y-%m-%d %H:%M:%S',
        #                     filename=logfullpath,
        #                     filemode='a')
        # logging.info("start_log at " + nowtimestr)

        logger = logging.getLogger()
        logger.setLevel(level=logging.NOTSET)

        if Utiltool.logfile_handler is not None:
            logger.removeHandler(Utiltool.logfile_handler)
        formatter = '%(levelname)s %(asctime)s %(filename)s[line:%(lineno)d]  %(message)s'
        datefmt = '%Y-%m-%d %H:%M:%S'
        fmt = logging.Formatter(formatter, datefmt)
        file_handler = logging.FileHandler(logfullpath, 'a')
        file_handler.setLevel(level=logging.NOTSET)
        # log_formatter = logging.Formatter(formatter)
        # file_handler.setFormatter(log_formatter)
        file_handler.setFormatter(fmt)
        Utiltool.logfile_handler = file_handler
        logger.addHandler(file_handler)

        # print info on console
        # console_handler = logging.StreamHandler()
        # console_handler.setLevel(level=logging.NOTSET)
        # console_formatter = logging.Formatter(formatter)
        # console_handler.setFormatter(console_formatter)
        # logger.addHandler(console_handler)

        logging.info("start_log at " + nowtimestr)

    @staticmethod
    def format_time(seconds):
        days = int(seconds / 3600 / 24)
        seconds = seconds - days * 3600 * 24
        hours = int(seconds / 3600)
        seconds = seconds - hours * 3600
        minutes = int(seconds / 60)
        seconds = seconds - minutes * 60
        secondsf = int(seconds)
        seconds = seconds - secondsf
        millis = int(seconds * 1000)

        f = ''
        i = 1
        if days > 0:
            f += str(days) + 'D'
            i += 1
        if hours > 0 and i <= 2:
            f += str(hours) + 'h'
            i += 1
        if minutes > 0 and i <= 2:
            f += str(minutes) + 'm'
            i += 1
        if secondsf > 0 and i <= 2:
            f += str(secondsf) + 's'
            i += 1
        if millis > 0 and i <= 2:
            f += str(millis) + 'ms'
            i += 1
        if f == '':
            f = '0ms'
        return f

    @staticmethod
    def progress_bar(current, total, msg=None, forceflush=False):
        """
        状态条
        """
        # if updatetime:
        #     last_time = time.time()
        #     begin_time = last_time

        if current == 0 or Utiltool.begin_time is None:
            Utiltool.begin_time = time.time()  # Reset for new bar.
            Utiltool.last_time = Utiltool.begin_time

        cur_len = int(Utiltool.TOTAL_BAR_LENGTH * current / total)
        rest_len = int(Utiltool.TOTAL_BAR_LENGTH - cur_len) - 1

        sys.stdout.write('\r[')
        for i in range(cur_len):
            sys.stdout.write('=')
        sys.stdout.write('>')
        for i in range(rest_len):
            sys.stdout.write('.')
        sys.stdout.write(']')

        cur_time = time.time()
        step_time = cur_time - Utiltool.last_time
        Utiltool.usetotal_time += step_time
        Utiltool.last_time = cur_time
        tot_time = cur_time - Utiltool.begin_time

        remain_time = Utiltool.usetotal_time / (current + 1) * (total - (current + 1))
        L = ['Step:%s' % Utiltool.format_time(step_time), ' Tot:%s' % Utiltool.format_time(tot_time), "<%s" % Utiltool.format_time(remain_time)]
        if msg:
            L.append('|' + msg)

        msg = ''.join(L)
        sys.stdout.write(msg)

        for i in range(Utiltool.BAR_TERM_WIDTH - int(Utiltool.TOTAL_BAR_LENGTH) - len(msg) - 3):
            sys.stdout.write(' ')

        # Go back to the center of the bar.
        for i in range(Utiltool.BAR_TERM_WIDTH - int(Utiltool.TOTAL_BAR_LENGTH / 2) + 2):
            sys.stdout.write('\b')

        sys.stdout.write(' [%.2f%%|%d/%d]' % ((current + 1) / total * 100.0, current + 1, total))

        if current + 1 >= total:
            Utiltool.usetotal_time = 0
            sys.stdout.write('\n')
        if forceflush:
            sys.stdout.flush()

    @staticmethod
    def print_config(params):
        print("************************************ model setting ************************************")
        for k, v in params.items():
            print("* {:10}{}".format(k, v))

    @staticmethod
    def cocoeval(gt_val_json, pred_val_json):
        gt = COCO(gt_val_json)
        predict = gt.loadRes(pred_val_json)
        evaluator = COCOeval(gt, predict, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    @staticmethod
    def convert_timedelta(duration) -> str:
        days, seconds = duration.days, duration.seconds
        hours = days * 24 + seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = (seconds % 60)
        return str(days) + ":" + str(hours) + ":" + str(minutes) + ":" + str(seconds)

    @staticmethod
    def init_pytorch_env(configpath):
        import matchx
        import argparse
        import cv2
        import importlib
        from shutil import copyfile
        import torch
        import torch.distributed as dist

        # 设置加速状态
        cpucorenum = multiprocessing.cpu_count()
        cv2.setNumThreads(max(2, cpucorenum - 2))
        cv2.ocl.setUseOpenCL(True)
        cv2.setUseOptimized(True)

        parser = argparse.ArgumentParser()
        parser.add_argument('--cfg', type=str, default=configpath, help='config file path')
        parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
        parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
        opt = parser.parse_args()

        # get DDP env
        opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
        opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1

        # modelconfig
        modelconfig = None
        if type(opt.cfg) is str:
            # with open(opt.cfg, 'r', encoding='utf-8') as configfile:
            #     modelconfig = yaml.load(configfile, Loader=yaml.FullLoader)
            # model_cfgdict = Dict(modelconfig)
            configpack = importlib.import_module(opt.cfg)
            modelconfig = configpack.modelconfig()
        else:
            raise Exception("Invalid config file {}!".format(opt.cfg))

        # init log
        workdir = None
        if opt.global_rank in [-1, 0]:
            modelname = f'{modelconfig.modcfg.train_type}_{modelconfig.modcfg.netname}_{modelconfig.modcfg.trainflg}'
            workdir = modelconfig.hypcfg.log_root + modelname
            if opt.resume is None or not opt.resume:
                extname = 1
                if os.path.exists(workdir):
                    while True:
                        newworkdir = workdir + "_" + str(extname)
                        extname += 1
                        if not os.path.exists(newworkdir):
                            workdir = newworkdir
                            break
                # if os.path.exists(workdir):
                #     shutil.rmtree(workdir)
            logdir = workdir + "/logs/"
            matchx.matchutils.Utiltool.initlogfile(logdir)
            logging.info(opt)
            print(modelconfig)
            # matchx.matchutils.Utiltool.print_config(modelconfig)
            # 备份配置文件
            # cfg = opt.cfg.replace('.', '/') + ".py"
            cfg = configpack.__file__
            # cfgfilename = os.path.basename(cfg)
            # copyfile(cfg, workdir + "/" + cfgfilename)
            copyfile(cfg, workdir + "/" + "modelcfg.py")

        # Set device
        opt.total_batch_size = modelconfig.hypcfg.trainbatchsize
        assert modelconfig.hypcfg.trainbatchsize > 0
        device = matchx.torchUtils.select_device(opt.device, modelconfig.hypcfg.trainbatchsize)
        if opt.local_rank != -1:
            assert torch.cuda.device_count() > opt.local_rank
            torch.cuda.set_device(opt.local_rank)
            device = torch.device('cuda', opt.local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
            assert modelconfig.hypcfg.trainbatchsize % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
            opt.train_batch = opt.total_batch_size // opt.world_size

        # 指定随机种子保证训练可复现
        matchx.torchUtils.init_seeds(True, modelconfig.hypcfg.seed + opt.global_rank)
        return modelconfig, opt, device, workdir


class NNtool(object):
    """
    网络工具类
    """

    def __init__(self):
        pass

    @staticmethod
    def get_cos_learning_rate(cur_step, warm_up_steps, total_steps, cur_lr):
        """
        calculate the learning rate
        """
        if cur_step < warm_up_steps:
            lr = cur_lr * (cur_step / warm_up_steps)
        else:
            current_step = cur_step - warm_up_steps
            total_step = total_steps - warm_up_steps
            lr = cur_lr * 0.5 * (1 + math.cos(math.pi * current_step / total_step))
        return lr


def testfun():
    Utiltool.initlogfile("./")
    # logging.info("这是信息 at hhh")
    # logging.warning("这是警告 at hhh")
    # logging.error("这是错误 at hhh")


if __name__ == '__main__':
    testfun()
