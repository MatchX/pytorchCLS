# encoding: utf-8

"""
模型训练器
python=3.7
pytorch=1.6

tensorboard --logdir=logs/summary_

anaconda源
清华源
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/menpo/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/peterjc123/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
中科大源
channels:
  - https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
  - https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
  - defaults
show_channel_urls: true
ssl_verify: true

安装第三方库
pip install -i https://mirrors.aliyun.com/pypi/simple --upgrade opencv-python
pip install -i https://mirrors.aliyun.com/pypi/simple --upgrade tensorwatch

安装指定版本tensorflow
pip install -i https://mirrors.aliyun.com/pypi/simple --upgrade tensorflow-gpu==2.0.0
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ --upgrade tensorflow-gpu==1.13.1
pip install -i https://pypi.doubanio.com/simple --upgrade tensorflow-gpu==2.2.0
pip install -i https://mirrors.aliyun.com/pypi/simple --upgrade tensorflow-gpu==2.2.0

#多卡训练脚本
python -m torch.distributed.launch --nproc_per_node 2 train.py --device 0,1
查看显卡使用状态
watch -n1 nvidia-smi

conda create -n detectron2 python=3.7
"""
import numpy as np

from matchx.matchutils import *
import torch
# import tensorwatch as tw
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.nn.init as init
from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
import cv2
import yaml
from tqdm import tqdm
from shutil import copyfile
import argparse
from matchx import torchtools
from addict import Dict
from pathlib import Path
import datetime
from .torch_utils import ModelEMAV2
from copy import deepcopy
# from skimage import io


def tensorboarddemo():
    writer = SummaryWriter('logs/summary_')

    resnet18 = torchvision.models.resnet18(False)
    sample_rate = 44100
    freqs = [262, 294, 330, 349, 392, 440, 440, 440, 440, 440, 440]

    for n_iter in range(100):

        dummy_s1 = torch.rand(1)
        dummy_s2 = torch.rand(1)
        # data grouping by `slash`
        writer.add_scalar('data/scalar1', dummy_s1[0], n_iter)
        writer.add_scalar('data/scalar2', dummy_s2[0], n_iter)

        writer.add_scalars('data/scalar_group', {'xsinx': n_iter * np.sin(n_iter),
                                                 'xcosx': n_iter * np.cos(n_iter),
                                                 'arctanx': np.arctan(n_iter)}, n_iter)

        dummy_img = torch.rand(32, 3, 64, 64)  # output from network
        if n_iter % 10 == 0:
            x = torchvision.utils.make_grid(dummy_img, normalize=True, scale_each=True)
            writer.add_image('Image', x, n_iter)

            dummy_audio = torch.zeros(sample_rate * 2)
            for i in range(x.size()[0]):
                # amplitude of sound should in [-1, 1]
                dummy_audio[i] = np.cos(freqs[n_iter // 10] * np.pi * float(i) / float(sample_rate))
            # writer.add_audio('myAudio', dummy_audio, n_iter, sample_rate=sample_rate)

            writer.add_text('Text', 'text logged at step:' + str(n_iter), n_iter)

            for name, param in resnet18.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), n_iter)

            # needs tensorboard 0.4RC or later
            writer.add_pr_curve('xoxo', np.random.randint(2, size=100), np.random.rand(100), n_iter)

    dataset = torchvision.datasets.MNIST('mnist', train=False, download=True)
    images = dataset.test_data[:100].float()
    label = dataset.test_labels[:100]

    features = images.view(100, 784)
    writer.add_embedding(features, metadata=label, label_img=images.unsqueeze(1))

    # export scalar data to JSON for external processing
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()


class Layerhooker(object):
    def __init__(self, model, layer_num):
        self._hook = model[layer_num].register_forward_hook(self.hook_fn)
        self.infeatures = None
        self.outfeatures = None

    def hook_fn(self, module, input, output):
        # self.infeatures = input[0].cpu().data.numpy()
        self.outfeatures = output.cpu().data.numpy()

    def remove(self):
        self._hook.remove()


class ModelTrainerbase(object):
    """
    模型训练类
    """

    def __init__(self, modelconfig, opt, dataset, device, workdir):
        self.argument = opt
        self.modelconfig = modelconfig
        self.dataset = dataset

        # yaml to Dict
        self.modelcfgdict = self.modelconfig
        self.dataconfig = self.modelcfgdict.datacfg
        logging.info(self.modelcfgdict)

        # Save run settings
        if self.argument.global_rank in [-1, 0]:
            basedir = Path(workdir)
            # with open(basedir / 'model.yaml', 'w', encoding='utf-8') as f:
            #     yaml.dump(self.modelconfig, f, sort_keys=False, allow_unicode=True)
            with open(basedir / 'opt.yaml', 'w', encoding='utf-8') as f:
                yaml.dump(vars(self.argument), f, sort_keys=False, allow_unicode=True)

        # 参数
        self.EPOCHs = self.modelcfgdict.hypcfg.totalepoch
        self.MILESTONES = self.modelcfgdict.hypcfg.MILESTONES
        if self.argument.global_rank in [-1, 0]:
            self.checkpointdir = workdir + "/checkpoint/"
            Utiltool.makedir(self.checkpointdir)
            self.logdir = workdir + "/logs/"
        self.startepoch = 0
        self.label_smooth = self.modelcfgdict.hypcfg.get("label_smooth", 0.0)
        self.saverate = self.modelcfgdict.hypcfg.saverate  # 当训练进度达到该比例时开始保存模型
        # 使用并行数据加载
        self.cpucorenum = multiprocessing.cpu_count()
        self.dataParallel = False

        # 生成网络
        net = self.makenet()

        # 选择设备
        self.device = device
        # cudnn.enabled = True  # 启用cudnn
        # cudnn.benchmark = True  # 优化卷积实现算法加快速度 只适用于输入尺寸不变的网络
        self.cudadevcount = torch.cuda.device_count()
        if self.argument.global_rank in [-1, 0]:
            devcount = "cudadevice_count=" + str(self.cudadevcount)
            print(devcount)
            logging.info(devcount)
        if self.argument.global_rank == -1 and self.cudadevcount > 1:
            # 多显卡时开启并行数据处理
            self.dataParallel = True

        # 混合精度训练设置
        self.ampscaler = torch.cuda.amp.GradScaler(enabled=self.modelcfgdict.hypcfg.useamp)

        self.trainnet = net.to(device=self.device)
        if self.modelcfgdict.hypcfg.sync_bn:
            self.trainnet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.trainnet)
        # nowtimestr = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
        if self.argument.global_rank in [-1, 0]:
            self.sumwriter = SummaryWriter(self.logdir + '/summary_')
            drawgraph = False
            if drawgraph:
                input_tensor = torch.Tensor(1, 3, self.modelcfgdict.datacfg.traininputsize[1], self.modelcfgdict.datacfg.traininputsize[0]).cuda()
                self.sumwriter.add_graph(self.trainnet, input_tensor)

        # 初始化优化器
        self.optimizer = self.makeoptimizer()

        # 学习率衰减器
        self.train_scheduler = self.makescheduler()

        # 损失函数
        self.loss_function = self.makelossfunction()

        # EMA
        self.model_ema = None
        if self.modelcfgdict.hypcfg.use_ema and self.argument.global_rank in [-1, 0]:
            self.model_ema = ModelEMAV2(self.trainnet, decay=self.modelcfgdict.hypcfg.ema_decay)

        # 获取训练检查点
        if self.argument.global_rank in [-1, 0]:
            ckptfile = self.getlastcheckpt(self.checkpointdir)
            # epoch = 1
            # ckptfile = "D:/AIbase/nnetdemo/dnnplayer/pytorch-image-models/run/facequ_15_vit_base_patch16_224_in21k/checkpoint/epoch_11_85.07.ptcp"
            if ckptfile is not None:
                # 有检查点
                # self.trainnet = torch.load(ckptfile)  # 加载整个网络
                checkpoint = torch.load(ckptfile)
                self.trainnet.load_state_dict(checkpoint['model_state_dict'].float().state_dict())  # 加载模型可学习参数
                if self.model_ema is not None and checkpoint['ema'] is not None:
                    self.model_ema.ema.load_state_dict(checkpoint['ema'].float().state_dict())  # 加载ema
                    self.model_ema.updates = checkpoint['ema_updates']  #
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器参数
                self.train_scheduler.load_state_dict(checkpoint['train_scheduler'])  # 加载学习率调度器
                if 'ampscaler' in checkpoint.keys():
                    self.ampscaler.load_state_dict(checkpoint['ampscaler'])  # 加载amp
                last_Acc = checkpoint['testacc']
                logging.info("last_acc=" + str(last_Acc))
                self.startepoch = checkpoint['epoch']  # 设置开始的epoch
                self.startepoch += 1  # 开始下一轮训练
                logging.info("train resume")
            else:
                # 没有检查点
                pass

        # 设置模型运行设备
        if self.dataParallel:
            self.trainnet = torch.nn.DataParallel(self.trainnet)

        # DDP mode
        if self.argument.global_rank != -1:
            self.trainnet = DDP(self.trainnet, device_ids=[self.argument.local_rank], output_device=self.argument.local_rank)

        # 记录网络层权重值
        self.showlayerweight = False

        # 生成网络0层输入hook
        self.showfeaturemap = False  # 显示网络特征可视化图
        if self.showfeaturemap:
            self.conv_out0 = Layerhooker(self.trainnet.features, 0)

    def makenet(self):
        pass

    def makeoptimizer(self):
        pass

    def makescheduler(self):
        pass

    def makelossfunction(self):
        pass

    def _train_epoch(self, epoch, lr=None):
        pass

    def evaluate(self, epoch) -> float:
        pass

    def _train_step(self, epoch, lr=None):
        self.trainnet.train()
        if lr is not None:
            self.optimizer.param_groups[0]['lr'] = lr
        self.train_scheduler.step(epoch)
        if self.argument.global_rank in [-1, 0]:
            self.sumwriter.add_scalar('data/lr', self.optimizer.param_groups[0]['lr'], epoch)
        self._train_epoch(epoch, lr)

    @staticmethod
    def get_mean_and_std(dataset):
        """Compute the mean and std value of dataset."""
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
        mean = torch.zeros(3)
        std = torch.zeros(3)
        print('==> Computing mean and std..')
        for inputs, targets in dataloader:
            for i in range(3):
                mean[i] += inputs[:, i, :, :].mean()
                std[i] += inputs[:, i, :, :].std()
        mean.div_(len(dataset))
        std.div_(len(dataset))
        return mean, std

    @staticmethod
    def init_params(net):
        """Init layer parameters."""
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=1e-3)
                if m.bias:
                    init.constant(m.bias, 0)

    @staticmethod
    def getlastcheckpt(checkpointdir):
        """
        find and feedback the latest ckpt in the folder
        """
        # all_checkpoints = os.listdir(checkpointdir)
        # max_epoch = -1
        # max_epoch_path = None
        # for i in range(len(all_checkpoints)):
        #     if all_checkpoints[i].endswith(".ptcp"):
        #         index = int(os.path.splitext(all_checkpoints[i])[0].split('_')[1])
        #         if index > max_epoch:
        #             max_epoch = index
        #             max_epoch_path = all_checkpoints[i]
        # if max_epoch >= 0:
        #     logging.info("load checkpoint=" + max_epoch_path)
        #     print("load checkpoint!")
        #     return os.path.join(checkpointdir, max_epoch_path), max_epoch
        # else:
        #     return None, max_epoch

        lastpt = checkpointdir + "/last.ptcp"
        if os.path.exists(lastpt):
            return lastpt
        else:
            return None

    @staticmethod
    def is_parallel(model):
        return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

    def savecheckpt(self, epoch, testacc, savetype=0):
        """
        Args:
            epoch:
            testacc:
            savetype: 保存类型 0=普通保存 1=最后训练节点 2=最好训练节点

        Returns:
        """
        fewafe = self.is_parallel(self.trainnet)
        filename = self.checkpointdir + "epoch_%d_%.2f.ptcp" % (epoch, testacc)
        if 1 == savetype:
            filename = self.checkpointdir + "last.ptcp"
            checkpoint = {"model_state_dict": self.trainnet.state_dict(),
                          # "model_state_dict": deepcopy(self.trainnet.module if self.is_parallel(self.trainnet) else self.trainnet),  # .half(),
                          "ema": deepcopy(self.model_ema.ema).half() if self.model_ema else None,
                          "ema_updates": self.model_ema.updates if self.model_ema else None,
                          "optimizer_state_dict": self.optimizer.state_dict(),
                          "train_scheduler": self.train_scheduler.state_dict(),
                          "ampscaler": self.ampscaler.state_dict(),
                          "epoch": epoch,
                          "testacc": testacc,
                          "islastflg": True
                          }
        elif 2 == savetype:
            filename = self.checkpointdir + "best.ptcp"
            checkpoint = {"model_state_dict": self.trainnet.state_dict(),
                          "epoch": epoch,
                          "testacc": testacc
                          }
        else:
            checkpoint = {"model_state_dict": self.trainnet.state_dict(),
                          "ema": deepcopy(self.model_ema.ema).half() if self.model_ema else None,
                          "ema_updates": self.model_ema.updates if self.model_ema else None,
                          "optimizer_state_dict": self.optimizer.state_dict(),
                          "train_scheduler": self.train_scheduler.state_dict(),
                          "ampscaler": self.ampscaler.state_dict(),
                          "epoch": epoch,
                          "testacc": testacc
                          }
        # torch.save(self.trainnet, filename)  # 保存整个网络
        Utiltool.makedir(self.checkpointdir)
        torch.save(checkpoint, filename)  # 保存网络参数
        if 0 == savetype:
            logging.info("Epoch_save_" + str(epoch) + " acc=" + str(round(testacc, 4)))

    def train(self):
        # 生成学习率列表
        # print(self.trainnet.state_dict().keys())
        start_time = datetime.datetime.now()
        logging.info("start train model")
        logging.info(str(self.trainnet))
        # lrlist = []
        epoch = self.startepoch
        # 生成学习率
        # for it in range(self.EPOCHs):
        #     lr = NNtool.get_cos_learning_rate(it + 1, self.modelcfgdict.hypcfg.warm_up_epoch, self.modelcfgdict.hypcfg.totalepoch, self.modelcfgdict.hypcfg.learning_rate)
        #     lrlist.append(lr)
        #     it += 1
        # 开始迭代训练
        max_ACC = 0.0
        # epoch = 0
        while epoch < self.EPOCHs:
            # try:
            torch.cuda.empty_cache()
            self._train_step(epoch)
            torch.cuda.empty_cache()
            if self.argument.global_rank in [-1, 0]:
                testacc = self.evaluate(epoch)
                self.savecheckpt(epoch, testacc, 1)
                if epoch / self.EPOCHs >= self.saverate:
                    self.savecheckpt(epoch, testacc)
                if max_ACC < testacc:
                    self.savecheckpt(epoch, testacc, 2)
                    max_ACC = testacc
                # except Exception as ex:
                #     tqdm.write(str(ex))
                #     # print(ex.__traceback__.tb_frame.f_globals["__file__"])
                #     # print(ex.__traceback__.tb_lineno)
                #     logging.info("exception=" + str(ex))
                #     break
                # 记录训练数据
                for name, param in self.trainnet.named_parameters():
                    self.sumwriter.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
                    # if "conv" in name or "weight" in name:
                    #     if 4 != len(param.size()):
                    #         continue
                    #     in_channels = param.size()[1]
                    #     out_channels = param.size()[0]
                    #     k_w, k_h = param.size()[3], param.size()[2]  # 卷积核大小
                    #     kernel_all = param.view(-1, 1, k_w, k_h)  # 每个通道的卷积核
                    #     kernel_grid = torchvision.utils.make_grid(kernel_all, normalize=True, scale_each=True, nrow=in_channels)
                    #     self.sumwriter.add_image(f'{name}_all', kernel_grid, global_step=0)

                self.sumwriter.flush()
                epoch += 1
        if self.argument.global_rank in [-1, 0]:
            self.sumwriter.export_scalars_to_json(self.logdir + "/summary_/all_scalars.json")
            self.sumwriter.close()
            logging.info("max_ACC=" + str(max_ACC))
            # tqdm.write("max_ACC=", max_ACC)
            print("max_ACC=", max_ACC)
            end_time = datetime.datetime.now()
            usetimestr = Utiltool.convert_timedelta(end_time - start_time)
            print("use time=" + usetimestr)
            logging.info("use time=" + usetimestr)
            logging.info("finish train model")


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # dataset = datasetMaker.makedataset(params["train_batch_size"], params["test_batch_size"])
    # resnet18 = torchvision.models.resnet18(False)
    # model = ModelTrainer(resnet18, params)
    # x = torch.ones((1, 3, 32, 32))
    # x = x.to(device)
    # model.trainnet.forward(x)
    print("torch_Base test finished")
