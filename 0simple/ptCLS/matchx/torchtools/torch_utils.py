# encoding: utf-8

"""
version=0.1
PyTorch utils
"""

import logging
import math
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Dict
import sys
import json
import multiprocessing
import ctypes
import inspect

import imgaug as ia
import imgaug.augmenters as iaa

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import torch.nn.utils.prune as prune
import torchvision.transforms as transforms

"""
开发版本为 python3.7
"""
assert (sys.version_info.major == 3 and sys.version_info.minor >= 7)

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None
logger = logging.getLogger(__name__)


# torch 数据生成工具
class dataParallelMaker(object):
    """
    并行数据生成类
    """

    def __init__(self, datafun, funarg, pnum=2, buffsize=80):
        self.__buffersize = buffsize  # 缓存数据长度
        self.__datafun = datafun  # 数据生成函数
        self.__funarg = funarg  # 数据生成函数参数
        self.__pnum = max(2, pnum)  # 进程数量
        self.__processlist = []  # 进程
        self.__exitmark = multiprocessing.Value(ctypes.c_bool, False)  # 停止标记
        self.__datalock = multiprocessing.Lock()
        self.__mg = multiprocessing.Manager()
        self.__datalistA = self.__mg.list()
        self.__datalistB = self.__mg.list()
        self.__dataoutiter = 0
        self.__databucket = self.__mg.list([self.__datalistA, self.__datalistB])
        self.__bucketWriteriter = multiprocessing.Value(ctypes.c_int32, 0)
        self.__delaytimes = 0

        self.__initmaker()

    def __initmaker(self):
        for i in range(self.__pnum):
            p = multiprocessing.Process(target=self.dataworker,
                                        args=(self.__datalock, self.__datafun, self.__funarg, self.__databucket, self.__bucketWriteriter, self.__buffersize, self.__exitmark))
            p.start()
            self.__processlist.append(p)

    @staticmethod
    def dataworker(datalock, datafun, funarg, databucket, bucketiter, buffersize, exitmark):
        while not exitmark.value:
            try:
                if len(databucket[bucketiter.value]) >= buffersize:
                    time.sleep(0.1)  # 休眠100ms
                else:
                    data = datafun(*funarg)
                    datalock.acquire()
                    databucket[bucketiter.value].append(data)
                    datalock.release()
            except Exception as ex:
                frame = inspect.currentframe()
                print(__file__, "function=", frame.f_code.co_name, " line=", frame.f_lineno, " ex=", ex)
                time.sleep(1)

    def getdata(self):
        while True:
            if len(self.__databucket[1 - self.__bucketWriteriter.value]) > self.__dataoutiter:
                data = self.__databucket[1 - self.__bucketWriteriter.value][self.__dataoutiter]
                self.__dataoutiter += 1
                # self.__databucket[1 - self.__bucketWriteriter.value].pop(0)
                return data
            else:
                if len(self.__databucket[self.__bucketWriteriter.value]) > self.__buffersize * 0.2:
                    self.__datalock.acquire()
                    del self.__databucket[1 - self.__bucketWriteriter.value][:]
                    self.__bucketWriteriter.value = 1 - self.__bucketWriteriter.value
                    self.__datalock.release()
                    self.__dataoutiter = 0
                else:
                    time.sleep(0.1)
                    self.__delaytimes += 1
                    if self.__delaytimes % 100 == 0:
                        print("warning delaytimes is too often delaytimes=", self.__delaytimes)  # 如果频繁出现延迟 则可以提高进程数量或者增大缓存大小如果不差钱就换CPU

    def stop(self):
        if not self.__exitmark.value:
            self.__exitmark.value = True
            for p in self.__processlist:
                p.join()

    def getdelaytimes(self):
        return self.__delaytimes

    # demo
    # def creatdata(a1, d):
    #     a = a1
    #     b = "fdsafas"
    #     c = time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime(time.time()))
    #     # time.sleep(0.1)
    #     return a, b, c, d
    #
    # def testparallefun():
    #     t0 = time.process_time()
    #     datamaker = torchtools.dataParallelMaker(datafun=creatdata, funarg=(7, "fewa"), pnum=4, buffsize=100)
    #     t1 = time.process_time()
    #     print("usetime=", (t1 - t0) * 1000, "ms")
    #
    #     tn = 0
    #     while tn < 5000:
    #         outdata = datamaker.getdata()
    #         print(tn, outdata[0], outdata[1], outdata[2])
    #         tn += 1
    #     datamaker.stop()


class processParallelMaker(object):
    """
    并行任务生成类
    """

    def __init__(self, processfun, funarg, pnum=2, buffsize=80, sleeptime=0.1):
        self.__buffersize = buffsize  # 缓存数据长度
        self.__datafun = processfun  # 数据生成函数
        self.__funarg = funarg  # 数据生成函数参数
        self.__pnum = max(2, pnum)  # 进程数量
        self.__processlist = []  # 进程
        self.__exitmark = multiprocessing.Value(ctypes.c_bool, False)  # 停止标记
        # self.__datalock = multiprocessing.Lock()
        self.__databucket = multiprocessing.Queue(self.__buffersize)
        self.__sleeptime = sleeptime

        self.__initmaker()

    def __initmaker(self):
        for i in range(self.__pnum):
            p = multiprocessing.Process(target=self.dataworker,
                                        args=(self.__datafun, self.__funarg, self.__databucket, self.__sleeptime, self.__exitmark))
            p.start()
            self.__processlist.append(p)

    @staticmethod
    def dataworker(datafun, funarg, databucket, sleeptime, exitmark):
        while not exitmark.value:
            try:
                # if databucket.qsize() >= buffersize:
                if databucket.full():
                    time.sleep(sleeptime)  # 休眠100ms
                else:
                    data = datafun(*funarg)
                    # datalock.acquire()
                    databucket.put(data)
                    print("input data")
                    # datalock.release()
            except Exception as ex:
                frame = inspect.currentframe()
                print(__file__, "function=", frame.f_code.co_name, " line=", frame.f_lineno, " ex=", ex)
                time.sleep(1)

    def getdata(self):
        delaytimes = 0
        while not self.__exitmark.value:
            # if self.__databucket.qsize() > 0:
            if not self.__databucket.empty():
                # self.__datalock.acquire()
                data = self.__databucket.get(True)
                # self.__datalock.acquire()
                # self.__databucket[1 - self.__bucketWriteriter.value].pop(0)
                return data
            else:
                time.sleep(0.1)
                delaytimes += 1
                if delaytimes % 100 == 0:
                    print("warning processParallelMaker delaytimes is too often delaytimes=", delaytimes)  # 如果频繁出现延迟 则可以提高进程数量或者增大缓存大小

    def stop(self):
        if not self.__exitmark.value:
            self.__exitmark.value = True
            for p in self.__processlist:
                p.join()


class data_Prefetcher(object):
    """
    数据预加载器
    """

    def __init__(self, loader, datamean, datastd):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([datamean[0] * 255, datamean[1] * 255, datamean[2] * 255]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([datastd[0] * 255, datastd[1] * 255, datastd[2] * 255]).cuda().view(1, 3, 1, 1)
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.next_target = None
        self.next_input = None
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        inputdata = self.next_input
        target = self.next_target
        self.preload()
        return inputdata, target

    @staticmethod
    def fast_collate(batch):
        imgs = [img[0] for img in batch]
        targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
        w = imgs[0].size[0]
        h = imgs[0].size[1]
        tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
        for i, img in enumerate(imgs):
            nump_array = np.asarray(img, dtype=np.uint8)
            # tens = torch.from_numpy(nump_array)
            if nump_array.ndim < 3:
                nump_array = np.expand_dims(nump_array, axis=-1)
            nump_array = np.rollaxis(nump_array, 2)
            tensor[i] += torch.from_numpy(nump_array.copy())

        return tensor, targets


class dataSetLoader(object):
    """
    数据集
    """

    def __init__(self, traindataset, testdataset, preLoader, datamean, datastd):
        self.traindataset = traindataset
        self.testdataset = testdataset
        self.traindatasize = len(self.traindataset)
        if self.testdataset is not None:
            self.testdatasize = len(self.testdataset)
        else:
            self.testdatasize = -1
        self.datamean = datamean
        self.datastd = datastd
        self.preloader = preLoader

    def maketraindataiter(self):
        if self.preloader:
            traindataiter = data_Prefetcher(self.traindataset, self.datamean, self.datastd)
        else:
            traindataiter = iter(self.traindataset)
        return traindataiter

    def maketestdataiter(self):
        if self.preloader:
            testdataiter = data_Prefetcher(self.testdataset, self.datamean, self.datastd)
        else:
            testdataiter = iter(self.testdataset)
        return testdataiter


class dataSetMaker(object):
    """
    数据集生成器
    """

    def __init__(self, traindataset, testdataset, preLoader, datamean, datastd):
        self.traindatasize = len(traindataset)
        self.testdatasize = len(testdataset)
        self.dataset = dataSetLoader(traindataset, testdataset, preLoader, datamean, datastd)
        self.traindataitermaker = processParallelMaker(processfun=self.dataset.maketraindataiter, funarg=("fdsa", "123"), pnum=1, buffsize=4)
        # self.testdataitermaker = dataParallelMaker(datafun=dataSet.maketestdataiter, funarg=(self, "123"), pnum=2, buffsize=4)

    def gettraindataiter(self):
        return self.traindataitermaker.getdata()

    def gettestdataiter(self):
        return self.testdataitermaker.getdata()


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    """
    use the multi-epochs-loader to save time at the beginning of every epoch
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class PrefetchLoader:
    """

    """

    def __init__(self,
                 loader,
                 mean,
                 std,
                 fp16=False,
                 re_prob=0.,
                 re_mode='const',
                 re_count=1,
                 re_num_splits=0):
        self.loader = loader
        self.mean = torch.tensor([x * 255 for x in mean]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([x * 255 for x in std]).cuda().view(1, 3, 1, 1)
        self.fp16 = fp16
        if fp16:
            self.mean = self.mean.half()
            self.std = self.std.half()
        if re_prob > 0.:
            self.random_erasing = None  # RandomErasing(probability=re_prob, mode=re_mode, max_count=re_count, num_splits=re_num_splits)
        else:
            self.random_erasing = None

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in self.loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                # if self.fp16:
                #     next_input = next_input.half().sub_(self.mean).div_(self.std)
                # else:
                #     next_input = next_input.float().sub_(self.mean).div_(self.std)
                # if self.random_erasing is not None:
                #     next_input = self.random_erasing(next_input)

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset

    # @property
    # def mixup_enabled(self):
    #     if isinstance(self.loader.collate_fn, FastCollateMixup):
    #         return self.loader.collate_fn.mixup_enabled
    #     else:
    #         return False
    #
    # @mixup_enabled.setter
    # def mixup_enabled(self, x):
    #     if isinstance(self.loader.collate_fn, FastCollateMixup):
    #         self.loader.collate_fn.mixup_enabled = x


class matchDatatool(object):
    """
    数据工具类
    """

    # static parame
    __version__ = 1

    def __init__(self):
        pass

    @staticmethod
    def makeimgaug(parammap):
        imgaugseqlist = []
        for key, val in parammap.items():
            imgaugtransclass = getattr(iaa, key)
            imgAugfun = imgaugtransclass(**val)
            imgaugseqlist.append(imgAugfun)
        imgaugseg = iaa.Sequential(imgaugseqlist)
        return imgaugseg

    @staticmethod
    def maketorchtransform(parammap):
        transformlist = []
        for key, val in parammap.items():
            if key == 'RandomChoice':
                choiceclass = getattr(transforms, key)
                choicefunlist = []
                for k, v in val.items():
                    torchtransclass = getattr(transforms, k)
                    torchtransfun = torchtransclass(**v)
                    choicefunlist.append(torchtransfun)
                torchtransfun = choiceclass(choicefunlist)
                transformlist.append(torchtransfun)
            else:
                torchtransclass = getattr(transforms, key)
                torchtransfun = torchtransclass(**val)
                transformlist.append(torchtransfun)
        torchtransforms = transforms.Compose(transformlist)
        return torchtransforms

    # @staticmethod
    # def mixup_data(x, y, alpha=1.0, use_cuda=True):
    #     # 对数据的mixup 操作 x = lambda*x_i+(1-lamdda)*x_j
    #     """Returns mixed inputs, pairs of targets, and lambda"""
    #     if alpha > 0:
    #         lam = np.random.beta(alpha, alpha)
    #     else:
    #         lam = 1
    #
    #     batch_size = x.size()[0]
    #     if use_cuda:
    #         index = torch.randperm(batch_size).cuda()
    #     else:
    #         index = torch.randperm(batch_size)
    #
    #     mixed_x = lam * x + (1 - lam) * x[index, :]  # 此处是对数据x_i 进行操作
    #     y_a, y_b = y, y[index]  # 记录下y_i 和y_j
    #     return mixed_x, y_a, y_b, lam  # 返回y_i 和y_j 以及lambda
    #
    # @staticmethod
    # def mixup_criterion(criterion, pred, y_a, y_b, lam):
    #     # 对loss函数进行混合，criterion是crossEntropy函数
    #     return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


class torchUtils(object):
    """
    pytorch工具类
    """

    def __init__(self):
        pass

    @staticmethod
    def init_seeds(flag, seed) -> None:
        # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
        if flag:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True  # 保证每次初始值都一样
            torch.backends.cudnn.benchmark = False
            logging.info("rand_seed=" + str(seed))
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = False

    @staticmethod
    def git_describe():
        # return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe
        if Path('.git').exists():
            return subprocess.check_output('git describe --tags --long --always', shell=True).decode('utf-8')[:-1]
        else:
            return ''

    @staticmethod
    def select_device(device='', batch_size=None):
        # device = 'cpu' or '0' or '0,1,2,3'
        s = f'🚀 torch {torch.__version__} '  # string
        cpu = device.lower() == 'cpu'
        if cpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
        elif device:  # non-cpu device requested
            os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
            assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

        cuda = not cpu and torch.cuda.is_available()
        if cuda:
            n = torch.cuda.device_count()
            if n > 1 and batch_size:  # check that batch_size is compatible with device_count
                assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
            space = ' ' * len(s)
            for i, d in enumerate(device.split(',') if device else range(n)):
                p = torch.cuda.get_device_properties(i)
                s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
        else:
            s += 'CPU\n'

        logger.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # emoji-safe
        return torch.device('cuda:0' if cuda else 'cpu')

    @staticmethod
    def time_synchronized():
        # pytorch-accurate time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()

    @staticmethod
    def is_parallel(model):
        return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

    @staticmethod
    def intersect_dicts(da, db, exclude=()):
        # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
        state_dict = {}
        for k, v in da.items():
            if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape:
                state_dict[k] = v
        return state_dict
        # return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}

    @staticmethod
    def initialize_weights(model):
        for m in model.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
                m.inplace = True

    @staticmethod
    def find_modules(model, mclass=nn.Conv2d):
        # Finds layer indices matching module class 'mclass'
        return [i for i, m in enumerate(model.module_list) if isinstance(m, mclass)]

    @staticmethod
    def sparsity(model):
        # Return global model sparsity
        a, b = 0., 0.
        for p in model.parameters():
            a += p.numel()
            b += (p == 0).sum()
        return b / a

    @staticmethod
    def prune(model, amount=0.3):
        # Prune model to requested global sparsity
        import torch.nn.utils.prune as prune
        print('Pruning model... ', end='')
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                prune.l1_unstructured(m, name='weight', amount=amount)  # prune
                prune.remove(m, 'weight')  # make permanent
        print(' %.3g global sparsity' % torchUtils.sparsity(model))

    @staticmethod
    def fuse_conv_and_bn(conv, bn):
        # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
        fusedconv = nn.Conv2d(conv.in_channels,
                              conv.out_channels,
                              kernel_size=conv.kernel_size,
                              stride=conv.stride,
                              padding=conv.padding,
                              groups=conv.groups,
                              bias=True).requires_grad_(False).to(conv.weight.device)

        # prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

        # prepare spatial bias
        b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        return fusedconv

    @staticmethod
    def model_info(model, verbose=False, img_size=640):
        # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
        n_p = sum(x.numel() for x in model.parameters())  # number parameters
        n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
        if verbose:
            print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
            for i, (name, p) in enumerate(model.named_parameters()):
                name = name.replace('module_list.', '')
                print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                      (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

        try:  # FLOPS
            from thop import profile
            stride = int(model.stride.max()) if hasattr(model, 'stride') else 32
            img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device)  # input
            flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2  # stride GFLOPS
            img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # expand if int/float
            fs = ', %.1f GFLOPS' % (flops * img_size[0] / stride * img_size[1] / stride)  # 640x640 GFLOPS
        except (ImportError, Exception):
            fs = ''

        logger.info(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")

    @staticmethod
    def load_classifier(name='resnet101', n=2):
        # Loads a pretrained model reshaped to n-class output
        model = torchvision.models.__dict__[name](pretrained=True)

        # ResNet model properties
        # input_size = [3, 224, 224]
        # input_space = 'RGB'
        # input_range = [0, 1]
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]

        # Reshape output to n classes
        filters = model.fc.weight.shape[1]
        model.fc.bias = nn.Parameter(torch.zeros(n), requires_grad=True)
        model.fc.weight = nn.Parameter(torch.zeros(n, filters), requires_grad=True)
        model.fc.out_features = n
        return model

    @staticmethod
    def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
        # scales img(bs,3,y,x) by ratio constrained to gs-multiple
        if ratio == 1.0:
            return img
        else:
            h, w = img.shape[2:]
            s = (int(h * ratio), int(w * ratio))  # new size
            img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
            if not same_shape:  # pad/crop img
                h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
            return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean

    @staticmethod
    def copy_attr(a, b, include=(), exclude=()):
        # Copy attributes from b to a, options to only include [...] and to exclude [...]
        for k, v in b.__dict__.items():
            if (len(include) and k not in include) or k.startswith('_') or k in exclude:
                continue
            else:
                setattr(a, k, v)

    @staticmethod
    def profile(x, ops, n=100, device=None):
        # profile a pytorch module or list of modules. Example usage:
        #     x = torch.randn(16, 3, 640, 640)  # input
        #     m1 = lambda x: x * torch.sigmoid(x)
        #     m2 = nn.SiLU()
        #     profile(x, [m1, m2], n=100)  # profile speed over 100 iterations

        device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        x = x.to(device)
        x.requires_grad = True
        print(torch.__version__, device.type, torch.cuda.get_device_properties(0) if device.type == 'cuda' else '')
        print(f"\n{'Params':>12s}{'GFLOPS':>12s}{'forward (ms)':>16s}{'backward (ms)':>16s}{'input':>24s}{'output':>24s}")
        for m in ops if isinstance(ops, list) else [ops]:
            m = m.to(device) if hasattr(m, 'to') else m  # device
            m = m.half() if hasattr(m, 'half') and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m  # type
            dtf, dtb, t = 0., 0., [0., 0., 0.]  # dt forward, backward
            try:
                flops = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # GFLOPS
            except Exception as ex:
                print(ex)
                flops = 0

            y = None
            for _ in range(n):
                t[0] = torchUtils.time_synchronized()
                y = m(x)
                t[1] = torchUtils.time_synchronized()
                try:
                    _ = y.sum().backward()
                    t[2] = torchUtils.time_synchronized()
                except Exception as ex:  # no backward method
                    print(ex)
                    t[2] = float('nan')
                dtf += (t[1] - t[0]) * 1000 / n  # ms per op forward
                dtb += (t[2] - t[1]) * 1000 / n  # ms per op backward

            s_in = tuple(x.shape) if isinstance(x, torch.Tensor) else 'list'
            s_out = tuple(y.shape) if isinstance(y, torch.Tensor) else 'list'
            p = sum(list(x.numel() for x in m.parameters())) if isinstance(m, nn.Module) else 0  # parameters
            print(f'{p:12.4g}{flops:12.4g}{dtf:16.4g}{dtb:16.4g}{str(s_in):>24s}{str(s_out):>24s}')


class pruneTool(object):
    """
    裁剪工具类
    """

    def __init__(self):
        pass

    @staticmethod
    def compute_final_pruning_rate(pruning_rate, num_iterations):
        """A function to compute the final pruning rate for iterative pruning.
            Note that this cannot be applied for global pruning rate if the pruning rate is heterogeneous among different layers.

        Args:
            pruning_rate (float): Pruning rate.
            num_iterations (int): Number of iterations.

        Returns:
            float: Final pruning rate.
        """

        final_pruning_rate = 1 - (1 - pruning_rate) ** num_iterations

        return final_pruning_rate

    @staticmethod
    def measure_module_sparsity(module, weight=True, bias=False, use_mask=False):
        num_zeros = 0
        num_elements = 0

        if use_mask:
            for buffer_name, buffer in module.named_buffers():
                if "weight_mask" in buffer_name and weight:
                    num_zeros += torch.sum(buffer == 0).item()
                    num_elements += buffer.nelement()
                if "bias_mask" in buffer_name and bias:
                    num_zeros += torch.sum(buffer == 0).item()
                    num_elements += buffer.nelement()
        else:
            for param_name, param in module.named_parameters():
                if "weight" in param_name and weight:
                    num_zeros += torch.sum(param == 0).item()
                    num_elements += param.nelement()
                if "bias" in param_name and bias:
                    num_zeros += torch.sum(param == 0).item()
                    num_elements += param.nelement()

        sparsity = num_zeros / num_elements
        return num_zeros, num_elements, sparsity

    @staticmethod
    def measure_global_sparsity(model,
                                weight=True,
                                bias=False,
                                conv2d_use_mask=False,
                                linear_use_mask=False):

        num_zeros = 0
        num_elements = 0

        for module_name, module in model.named_modules():

            if isinstance(module, torch.nn.Conv2d):
                module_num_zeros, module_num_elements, _ = pruneTool.measure_module_sparsity(
                    module, weight=weight, bias=bias, use_mask=conv2d_use_mask)
                num_zeros += module_num_zeros
                num_elements += module_num_elements

            elif isinstance(module, torch.nn.Linear):

                module_num_zeros, module_num_elements, _ = pruneTool.measure_module_sparsity(
                    module, weight=weight, bias=bias, use_mask=linear_use_mask)
                num_zeros += module_num_zeros
                num_elements += module_num_elements

        sparsity = num_zeros / num_elements

        return num_zeros, num_elements, sparsity

    @staticmethod
    def pruning(model, conv2d_prune_amount=0.4, linear_prune_amount=0.2, num_iterations=10, grouped_pruning=False) -> None:
        for i in range(num_iterations):
            print("Pruning and Finetuning {}/{}".format(i + 1, num_iterations))
            num_zeros, num_elements, sparsity = pruneTool.measure_global_sparsity(model, weight=True, bias=False, conv2d_use_mask=True, linear_use_mask=False)

            print("Global Sparsity:")
            print("{:.2f}".format(sparsity))

            print("Pruning...")
            if grouped_pruning:
                # Global pruning
                # I would rather call it grouped pruning.
                parameters_to_prune = []
                for module_name, module in model.named_modules():
                    if isinstance(module, torch.nn.Conv2d):
                        parameters_to_prune.append((module, "weight"))
                prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=conv2d_prune_amount, )
            else:
                for module_name, module in model.named_modules():
                    if isinstance(module, torch.nn.Conv2d):
                        prune.l1_unstructured(module, name="weight", amount=conv2d_prune_amount)
                    elif isinstance(module, torch.nn.Linear):
                        prune.l1_unstructured(module, name="weight", amount=linear_prune_amount)

            num_zeros, num_elements, sparsity = pruneTool.measure_global_sparsity(model, weight=True, bias=False, conv2d_use_mask=True, linear_use_mask=False)
            print("Global Sparsity:")
            print("{:.2f}".format(sparsity))
        return model

    @staticmethod
    def remove_parameters(model):
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                try:
                    prune.remove(module, "weight")
                except Exception as ex:
                    print(ex)
                try:
                    prune.remove(module, "bias")
                except Exception as ex:
                    print(ex)
            elif isinstance(module, torch.nn.Linear):
                try:
                    prune.remove(module, "weight")
                except Exception as ex:
                    print(ex)
                try:
                    prune.remove(module, "bias")
                except Exception as ex:
                    print(ex)
        return model


class WarmUpLR(torch.optim.lr_scheduler._LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model.module if torchUtils.is_parallel(model) else model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if torchUtils.is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        torchUtils.copy_attr(self.ema, model, include, exclude)


class ModelEMAV2:
    """
    Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        """
        Args:
            model (nn.Module): model to apply EMA.
            decay (float): ema decay reate.
            updates (int): counter of EMA updates.
        """
        # Create EMA(FP32)
        self.ema = deepcopy(model.module if ModelEMAV2.is_parallel(model) else model).eval()
        self.updates = updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = (
                model.module.state_dict() if ModelEMAV2.is_parallel(model) else model.state_dict()
            )  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()

    @staticmethod
    def is_parallel(model):
        """check if model is in parallel mode."""
        parallel_type = (
            nn.parallel.DataParallel,
            nn.parallel.DistributedDataParallel,
        )
        return isinstance(model, parallel_type)


class EMA(object):
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def get(self, name):
        return self.shadow[name]

    def update(self, name, x):
        assert name in self.shadow
        new_average = (1.0 - self.decay) * x + self.decay * self.shadow[name]
        self.shadow[name] = new_average.clone()


def emademo():
    # init
    model = torchvision.models.resnet18(pretrained=True)
    ema = EMA(0.999)

    # register
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)

    # update
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.update(name, param.data)


if __name__ == '__main__':
    pass
