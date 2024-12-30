# encoding: utf-8

"""
模型训练器

tensorboard --logdir=logs/summary_

#多卡训练脚本
python -m torch.distributed.launch --nproc_per_node 2 train.py --device 0,1
查看显卡使用状态
watch -n1 nvidia-smi

conda create -n detectron2 python=3.7
"""
import sys
sys.path.append('./')
import numpy as np

from matchx.matchutils import *
import torch
# import tensorwatch as tw
import torch.nn as nn
import torch.distributed as dist
import torchvision
import cv2
import yaml
from tqdm import tqdm
from shutil import copyfile
import argparse
from matchx import torchtools
from addict import Dict
from pathlib import Path
import importlib
# from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, \
#     convert_splitbn_model, model_parameters

# from skimage import io


"""
##################################################################################################################
在此之后添加需要的功能
"""


class ModelTrainer(torchtools.ModelTrainerbase):
    """
    modelconfig: 模型配置信息
    opt: 进程启动参数
    dateset: 数据集合
    device: 模型运行设备
    workdir: 工作路径 用于保存日志和模型
    """

    def __init__(self, modelconfig, opt, dataset, device, workdir):
        super(ModelTrainer, self).__init__(modelconfig, opt, dataset, device, workdir)
        # super().__init__(modelconfig, opt, dataset, device, workdir)
        # ModelTrainerbase.__init__(self, modelconfig, opt, dataset, device, workdir)
        self.mixup_enable = False
        self.mixup_fn = None
        if 'transformpipeline' in self.dataconfig:
            if 'mixup' in self.dataconfig.transformpipeline:
                mixupcfg = self.dataconfig.transformpipeline.mixup
                self.mixup_fn = torchtools.Mixup(**mixupcfg)
                self.mixup_enable = True

    def makenet(self):
        from timm.models import create_model
        net = create_model(self.modelcfgdict.modcfg.netname, pretrained=True, num_classes=self.modelcfgdict.datacfg.num_class)
        assert net is not None, f'net is None'
        return net

    def makeoptimizer(self):
        optimizer = None
        # 初始化优化器
        if self.modelcfgdict.schedulecfg.optim_type == "sgd":
            optimizer = torch.optim.SGD(self.trainnet.parameters(),
                                        lr=self.modelcfgdict.schedulecfg.base_lr,
                                        momentum=self.modelcfgdict.schedulecfg.momentum,
                                        weight_decay=self.modelcfgdict.schedulecfg.weight_decay)
        elif self.modelcfgdict.schedulecfg.optim_type == "adamw":
            optimizer = torch.optim.AdamW(self.trainnet.parameters(),
                                          lr=self.modelcfgdict.schedulecfg.base_lr,
                                          weight_decay=self.modelcfgdict.schedulecfg.weight_decay)
        elif self.modelcfgdict.schedulecfg.optim_type == "sam":
            base_optimizer = torch.optim.SGD
            optimizer = torchtools.SAM(self.trainnet.parameters(),
                                       base_optimizer,
                                       rho=self.modelcfgdict.schedulecfg.rho,
                                       adaptive=self.modelcfgdict.schedulecfg.adaptive,
                                       lr=self.modelcfgdict.schedulecfg.base_lr,
                                       momentum=self.modelcfgdict.schedulecfg.momentum,
                                       weight_decay=self.modelcfgdict.schedulecfg.weight_decay)

        assert optimizer is not None, f'optimizer is None'
        return optimizer

    def makescheduler(self):
        RSchedulerClass = getattr(torchtools, self.modelcfgdict.schedulecfg.lr_schedule)
        sch = RSchedulerClass(self.optimizer,
                              t_initial=self.modelcfgdict.hypcfg.totalepoch,
                              warmup_t=self.modelcfgdict.schedulecfg.warm_up_epoch,
                              warmup_lr_init=self.modelcfgdict.schedulecfg.get("warmup_lr_init", self.modelcfgdict.schedulecfg.base_lr / 10),
                              lr_min=self.modelcfgdict.schedulecfg.get("lr_min", self.modelcfgdict.schedulecfg.base_lr / 1000),
                              )
        assert sch is not None, f'scheduler is None'
        return sch

    def makelossfunction(self):
        fname = self.modelcfgdict.schedulecfg.lossfun + "()"
        loss_function = eval(fname)
        assert loss_function is not None, f'loss_function is None'
        return loss_function

    def _train_epoch(self, epoch, lr=None):
        train_loss = 0
        correct = 0
        total = 0
        loss = 0
        trainacc = 0
        logmsg = None
        epochratio = epoch / self.EPOCHs
        if epochratio >= self.dataconfig.imgaugseg_epoch_ratio:
            self.dataset.traindataset.dataset.IMGAUGSEG_Enable.value = False

        datasize = self.dataset.traindatasize
        # batch_idx = 0
        # traindataiter = self.dataset.maketraindataiter()
        if self.argument.global_rank in [-1, 0]:
            pbar = tqdm(total=datasize, ncols=160)  # , ncols=80
        else:
            pbar = None
        for batch_idx, (inputs, targets) in enumerate(self.dataset.traindataset):
            # torch.cuda.empty_cache()
            if batch_idx >= datasize:
                break
            tarlabel = np.copy(targets)
            badcount = 0
            losscount = 0
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            if self.modelcfgdict.schedulecfg.optim_type == "sam":
                outputs = self.trainnet(inputs)
                loss = self.loss_function(outputs, targets)
                loss.backward()
                self.optimizer.first_step(zero_grad=True)

                # second forward-backward step
                self.loss_function(self.trainnet(inputs), targets).backward()
                self.optimizer.second_step(zero_grad=True)
            else:
                self.optimizer.zero_grad()
                # outputs = self.trainnet.forward(inputs)
                # loss = self.loss_function(outputs, targets)
                # loss.backward()
                # self.optimizer.step()
                if self.mixup_fn is not None:
                    # inputs, targets_a, targets_b, lam = torchtools.matchDatatool.mixup_data(inputs, targets, self.mixup_alpha, self.mixup_usecuda)
                    inputs, targets = self.mixup_fn(inputs, targets)
                    with torch.cuda.amp.autocast(enabled=self.modelcfgdict.hypcfg.useamp):
                        outputs = self.trainnet(inputs)
                        loss = self.loss_function(outputs, targets)
                        # loss = torchtools.matchDatatool.mixup_criterion(self.loss_function, outputs, targets_a, targets_b, lam)  # 对loss#函数进行mixup操作
                else:
                    with torch.cuda.amp.autocast(enabled=self.modelcfgdict.hypcfg.useamp):
                        outputs = self.trainnet(inputs)
                        loss = self.loss_function(outputs, targets)

                #  计算漏报与误报数量 将二者作为惩罚项加入损失值
                outnp = torch.sigmoid(outputs).cpu().detach().numpy()
                outmaxconf = outnp.max(1)
                outlabel = np.nanargmax(outnp, axis=1)

                for inx in range(len(tarlabel)):
                    tarlb = tarlabel[inx]
                    if tarlb == 0:
                        #  漏报个数
                        if outlabel[inx] != 0:
                            pass
                            losscount += 1
                    else:
                        # 误报个数
                        if outlabel[inx] == 0 and outmaxconf[inx] > 0.6:
                            badcount += 1
                self.ampscaler.scale(loss * (1 + self.modelcfgdict.hypcfg.baddatafactor * badcount + self.modelcfgdict.hypcfg.lossdatafactor * losscount)).backward()
                self.ampscaler.step(self.optimizer)
                self.ampscaler.update()
                if self.model_ema:
                    self.model_ema.update(self.trainnet)

            if self.argument.global_rank in [-1, 0]:
                train_loss += loss.item()

                # _, preds = torch.max(outputs, 1)
                # preds = preds == targets.data
                # acc = torch.mean(preds.float())

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                trainacc = 100. * correct / total

                if batch_idx == 1:
                    x = torchvision.utils.make_grid(inputs, normalize=True, scale_each=True)
                    self.sumwriter.add_image('Image', x, epoch)
                gpumem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                logmsg = 'gpumem:%s epoch:%d/%d Lr:%.7f| Loss:%.6f trainAcc:%.3f%%(%d/%d)' % \
                         (gpumem, epoch, self.EPOCHs, self.optimizer.param_groups[0]['lr'],
                          train_loss / (batch_idx + 1), trainacc, correct, total)
                # Utiltool.progress_bar(batch_idx, datasize, logmsg)
                if pbar is not None:
                    pbar.set_description("Train", refresh=False)
                    pbar.set_postfix(Info=logmsg, refresh=False)
                    pbar.update()
            # batch_idx += 1
            del inputs
            del targets
            del loss

        if pbar is not None:
            pbar.close()
        # 记录网络权重状态
        if self.showlayerweight and self.argument.global_rank in [-1, 0]:
            # cnn_weights = self.trainnet.state_dict()['features.0.weight'].cpu()
            layerlist = self.trainnet.state_dict().values()
            for cnn_weights in layerlist:
                cnn_weight_img = torchvision.utils.make_grid(cnn_weights, normalize=True, scale_each=True)
                self.sumwriter.add_image('weight_Image', cnn_weight_img, epoch)
                break
        # 记录网络输出状态
        if self.argument.global_rank in [-1, 0]:
            # cnn_out = torchvision.utils.make_grid(self.conv_out0.outfeatures, normalize=True, scale_each=True)
            # self.sumwriter.add_image('out_Image', cnn_out, epoch)
            if self.showfeaturemap and epoch > self.modelcfgdict.hypcfg.totalepoch * 0.8:
                fig = plt.figure(figsize=(20, 20))
                fig.subplots_adjust(left=0, right=1, bottom=0, top=0.8, hspace=0, wspace=0.2)
                for i in range(64):
                    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
                    ax.imshow(self.conv_out0.outfeatures[0][i])
                plt.show()
            # 记录训练状态
            self.sumwriter.add_scalars('train_data/scalar_group', {'loss': train_loss, 'acc': trainacc}, epoch)
            logging.info(logmsg)
        # print("train_lr=", self.train_scheduler.get_last_lr())

    # @torch.no_grad()
    def evaluate(self, epoch) -> float:
        if self.dataset.testdataset is None:
            return -1
        self.trainnet.eval()
        test_loss = 0
        correct = 0
        total = 0
        logmsg = None
        badcount = 0  # 未戴帽误报个数
        losscount = 0  # 未戴帽漏报个数

        datasize = len(self.dataset.testdataset)
        pbar = tqdm(total=datasize, ncols=160)

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.dataset.testdataset):
                tarlabel = np.copy(targets)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.trainnet(inputs)

                test_loss += 0.  # loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                outnp = torch.sigmoid(outputs).to("cpu").numpy()
                outmaxconf = outnp.max(1)
                outlabel = np.nanargmax(outnp, axis=1)

                for inx in range(len(tarlabel)):
                    tarlb = tarlabel[inx]
                    if tarlb == 0:
                        #  漏报个数
                        if outlabel[inx] != 0:
                            losscount += 1
                    else:
                        # 误报个数
                        if outlabel[inx] == 0 and outmaxconf[inx] > 0.6:
                            badcount += 1

                logmsg = 'Loss:%.4f | testAcc:%.4f%% baddata:%.4f%% lossdata:%.4f%% (%d/%d)' % \
                         (test_loss / (batch_idx + 1), 100. * correct / total, 100. * badcount / total, 100. * losscount / total, correct, total)
                # Utiltool.progress_bar(batch_idx, len(self.dataset.testdataset), logmsg)
                pbar.set_description("Test", refresh=False)
                pbar.set_postfix(Info=logmsg, refresh=False)
                pbar.update()
                # if batch_idx == 1 and epoch == 40:
                #     features = inputs.view(64, 3 * 32 * 32)
                #     self.sumwriter.add_embedding(torch.randn(64, 5), metadata=targets, label_img=inputs)

        pbar.close()
        acc = 100. * correct / total
        print("test_Acc=", round(acc, 5))
        self.sumwriter.add_scalars('test_data/scalar_group', {'loss': test_loss, 'acc': acc}, epoch)
        logging.info("Epoch=" + str(epoch) + " " + logmsg)
        return acc

    @staticmethod
    def inittrainer(configpath):
        import matchx
        from task.kitchen_cigar.datasets.datamaker_cigar import dataloaderMaker

        # init env
        modelconfig, opt, device, workdir = matchx.Utiltool.init_pytorch_env(configpath)
        # make data
        datasetloader, dataset  = dataloaderMaker.makeroad_dataset(modelconfig, opt.global_rank)
        # 创建训练器
        modeltrainer = ModelTrainer(modelconfig, opt, datasetloader, device, workdir)
        return modeltrainer

    """
    ##################################################################################################################
    在此之后添加临时测试功能
    """

    def clpic(self):
        from shutil import copyfile
        rootdir = "D:/dataset/face_qu/test_back/newdata/p2org"
        image_namelist = []
        for s in os.listdir(rootdir):
            filep = os.path.join(rootdir, s)
            if os.path.isfile(filep):
                image_namelist.append(filep)
        with torch.no_grad():
            for imgpath in image_namelist:
                cvimg = cv2.imread(imgpath)
                imgH = cvimg.shape[0]
                imgW = cvimg.shape[1]
                stepH = int(imgH / 3)
                stepW = int(imgW / 3)
                if stepH < 36 or stepW < 36:
                    continue
                cvimg = cvimg[stepH:stepH * 2, stepW:stepW * 2]
                # cv2.namedWindow('dd', cv2.WINDOW_AUTOSIZE)
                # cv2.imshow("dd", img)
                # cv2.waitKey(1)
                img = cv2.resize(cvimg, (224, 224)).astype(float)
                img /= 255.0
                # img[:, :, 0] = (img[:, :, 0] - 0.4914) / 0.2023
                # img[:, :, 1] = (img[:, :, 1] - 0.4822) / 0.1994
                # img[:, :, 2] = (img[:, :, 2] - 0.4465) / 0.2010
                img0 = torch.from_numpy(img)
                img1 = img0.permute(2, 0, 1).float()
                imgtensor = img1.unsqueeze(0).to(self.device)
                outputs = self.trainnet(imgtensor)
                inx = outputs.to("cpu").numpy().argmax()
                (filepath, filename) = os.path.split(imgpath)
                distpath = "D:/dataset/face_qu/test_back/newdata/p2/" + str(inx) + "/" + filename
                copyfile(imgpath, distpath)
                cv2.imwrite(distpath, cvimg)

    def classdemo(self):
        import json

        checkpoint = torch.load("./epoch_67_93.74.ptcp")
        self.trainnet.load_state_dict(checkpoint['model_state_dict'])  # 加载模型可学习参数

        self.trainnet.eval()
        with torch.no_grad():
            data_mean = [0.485, 0.456, 0.406]
            std_mean = [0.229, 0.224, 0.225]
            self.trainnet.eval()
            with torch.no_grad():
                traindata = json.load(open("results.jon", 'r'))
                inx = 1
                filecount = len(traindata)
                lastfilename = ""
                imgbgr = None
                outdata = []
                for it in traindata:
                    print(str(inx) + "/" + str(filecount))
                    inx += 1
                    score = it['score']
                    if score < 0.0005:
                        continue
                    imgid = it['image_id']
                    bbox = it['bbox']
                    bboxint = [int(i) for i in bbox]
                    fileid = '{:0>5d}'.format(imgid) + ".jpg"
                    fullfile = "D:/dataset/road/images/test/" + fileid
                    if fullfile != lastfilename:
                        imgbgr = cv2.imread(fullfile)
                    lastfilename = fullfile
                    # bxmin = int(bbox[0])
                    # bxmax = int(bbox[0] + bbox[2])
                    # bymin = int(bbox[1])
                    # bymax = int(bbox[1] + bbox[3])
                    subimg = imgbgr[bboxint[1]:(bboxint[1] + bboxint[3]), bboxint[0]:(bboxint[0] + bboxint[2])]
                    subimg = cv2.resize(subimg, (224, 224), interpolation=cv2.INTER_CUBIC)
                    # if True:
                    #     cv2.imshow("aa", subimg)
                    #     cv2.waitKey(1)
                    subimg = subimg[:, :, ::-1].transpose((2, 0, 1))  # BGR to RGB and HWC to CHW
                    subimg = np.ascontiguousarray(subimg) / 255.0
                    subimg[0] = (subimg[0][:] - data_mean[0]) / std_mean[0]
                    subimg[1] = (subimg[1][:] - data_mean[1]) / std_mean[0]
                    subimg[2] = (subimg[2][:] - data_mean[2]) / std_mean[0]
                    subimg = subimg.astype(np.float32)
                    subimgtensor = torch.from_numpy(subimg)
                    imgtensor = subimgtensor.to(self.device)
                    inputs = imgtensor.unsqueeze(0)
                    outputs = self.trainnet(inputs)
                    labels = outputs.to('cpu').numpy()
                    cls = labels.argmax() + 1
                    it['category_id'] = int(cls)
                    outdata.append(it)

                with open("yoloout.json", "w", encoding="utf-8") as f:
                    json.dump(outdata, f, indent=4)
            print("finish")

    def convert2onnx(self, pthpath, outpath):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(pthpath)
        self.trainnet.load_state_dict(checkpoint['model_state_dict'])

        all_net = torch.nn.Sequential(
            self.trainnet,
            torch.nn.Softmax(dim=-1)
        )
        all_net.to(device)

        all_net.eval()
        inputs = torch.randn(1, 3, 224, 224)
        inputs = inputs.to(device)

        outputs = all_net(inputs)

        for item in outputs:
            print(item.shape)
        # 输出配置
        output_onnx = outpath
        input_names = ["inputs"]
        output_names = ["outputs"]

        torch.onnx.export(all_net,
                          inputs,
                          output_onnx,
                          export_params=True,
                          opset_version=14,
                          do_constant_folding=True,
                          input_names=input_names,
                          output_names=output_names,
                          dynamic_axes={
                              "inputs": {0: "batch"},
                              "outputs": {0: "batch"}
                          }
                          # dynamic_axes={"input0":{0:"batch_size"},
                          #               "bbox":{0:"batch_size"},
                          #               "cls":{0:"batch_size"},
                          #               }
                          )


if __name__ == '__main__':
    params = {
        "logdir": "logs/",
        "checkpointdir": "checkpoint/",
        "learning_rate": 0.01,
        "momentum": 0.9,
        "warm_up_epoch": 10,
        "total_epoch": 60,
        "dataset": "tfrecords/test",
        "train_batch_size": 32,
        "test_batch_size": 32
    }

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
    print("module test finished")
