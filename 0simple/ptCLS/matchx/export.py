# encoding: utf-8

"""
version=0.1
模型导出器

libtorch

CPU:
https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.2.0.zip
https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-without-deps-1.2.0.zip
https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-static-with-deps-1.2.0.zip
https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-static-without-deps-1.2.0.zip

CUDA 9.2:
https://download.pytorch.org/libtorch/cu92/libtorch-cxx11-abi-shared-with-deps-1.2.0.zip
https://download.pytorch.org/libtorch/cu92/libtorch-cxx11-abi-shared-without-deps-1.2.0.zip
https://download.pytorch.org/libtorch/cu92/libtorch-cxx11-abi-static-with-deps-1.2.0.zip
https://download.pytorch.org/libtorch/cu92/libtorch-cxx11-abi-static-without-deps-1.2.0.zip

CUDA 10.0:
https://download.pytorch.org/libtorch/cu100/libtorch-cxx11-abi-shared-with-deps-1.2.0.zip
https://download.pytorch.org/libtorch/cu100/libtorch-cxx11-abi-shared-without-deps-1.2.0.zip
https://download.pytorch.org/libtorch/cu100/libtorch-cxx11-abi-static-with-deps-1.2.0.zip
https://download.pytorch.org/libtorch/cu100/libtorch-cxx11-abi-static-without-deps-1.2.0.zip

"""


from matchx.matchutils import *
import torch
import tensorwatch as tw
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.nn.init as init
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import onnx


class modelexport(object):
    """
    pytorch工具
    """

    @staticmethod
    def export2onnx(model):
        # 导出模型为onnx格式
        pthfile = r'checkpoint/epoch_2_68.4700.ptcp'
        loaded_model = torch.load(pthfile, map_location='cpu')
        # try:
        #   loaded_model.eval()
        # except AttributeError as error:
        #   print(error)

        model.load_state_dict(loaded_model['model_state_dict'])
        # model = model.to(device)

        # data type nchw
        dummy_input1 = torch.randn(1, 3, 32, 32).cpu()
        # dummy_input2 = torch.randn(1, 3, 64, 64)
        # dummy_input3 = torch.randn(1, 3, 64, 64)
        input_names = ["actual_input_1"]
        output_names = ["output1"]
        # torch.onnx.export(model, (dummy_input1, dummy_input2, dummy_input3), "C3AE.onnx", verbose=True, input_names=input_names, output_names=output_names)
        export_model_name = "mnetv2.onnx"
        torch.onnx.export(model, args=dummy_input1, f=export_model_name, opset_version=11, export_params=True, verbose=True, input_names=input_names, output_names=output_names)
        onnx_model = onnx.load(export_model_name)
        onnx.checker.check_model(onnx_model)
        graph_output = onnx.helper.printable_graph(onnx_model.graph)
        with open("graph_output.txt", mode="w") as fout:
            fout.write(graph_output)
        print("export OK")

    @staticmethod
    def export2onnx_1(modelnet):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # num_class = 11

        weight_path = "/home/han/D/projects/classification/pytorch-image-models/my_code/color_logs/cp-0049.pth"

        modelnet.load_state_dict(torch.load(weight_path))

        all_net = torch.nn.Sequential(
            modelnet,
            torch.nn.Softmax(dim=-1)
        )
        all_net.to(device)

        all_net.eval()
        inputs = torch.randn(1, 3, 192, 64)
        inputs = inputs.to(device)

        outputs = all_net(inputs)

        for item in outputs:
            print(item.shape)
        #
        output_onnx = 'upboday_color.onnx'
        input_names = ["inputs"]
        output_names = ["outputs"]

        torch.onnx.export(all_net,
                          inputs,
                          output_onnx,
                          export_params=True,
                          # opset_version=11,
                          do_constant_folding=True,
                          input_names=input_names,
                          output_names=output_names,
                          # dynamic_axes={"input0":{0:"batch_size"},
                          #               "bbox":{0:"batch_size"},
                          #               "cls":{0:"batch_size"},
                          #               }
                          )


if __name__ == '__main__':
    print("module test finished")
