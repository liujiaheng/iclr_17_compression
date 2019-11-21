#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from .GDN import GDN
from torch.autograd import Variable
import imageio


out_channel_N = 192
out_channel_M = 320
# out_channel_N = 128
# out_channel_M = 192

def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def tensorimwrite(image, name='im'):
    # means = np.array([0.485, 0.456, 0.406])
    # stds = np.array([0.229, 0.224, 0.225])
    if len(image.size()) == 4:
        image = image[0]
    image = image.detach().cpu().numpy().transpose(1, 2, 0)
    image = image * 255
    imageio.imwrite(name + ".png", image.astype(np.uint8))

def relu(x):
    return x


def yuv_import_444(filename, dims, numfrm, startfrm):
    fp = open(filename, 'rb')
    # fp=open(filename,'rb')

    blk_size = int(dims[0] * dims[1] * 3)
    fp.seek(blk_size * startfrm, 0)
    Y = []
    U = []
    V = []
    # print(dims[0])
    # print(dims[1])
    d00 = dims[0]
    d01 = dims[1]
    # print(d00)
    # print(d01)
    Yt = np.zeros((dims[0], dims[1]), np.int, 'C')
    Ut = np.zeros((d00, d01), np.int, 'C')
    Vt = np.zeros((d00, d01), np.int, 'C')
    print(dims[0])
    YUV = np.zeros((dims[0], dims[1], 3))

    for m in range(dims[0]):
        for n in range(dims[1]):
            Yt[m, n] = ord(fp.read(1))
    for m in range(d00):
        for n in range(d01):
            Ut[m, n] = ord(fp.read(1))
    for m in range(d00):
        for n in range(d01):
            Vt[m, n] = ord(fp.read(1))

    YUV[:, :, 0] = Yt
    YUV[:, :, 1] = Ut
    YUV[:, :, 2] = Vt
    fp.close()
    return YUV


def CalcuPSNR(target, ref):
    diff = ref - target
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff**2.))
    return 20 * math.log10(1.0 / (rmse))

def MSE2PSNR(MSE):
    return 10 * math.log10(1.0 / (MSE))

def geti(lamb):
    if lamb == 2048:
        return 'L12000'
    elif lamb == 1024:
        return 'L4096'
    elif lamb == 512:
        return 'L2048'
    elif lamb == 256:
        return 'L1024'
    elif lamb == 8:
        return 'L16'
    elif lamb == 16:
        return 'L32'
    elif lamb == 32:
        return 'L64'
    elif lamb == 64:
        return 'L128'
    else:
        print("cannot find lambda : %d"%(lamb))
        exit(0)


def conv2d_same_padding(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    # 函数中padding参数可以无视，实际实现的是padding=same的效果
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        input = torch.pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(input, weight, bias, stride,
                  padding=(padding_rows // 2, padding_cols // 2),
                  dilation=dilation, groups=groups)
