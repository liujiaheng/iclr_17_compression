import numpy as np
import os
import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import math
import torch.nn.init as init
import logging
from torch.nn.parameter import Parameter
from models import *


def save_model(model, iter, name):
    torch.save(model.state_dict(), os.path.join(name, "iter_{}.pth.tar".format(iter)))


def load_model(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter_') != -1 and f.find('.pth') != -1:
        st = f.find('iter_') + 5
        ed = f.find('.pth', st)
        return int(f[st:ed])
    else:
        return 0


class ImageCompressor(nn.Module):
    def __init__(self, out_channel_N=128, lambda_num=8192):
        super(ImageCompressor, self).__init__()
        self.Encoder = Analysis_net_17(out_channel_N=out_channel_N)
        self.Decoder = Synthesis_net_block(out_channel_N=out_channel_N)
        self.bitEstimator = BitEstimator(channel=out_channel_N)
        self.out_channel_N = out_channel_N
        self.lambda_num = lambda_num

    def getrd(self, input_image, recon_image, quant_feature, k):
        batch_size = input_image.size()[0]
        # distortion
        mse_loss = torch.mean((recon_image - input_image).pow(2))
        mseloss = F.avg_pool2d(torch.unsqueeze(torch.mean((recon_image - input_image).pow(2), 1), 1), kernel_size=k, stride=k) * k * k

        def iclr18_estimate_bits_z(z):
            prob = self.bitEstimator(z + 0.5) - self.bitEstimator(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, prob

        total_bits_feature, _ = iclr18_estimate_bits_z(quant_feature)
        im_shape = input_image.size()
        bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
        rd_loss = self.lambda_num * (mseloss) + bpp_feature
        return mse_loss, bpp_feature, rd_loss

    def round(self, input_image, feature, upsample=1):
        quant_noise_feature = torch.zeros(input_image.size(0), self.out_channel_N, input_image.size(2) // 16 // upsample,
                                          input_image.size(3) // 16 // upsample).cuda()
        quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature), -0.5, 0.5)
        if self.training:
            feature = feature + quant_noise_feature
        else:
            feature = torch.round(feature)
        return feature

    def forward(self, input_image):
        feature = self.Encoder(input_image)
        x1 = feature
        x2 = F.avg_pool2d(x1, 2, 2)
        x3 = F.avg_pool2d(x2, 2, 2)
        quant_feat1 = self.round(input_image, x1)
        quant_feat2 = self.round(input_image, x2)
        quant_feat3 = self.round(input_image, x3)
        quant_mv_upsample1 = self.Decoder.forwardseprate1(quant_feat1)
        quant_mv_upsample2 = self.Decoder.forwardseprate2(quant_feat2)
        quant_mv_upsample3 = self.Decoder.forwardseprate3(quant_feat3)

        _, _,  rdc1 = self.getrd(input_image, quant_mv_upsample1, quant_feat1, 2 ** 6)
        _, _, rdc2 = self.getrd(input_image, quant_mv_upsample2, quant_feat2, 2 ** 6)
        _, _, rdc3 = self.getrd(input_image, quant_mv_upsample3, quant_feat3, 2 ** 6)

        rdall = torch.cat([rdc1, rdc2, rdc3], 1)
        rdidx = torch.argmin(rdall, dim=1, keepdim=True)

        mask1 = F.upsample((rdidx == 0).float(), size=(quant_feat1.size()[2], quant_feat1.size()[3]), mode='nearest')
        mask2 = F.upsample((rdidx == 1).float(), size=(quant_feat2.size()[2], quant_feat2.size()[3]), mode='nearest')
        mask3 = F.upsample((rdidx == 2).float(), size=(quant_feat3.size()[2], quant_feat3.size()[3]), mode='nearest')
        recon_image = self.Decoder(quant_feat1 * mask1, quant_feat2 * mask2, quant_feat3 * mask3)
        clipped_recon_image = recon_image.clamp(0., 1.)
        mse_loss = torch.mean((recon_image - input_image).pow(2))

        def iclr18_estrate_bits_mv(mv, mask):
            prob = self.bitEstimator_mv(mv + 0.5) - self.bitEstimator_mv(mv - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50) * mask)
            return total_bits, prob

        total_bits_mv1, _ = iclr18_estrate_bits_mv(quant_feat1, mask1)
        total_bits_mv2, _ = iclr18_estrate_bits_mv(quant_feat2, mask2)
        total_bits_mv3, _ = iclr18_estrate_bits_mv(quant_feat3, mask3)
        total_bits_mv = total_bits_mv1 + total_bits_mv2 + total_bits_mv3
        im_shape = input_image.size()
        bpp_feature = total_bits_mv / (im_shape[0] * im_shape[2] * im_shape[3])
        return clipped_recon_image, mse_loss, bpp_feature
