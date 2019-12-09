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
    def __init__(self, out_channel_N=128):
        super(ImageCompressor, self).__init__()
        self.Encoder = Analysis_net_17(out_channel_N=out_channel_N)
        self.Decoder = Synthesis_net_17(out_channel_N=out_channel_N)
        self.bitEstimator = BitEstimator(channel=out_channel_N)
        self.out_channel_N = out_channel_N

    def forward(self, input_image):
        quant_noise_feature = torch.zeros(input_image.size(0), self.out_channel_N, input_image.size(2) // 16, input_image.size(3) // 16).cuda()
        quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature), -0.5, 0.5)
        feature = self.Encoder(input_image)
        batch_size = feature.size()[0]
        feature_renorm = feature
        if self.training:
            compressed_feature_renorm = feature_renorm + quant_noise_feature
        else:
            compressed_feature_renorm = torch.round(feature_renorm)
        recon_image = self.Decoder(compressed_feature_renorm)
        # recon_image = prediction + recon_res
        clipped_recon_image = recon_image.clamp(0., 1.)
        # distortion
        mse_loss = torch.mean((recon_image - input_image).pow(2))

        # def feature_probs_based_sigma(feature, sigma):
        #     mu = torch.zeros_like(sigma)
        #     sigma = sigma.clamp(1e-10, 1e10)
        #     gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        #     probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
        #     total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50))
        #     return total_bits, probs

        def iclr18_estimate_bits_z(z):
            prob = self.bitEstimator(z + 0.5) - self.bitEstimator(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, prob

        total_bits_feature, _ = iclr18_estimate_bits_z(compressed_feature_renorm)
        im_shape = input_image.size()
        bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])

        return clipped_recon_image, mse_loss, bpp_feature
