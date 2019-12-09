from .analysis_17 import Analysis_net_17
import torch.nn as nn
import torch.nn.functional as F
from .GDN import GDN
import torch
import math


class Synthesis_net_block(nn.Module):
    '''
    Decode synthesis
    '''

    def __init__(self, out_channel_N=192):
        super(Synthesis_net_block, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, (math.sqrt(2 * 1 )))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.igdn1 = GDN(out_channel_N, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.igdn2 = GDN(out_channel_N, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(out_channel_N, 3, 9, stride=4, padding=4, output_padding=3)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)

    def forward(self, x1, x2, x3):
        x = x3
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = x + x2
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = x + x1
        x = self.igdn1(self.deconv1(x))
        x = self.igdn2(self.deconv2(x))
        x = self.deconv3(x)
        return x

    def forwardseprate3(self, x3):
        x3 = F.interpolate(x3, scale_factor=4, mode='nearest')
        x = self.igdn1(self.deconv1(x3))
        x = self.igdn2(self.deconv2(x))
        x = self.deconv3(x)
        return x


    def forwardseprate2(self, x2):
        x2 = F.interpolate(x2, scale_factor=2, mode='nearest')
        x = self.igdn1(self.deconv1(x2))
        x = self.igdn2(self.deconv2(x))
        x = self.deconv3(x)
        return x

    def forwardseprate1(self, x1):
        x = self.igdn1(self.deconv1(x1))
        x = self.igdn2(self.deconv2(x))
        x = self.deconv3(x)
        return x

def build_model():
    input_image = torch.zeros([7,3,256,256])
    analysis_net = Analysis_net_17()
    synthesis_net = Synthesis_net_block()
    feature = analysis_net(input_image)
    recon_image = synthesis_net(feature)

    print("input_image : ", input_image.size())
    print("feature : ", feature.size())
    print("recon_image : ", recon_image.size())

# def main(_):
#   build_model()


if __name__ == '__main__':
    build_model()
