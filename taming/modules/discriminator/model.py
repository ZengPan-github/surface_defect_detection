import functools
import torch.nn as nn

from torch.nn.utils.parametrizations import spectral_norm
from taming.modules.util import ActNorm


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def add_sn(m):
    for name, layer in m.named_children():
        m.add_module(name, add_sn(layer))
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            return spectral_norm(m)
        else:
            return m


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=1, ndf=64, n_layers=3):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        
        kw = 3
        padw = 1
        
        sequence = [spectral_norm(nn.Conv2d(input_nc, ndf, 1, 1))]
        sequence += [spectral_norm(nn.Conv2d(ndf, ndf, kw, 2, padw,padding_mode='reflect',bias=False)),
                     nn.GroupNorm(num_groups=ndf//16, num_channels=ndf, eps=1e-6, affine=True), 
                     nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, 2, padw, padding_mode='reflect',bias=False)),
                         nn.GroupNorm(num_groups=ndf * nf_mult//16, num_channels=ndf * nf_mult, eps=1e-6, affine=True), 
                         nn.LeakyReLU(0.2, True)]
            
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        
        sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, 2, padw,padding_mode='reflect',bias=False)),
                     nn.GroupNorm(num_groups=ndf * nf_mult//16, num_channels=ndf * nf_mult, eps=1e-6, affine=True),
                     nn.LeakyReLU(0.2, True)]
        

        sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult, ndf, 1, 1)),
                     nn.GroupNorm(num_groups=ndf//16, num_channels=ndf, eps=1e-6, affine=True),
                     nn.LeakyReLU(0.2, True),
                     spectral_norm(nn.Conv2d(ndf, 1, 1, 1))]  # output 1 channel prediction map
        
        
        self.main = nn.Sequential(*sequence)

    def forward(self, inputs):
        """Standard forward."""
        return self.main(inputs)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, loss_type='ls'):
        """Construct a 1x1 PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        
        use_bias = False

        sequence = [
            spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0)),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ndf * 2, ndf, kernel_size=1, stride=1, padding=0, bias=use_bias)),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ndf, 1, kernel_size=1, stride=1, padding=0, bias=use_bias))]
        
        if loss_type == 'ls':
            sequence += [nn.Sigmoid()]
        self.net = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)