
# -----------------------------------------------------------------------------------
# SwinIR: Image Restoration Using Swin Transformer, https://arxiv.org/abs/2108.10257
# Originally Written by Ze Liu, Modified by Jingyun Liang.
# -----------------------------------------------------------------------------------

import math
import torch
import functools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn.utils import spectral_norm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from SR_scripts.swinIR_support_files.models_support_blocks import *
from models_replace_conv_first import *
import matplotlib.pyplot as plt
import torchvision

class Upsample_conv(nn.Module):
    """Upsample module.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):# scale = 2^n
        super().__init__()
        self.model= nn.Sequential(nn.ConvTranspose2d(num_feat, num_feat, kernel_size= 4, stride=2, padding= 1),
                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                 nn.ConvTranspose2d(num_feat, num_feat, kernel_size= 4, stride=2, padding= 1),
                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                 nn.ConvTranspose2d(num_feat, num_feat, kernel_size= 4, stride=2, padding= 1),
                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                 nn.Conv2d(num_feat, num_feat, kernel_size= 3, stride=1, padding= 1),
                                 nn.LeakyReLU(negative_slope=0.2, inplace=True))


    def forward(self, x):
        return self.model(x)
        


class SwinIRlikeConvDecoder_noPixelShuffle(nn.Module):
    r""" SwinIR
        A PyTorch impl of : `SwinIR: Image Restoration Using Swin Transformer`, based on Swin Transformer.
    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self, img_size=64, patch_size=1, in_chans=3,
                 embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1., upsampler='', resi_connection='1conv',
                 conv_first= None,
                 force_last_residual= False,
                 **kwargs):
        super().__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        
        self.img_range = img_range
        self.conv_first_func= conv_first
        self.force_last_residual= force_last_residual
        self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        
        #####################################################################################################
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        self.forward_features_model= nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim//2, kernel_size= 3, stride=1, padding= 1),
            nn.Conv2d(embed_dim//2, embed_dim, kernel_size= 3, stride=1, padding= 1))
        self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                             nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                             nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                             nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                             nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        
        self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
        self.upsample = Upsample_conv(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.apply(self._init_weights)
    

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x= self.forward_features_model(x)
        return x

    def forward(self, x, residual=None):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        x_input= x.clone()
        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x)) + x
        x = self.conv_before_upsample(x)
        x = self.conv_last(self.upsample(x))

        return x[:, :, :H*self.upscale, :W*self.upscale]

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops

########################


class Discriminator_UNet(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)"""

    def __init__(self, input_nc=3, ndf=64):
        super(Discriminator_UNet, self).__init__()
        norm = spectral_norm

        self.conv0 = nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1)

        self.conv1 = norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(ndf * 8, ndf * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(ndf * 4, ndf * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(ndf * 2, ndf, 3, 1, 1, bias=False))

        # extra
        self.conv7 = norm(nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False))

        self.conv9 = nn.Conv2d(ndf, 1, 3, 1, 1)

    def forward(self, x):
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        x6 = x6 + x0

        # extra
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out

if __name__ == '__main__':
    upscale = 4
    window_size = 8
    height = (1024 // upscale // window_size + 1) * window_size
    width = (720 // upscale // window_size + 1) * window_size
    model = SwinIR(upscale=2, img_size=(height, width),
                   window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')

    x = torch.randn((1, 3, height, width))
    print(x.shape)
    x = model(x)
    print(x.shape)

    x = torch.rand(1, 3, 128, 128)
    net = Discriminator_UNet()
    net.eval()
    with torch.no_grad():
        y = net(x)
    print(y.size())