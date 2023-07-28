from torch import nn
import torch
import numpy as np

class upsample_transconv_relu_bn_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upsample_transconv_relu_bn_block, self).__init__()
        self.seq= nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size= 4, padding=1, stride=2),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels))
    def forward(self, x):
        return self.seq(x)  
    
import torch
from torch import nn
from torch.nn import functional as F
import math
import numpy as np

    
class conv_relu_bn_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(conv_relu_bn_block, self).__init__()
        self.seq= nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size= kernel_size, padding=padding, stride=stride),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels))
    def forward(self, x):
        return self.seq(x)

class swinir_exp_linearEncoder(nn.Module):
    def __init__(self, num_in_ch, embed_dim, img_size= 256, n_layers= 8):
        super(swinir_exp_linearEncoder, self).__init__()                
        self.downsample_blocks: nn.ModuleList[nn.Conv2d] = nn.ModuleList()
        self.avgpool= nn.AvgPool2d(2)
        
        self.embed_dim= embed_dim
        self.output_size= img_size// (2**n_layers)
        
        for i in range(n_layers):
            if i==0:
                in_channels= num_in_ch
                out_channels= embed_dim//2
            elif i== n_layers-1:
                in_channels= embed_dim//2
                out_channels= embed_dim
            else:
                in_channels= embed_dim//2
                out_channels= embed_dim//2
                
            self.downsample_blocks.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1))
            
        print(f'LinearEncoder :: output shape (verified in forward pass): (b, {self.embed_dim}, {self.output_size}, {self.output_size})')
    def forward(self, x):
        batch_size= x.shape[0]
        for i in range(len(self.downsample_blocks)):
            x = self.downsample_blocks[i](x)
            x= self.avgpool(x)
        
        return x.view(batch_size, self.embed_dim, self.output_size, self.output_size) 
    
class swinir_exp_TransposeConvDecoder(nn.Module): #final_embed_dim, latent_embed_dim
    def __init__(self, final_embed_dim, embed_dim, img_size= 256, n_layers= 8):
        super(swinir_exp_TransposeConvDecoder, self).__init__() 
        self.upsample_blocks: nn.ModuleList[upsample_transconv_relu_bn_block] = nn.ModuleList()
        
        self.final_embed_dim= final_embed_dim
        self.output_size= img_size
        
        for i in range(n_layers):
            if i==0:
                in_channels= embed_dim
                out_channels= embed_dim//2
            elif i== n_layers-1:
                in_channels= embed_dim//2
                out_channels= final_embed_dim
            else:
                in_channels= embed_dim//2
                out_channels= embed_dim//2
                
            self.upsample_blocks.append(upsample_transconv_relu_bn_block(in_channels, out_channels))
        print(f'LinearEncoderTransposeConvDecoder --> TransposeConvDecoder :: output shape (verified in forward pass): (b, {self.final_embed_dim}, {self.output_size}, {self.output_size}')

    def forward(self, x):
        batch_size= x.shape[0]
        for i in range(len(self.upsample_blocks)):
            x = self.upsample_blocks[i](x)        
        return x.view(batch_size, self.final_embed_dim, self.output_size, self.output_size)
    
class custom_v2(nn.Module):
    """
    Idea: the same concept in "custom_v1" but implemented in very much efficient way
    """
    def __init__(self, n_output_channels, n_input_channels, recon_img_size, upscale_factor, init_method= 'xavier_normal'):
        super(custom_v2, self).__init__()
        
        self.n_input_channels= n_input_channels
        self.n_output_channels= n_output_channels
        self.upscale_factor= upscale_factor #upscale factor for a single side -> (eg: x2, x4, x8, ...)

        self.recon_img_size= recon_img_size
        self.init_method=  init_method
        self.bias= True
        
        self.yt_img_size= self.recon_img_size//self.upscale_factor
                
        if self.n_input_channels!=1:
            self.seq_block= nn.Sequential(conv_relu_bn_block(in_channels= 1, out_channels= self.n_output_channels//2, kernel_size= 3, padding= 1, stride=1),
                                          conv_relu_bn_block(in_channels= self.n_output_channels//2, out_channels= self.n_output_channels, kernel_size= 3, padding= 1, stride=1))
        else:
            self.seq_block= nn.Sequential(conv_relu_bn_block(in_channels= 1, out_channels= self.n_output_channels, kernel_size= 3, padding= 1, stride=1),
                                          conv_relu_bn_block(in_channels= self.n_output_channels, out_channels= self.n_output_channels, kernel_size= 3, padding= 1, stride=1))

        self.weights= nn.Parameter(torch.randn(self.yt_img_size* self.yt_img_size, self.n_input_channels, self.upscale_factor**2), requires_grad= True)
        if self.bias:self.biases= nn.Parameter(torch.randn(self.yt_img_size* self.yt_img_size, 1, self.upscale_factor**2), requires_grad= True)
        
        if self.init_method== 'linear_default':
            stdv = 1. / math.sqrt(self.weights.size(1))
            self.weights.data.uniform_(-stdv, stdv)
            if self.bias:self.biases.data.uniform_(-stdv, stdv)

        elif self.init_method=='xavier_normal':
            torch.nn.init.xavier_normal_(self.weights)
            if self.bias:torch.nn.init.zeros_(self.biases)

        
    def forward(self, yt):
        batch_size= yt.shape[0]
        yt_input = yt.view(batch_size, self.n_input_channels, self.yt_img_size, self.yt_img_size)
        yt= yt_input.reshape(batch_size, self.n_input_channels, self.yt_img_size* self.yt_img_size).permute(2, 0, 1)  #shape: (yt_img_size**2, batch_size, T)

        if self.bias:
            yt_upsample= torch.matmul(yt, self.weights) + self.biases  #shape: (yt_img_size**2, batch_size, upscale_factor**2)
        else:yt_upsample= torch.matmul(yt, self.weights)  #shape: (yt_img_size**2, batch_size, upscale_factor**2)
            
        yt_upsample = yt_upsample.view(self.yt_img_size, self.yt_img_size, batch_size, self.upscale_factor, self.upscale_factor).permute(2, 0, 3, 1, 4) # shape: (batch_size, yt_img_size, upscale_factor, yt_img_size, upscale_factor)
        yt_upsample= yt_upsample.reshape(batch_size, self.recon_img_size, self.recon_img_size).unsqueeze(dim= 1) # shape: (batch_size, 1, yt_img_size*upscale_factor, yt_img_size*upscale_factor)
        output= self.seq_block(yt_upsample) # shape: (batch_size, n_output_channels, recon_img_size, recon_img_size)    
        return output
    
    
class LinearEncoderTransposeConvDecoder(nn.Module):
    def __init__(self, num_in_ch, final_embed_dim, latent_embed_dim, in_img_size, out_img_size, n_layers_enc):
        super().__init__()
        self.encoder= swinir_exp_linearEncoder(num_in_ch, latent_embed_dim, in_img_size, n_layers_enc)
        latent_side_len = in_img_size// 2**n_layers_enc
        n_layers_dec= int(np.log2(out_img_size // latent_side_len))
        
        print(f'LinearEncoderTransposeConvDecoder --> n_layers_enc = {n_layers_enc}, n_layers_dec = {n_layers_dec}')
        print(f'LinearEncoderTransposeConvDecoder --> in_img_size = {in_img_size}, out_img_size = {out_img_size}')
        self.decoder= swinir_exp_TransposeConvDecoder(final_embed_dim, latent_embed_dim, out_img_size, n_layers_dec)
        
    def forward(self, x):
        embeds= self.encoder(x)
        spatial_embeds= self.decoder(embeds)
        return spatial_embeds
    

    
class LinearEncoderCustomV2Decoder(nn.Module):
    def __init__(self, num_in_ch, final_embed_dim, latent_embed_dim, in_img_size, out_img_size, n_layers_enc):
        super().__init__()
        self.encoder= swinir_exp_linearEncoder(num_in_ch, latent_embed_dim, in_img_size, n_layers_enc)
        latent_side_len = in_img_size// 2**n_layers_enc
        
        print(f'LinearEncoderCustomV2Decoder --> n_layers_enc = {n_layers_enc}')
        print(f'LinearEncoderCustomV2Decoder --> in_img_size = {in_img_size}, out_img_size = {out_img_size}')
        self.decoder= custom_v2(final_embed_dim, latent_embed_dim, out_img_size, upscale_factor= out_img_size // latent_side_len)
                
    def forward(self, x):
        embeds= self.encoder(x)
        spatial_embeds= self.decoder(embeds)
        return spatial_embeds
    
class LinearEncoderCustomV2ifftRealDecoder(nn.Module):
    def __init__(self, num_in_ch, final_embed_dim, latent_embed_dim, in_img_size, out_img_size, n_layers_enc):
        super().__init__()
        self.encoder= swinir_exp_linearEncoder(num_in_ch, latent_embed_dim, in_img_size, n_layers_enc)
        latent_side_len = in_img_size// 2**n_layers_enc
        
        print(f'LinearEncoderCustomV2Decoder --> n_layers_enc = {n_layers_enc}')
        print(f'LinearEncoderCustomV2Decoder --> in_img_size = {in_img_size}, out_img_size = {out_img_size}')
        self.decoder= custom_v2(final_embed_dim, latent_embed_dim, out_img_size, upscale_factor= out_img_size // latent_side_len)
                
    def forward(self, x):
        embeds= self.encoder(x)
        fft_centered= self.decoder(embeds) # learns freq-domain
        #spatial_embeds = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(fft_centered))).abs().float()
        spatial_embeds = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(fft_centered))).real

        return spatial_embeds
    
    
    
class LinearEncoderCustomV2ifftAbsDecoder(nn.Module):
    def __init__(self, num_in_ch, final_embed_dim, latent_embed_dim, in_img_size, out_img_size, n_layers_enc):
        super().__init__()
        self.encoder= swinir_exp_linearEncoder(num_in_ch, latent_embed_dim, in_img_size, n_layers_enc)
        latent_side_len = in_img_size// 2**n_layers_enc
        
        print(f'LinearEncoderCustomV2Decoder --> n_layers_enc = {n_layers_enc}')
        print(f'LinearEncoderCustomV2Decoder --> in_img_size = {in_img_size}, out_img_size = {out_img_size}')
        self.decoder= custom_v2(final_embed_dim, latent_embed_dim, out_img_size, upscale_factor= out_img_size // latent_side_len)
                
    def forward(self, x):
        embeds= self.encoder(x)
        fft_centered= self.decoder(embeds) # learns freq-domain
        spatial_embeds = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(fft_centered))).abs().float()
        #spatial_embeds = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(fft_centered))).real

        return spatial_embeds