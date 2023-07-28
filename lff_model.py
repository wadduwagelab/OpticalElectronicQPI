import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt

class fourier_layer(nn.Module):
    def __init__(self, n_neurons_input, n_neurons_output, neuron_size, learn_type='both', device= 'cpu', weights= None, circular = False, **kwargs):
        super(fourier_layer, self).__init__()
        self.n_i               = n_neurons_input
        self.n_o               = n_neurons_output
        self.neuron_size       = neuron_size
        self.learn_type        = learn_type
        self.circular          = circular # if the filter is circular or not
        
        if weights!= None:
            amp_weights= weights['amp_weights']
            phase_weights= weights['phase_weights']    
            
        if (self.learn_type=='amp'):
            print('Learnable transmission coefficient: Amplitude only')
            if weights== None:
                self.amp_weights = nn.Parameter(torch.randn((self.n_i, self.n_i), dtype= torch.float), requires_grad= True)
                self.phase_weights= torch.zeros((self.n_i, self.n_i), dtype= torch.float).to(device)
            else:
                print('loading weights ... ')
                self.amp_weights = nn.Parameter(amp_weights, requires_grad= True)
                self.phase_weights= phase_weights.to(device)   
        elif (self.learn_type=='phase'):
            print('Learnable transmission coefficient: Phase only')
            if weights== None:
                self.amp_weights = torch.ones((self.n_i, self.n_i), dtype= torch.float).to(device) *100000
                self.phase_weights= nn.Parameter(torch.randn((self.n_i, self.n_i), dtype= torch.float), requires_grad= True)
            else:
                print('loading weights ... ')
                self.amp_weights = amp_weights.to(device)
                self.phase_weights= nn.Parameter(phase_weights, requires_grad= True)
                
        elif (self.learn_type=='both'):
            print('Learnable transmission coefficient: Amplitude and Phase')
            if weights== None:
                self.phase_weights= nn.Parameter(torch.randn((self.n_i, self.n_i), dtype= torch.float), requires_grad= True)
                self.amp_weights = nn.Parameter(torch.randn((self.n_i, self.n_i), dtype= torch.float), requires_grad= True)
            else:
                print('loading weights ... ')
                self.phase_weights= nn.Parameter(phase_weights, requires_grad= True)
                self.amp_weights = nn.Parameter(amp_weights, requires_grad= True)
        else:
            print('No learnable transmission coefficients')
            if weights== None:
                self.phase_weights= torch.zeros((self.n_i, self.n_i), dtype= torch.float).to(device)
                self.amp_weights = torch.ones((self.n_i, self.n_i), dtype= torch.float).to(device)*100000
            else:
                print('loading weights ... ')
                self.phase_weights= phase_weights.to(device)
                self.amp_weights = amp_weights.to(device)*100000  
                
    def forward(self, input_e_field):
        device = input_e_field.device
        batch_size = input_e_field.shape[0]
        
        ts = (torch.sigmoid(self.amp_weights) * torch.exp(1j*self.phase_weights)).view(1, self.n_i, self.n_i)
        if self.circular:
            rc = self.n_i//2
            xc = torch.arange(-self.n_i//2,self.n_i//2,1) 
            xc = torch.tile(xc, (1,self.n_i)).view(self.n_i,self.n_i).to(torch.cfloat)

            yc = torch.arange(self.n_i//2,-self.n_i//2,-1).view(self.n_i,1)
            yc = torch.tile(yc, (1,self.n_i)).view(self.n_i,self.n_i).to(torch.cfloat)

            circ = (abs(xc)**2 + abs(yc)**2 <= (rc)**2).to(torch.float32).view(1,self.n_i,self.n_i).to(device)
            
            ts = ts * circ
        
        input_e_field = input_e_field.view(batch_size, self.n_i, self.n_i)
        
        output_e_field = input_e_field * ts

        return output_e_field

class fourier_model(nn.Module):
    def __init__(self, cfg, layer= fourier_layer):
        super(fourier_model, self).__init__()
        
        self.n_i = cfg['img_size']
        self.n_o= cfg['img_size']
        self.neuron_size= cfg['neuron_size']
        self.learn_type= cfg['learn_type']
        self.device= cfg['device']
        self.n_layers= 1
        
        if 'filter_circular' not in cfg.keys():cfg['filter_circular']= True
        if 'weights' not in cfg.keys():cfg['weights'] =  None
            
        self.circular = cfg['filter_circular']
        self.weights = cfg['weights']
        
        if self.weights== None:
            self.weights= {}
            for idx in range(self.n_layers):
                self.weights[f'layer{idx}']= None
        
        n_hidden= (self.n_i+ self.n_o)//2
  
        self.layer_blocks: nn.ModuleList[layer] = nn.ModuleList()
            
        self.layer_blocks.append(layer(self.n_i, n_hidden, neuron_size= self.neuron_size, learn_type=self.learn_type, device= self.device, weights= self.weights[f'layer{self.n_layers-1}'], circular = self.circular))
        
    def forward(self, input_e_field):
        x= input_e_field.view(-1, self.n_i, self.n_i)
        device = input_e_field.device

        Fs = torch.fft.fft2(x)
        X = torch.fft.fftshift(Fs)
            
        for idx in range(len(self.layer_blocks)):
            X= self.layer_blocks[idx](X)
        
        x_o = torch.fft.ifft2(torch.fft.ifftshift(X))

        return x_o.abs().float()**2
    


class final_model(nn.Module):
    def __init__(self, cfg, decoder_name):
        super().__init__()
        self.optics_model= fourier_model(cfg) 
        #self.optics_model= d2nnASwWindow(cfg)
        
    def forward(self, x):
        x = self.optics_model(x)
        x= x.abs().float()**2
        return x