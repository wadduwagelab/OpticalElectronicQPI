
from torch.nn import init
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam
import yaml

from SR_scripts.swinIR_support_files.utils_regularizers import regularizer_orth, regularizer_clip
from SR_scripts.swinIR_support_files.models_simple import *
from SR_scripts.swinIR_support_files.losses import *
from SR_scripts.swinIR_support_files.ModelBase import ModelBase
from SR_scripts.swinIR_support_files import utils_options

## for optical-complex encoder
from modules.nonoptical_encoders import *
from modules.fourier_model import *
from modules.d2nn_models import *


def define_G(opt):
    opt_net = opt['netG']
    if 'conv_first' not in opt:
        opt['conv_first']= 'nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)'
        
        to_print= opt['conv_first']
        print(f'"conv_first" not defined in opts --> "{to_print}" is used')
        
    if 'force_last_residual' not in opt:
        opt['force_last_residual']= False
        
        to_print= opt['force_last_residual']
        print(f'"force_last_residual" not defined in opts --> "{to_print}" is used')
        
    
    netG = SwinIRlikeConvDecoder_noPixelShuffle(upscale=opt_net['upscale'],
                    in_chans=opt_net['in_chans'],
                    img_size=opt_net['img_size'],
                    window_size=opt_net['window_size'],
                    img_range=opt_net['img_range'],
                    depths=opt_net['depths'],
                    embed_dim=opt_net['embed_dim'],
                    num_heads=opt_net['num_heads'],
                    mlp_ratio=opt_net['mlp_ratio'],
                    upsampler=opt_net['upsampler'],
                    resi_connection=opt_net['resi_connection'],
                    conv_first = opt['conv_first'],
                    force_last_residual= opt['force_last_residual'])
    if opt['is_train']:
        init_weights(netG,
                        init_type=opt_net['init_type'],
                        init_bn_type=None, #opt_net['init_bn_type'],
                        gain=None) #opt_net['init_gain'])
    return netG

def define_D(opt):
    opt_net = opt['netD']
    netD = Discriminator_UNet(input_nc=opt_net['in_nc'],
                            ndf=opt_net['base_nc'])
    init_weights(netD,
                    init_type=opt_net['init_type'],
                    init_bn_type=opt_net['init_bn_type'],
                    gain=opt_net['init_gain'])
    return netD


def init_weights(net, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
    """
    # Kai Zhang, https://github.com/cszn/KAIR
    #
    # Args:
    #   init_type:
    #       default, none: pass init_weights
    #       normal; normal; xavier_normal; xavier_uniform;
    #       kaiming_normal; kaiming_uniform; orthogonal
    #   init_bn_type:
    #       uniform; constant
    #   gain:
    #       0.2
    """

    def init_fn(m, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
        classname = m.__class__.__name__

        if classname.find('Conv') != -1 or classname.find('Linear') != -1:

            if init_type == 'normal':
                init.normal_(m.weight.data, 0, 0.1)
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'uniform':
                init.uniform_(m.weight.data, -0.2, 0.2)
                m.weight.data.mul_(gain)

            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight.data, gain=gain)
                m.weight.data.clamp_(-1, 1)

            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain)

            elif init_type == 'kaiming_normal':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.mul_(gain)

            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)

            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_type))

            if m.bias is not None:
                m.bias.data.zero_()

        elif classname.find('BatchNorm2d') != -1:

            if init_bn_type == 'uniform':  # preferred
                if m.affine:
                    init.uniform_(m.weight.data, 0.1, 1.0)
                    init.constant_(m.bias.data, 0.0)
            elif init_bn_type == 'constant':
                if m.affine:
                    init.constant_(m.weight.data, 1.0)
                    init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_bn_type))

    if init_type not in ['default', 'none']:
        print('Initialization method [{:s} + {:s}], gain is [{:.2f}]'.format(init_type, init_bn_type, gain))
        fn = functools.partial(init_fn, init_type=init_type, init_bn_type=init_bn_type, gain=gain)
        net.apply(fn)
    else:
        print('Pass this initialization! Initialization was done during network defination!')

########################### FINAL MODEL

class swinIR_generative_model(nn.Module):
    def __init__(self, opt_file_path= 'swinIR_support_files/opt.yaml', prev_cfg= None):
        super(swinIR_generative_model, self).__init__()

        with open(opt_file_path, 'r') as file:
            opt = yaml.load(file)
        opt.update(prev_cfg)
        self.opt = opt                         # opt
        self.device = prev_cfg['device'] #torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']        # training or not
        self.schedulers = []                   # schedulers

        self.opt_train = self.opt['train']    # training option
        self.netG = define_G(opt)
        self.netG = self.model_to_device(self.netG)
        if self.is_train:
            self.netD = define_D(opt)
            self.netD = self.model_to_device(self.netD)
            if self.opt_train['E_decay'] > 0:
                self.netE = define_G(opt).to(self.device).eval()
                
                
        #self.optical_encoder_model= eval(prev_cfg['encoder'])(prev_cfg).to(prev_cfg['device'])

    def update_learning_rate(self, n):
        for scheduler in self.schedulers:
            #scheduler.step(n) # gives depreciation warning
            scheduler.step()

    def current_learning_rate(self):
        return self.schedulers[0].get_lr()[0]

    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def model_to_device(self, network):return network.to(self.device)

    def update_E(self, decay=0.999):
        netG = self.netG
        netG_params = dict(netG.named_parameters())
        netE_params = dict(self.netE.named_parameters())
        for k in netG_params.keys():
            netE_params[k].data.mul_(decay).add_(netG_params[k].data, alpha=1-decay)

    def merge_bnorm_train(self):
        merge_bn(self.netG)
        tidy_sequential(self.netG)
        self.define_optimizer()
        self.define_scheduler()

    def merge_bnorm_test(self):
        merge_bn(self.netG)
        tidy_sequential(self.netG)

    def init_train(self):
        self.load()                           # load model
        self.netG.train()                     # set training mode,for BN
        self.netD.train()                     # set training mode,for BN
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log

    def load(self):
        self.update_E(0)
        self.netE.eval()

    def define_loss(self):
        # ------------------------------------
        # 1) G_loss
        # ------------------------------------
        if self.opt_train['G_lossfn_weight'] > 0:
            G_lossfn_type = self.opt_train['G_lossfn_type']
            if G_lossfn_type == 'l1':
                self.G_lossfn = nn.L1Loss().to(self.device)
            elif G_lossfn_type == 'l2':
                self.G_lossfn = nn.MSELoss().to(self.device)
            elif G_lossfn_type == 'l2sum':
                self.G_lossfn = nn.MSELoss(reduction='sum').to(self.device)
            elif G_lossfn_type == 'ssim':
                self.G_lossfn = SSIMLoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
            self.G_lossfn_weight = self.opt_train['G_lossfn_weight']
        else:
            print('Do not use pixel loss.')
            self.G_lossfn = None

        # ------------------------------------
        # 2) F_loss
        # ------------------------------------
        if self.opt_train['F_lossfn_weight'] > 0:
            F_feature_layer = self.opt_train['F_feature_layer']
            F_weights = self.opt_train['F_weights']
            F_lossfn_type = self.opt_train['F_lossfn_type']
            F_use_input_norm = self.opt_train['F_use_input_norm']
            F_use_range_norm = self.opt_train['F_use_range_norm']
            if self.opt['dist']:
                self.F_lossfn = PerceptualLoss(feature_layer=F_feature_layer, weights=F_weights, lossfn_type=F_lossfn_type, use_input_norm=F_use_input_norm, use_range_norm=F_use_range_norm).to(self.device)
            else:
                self.F_lossfn = PerceptualLoss(feature_layer=F_feature_layer, weights=F_weights, lossfn_type=F_lossfn_type, use_input_norm=F_use_input_norm, use_range_norm=F_use_range_norm)
                self.F_lossfn.vgg = self.model_to_device(self.F_lossfn.vgg)
                self.F_lossfn.lossfn = self.F_lossfn.lossfn.to(self.device)
            self.F_lossfn_weight = self.opt_train['F_lossfn_weight']
        else:
            print('Do not use feature loss.')
            self.F_lossfn = None

        # ------------------------------------
        # 3) D_loss
        # ------------------------------------
        self.D_lossfn = GANLoss(self.opt_train['gan_type'], 1.0, 0.0).to(self.device)
        self.D_lossfn_weight = self.opt_train['D_lossfn_weight']

        self.D_update_ratio = self.opt_train['D_update_ratio'] if self.opt_train['D_update_ratio'] else 1
        self.D_init_iters = self.opt_train['D_init_iters'] if self.opt_train['D_init_iters'] else 0

    # ----------------------------------------
    # define optimizer, G and D
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))

        self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'], weight_decay=0)
        self.D_optimizer = Adam(self.netD.parameters(), lr=self.opt_train['D_optimizer_lr'], weight_decay=0)

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                        self.opt_train['G_scheduler_milestones'],
                                                        self.opt_train['G_scheduler_gamma']
                                                        ))
        self.schedulers.append(lr_scheduler.MultiStepLR(self.D_optimizer,
                                                        self.opt_train['D_scheduler_milestones'],
                                                        self.opt_train['D_scheduler_gamma']
                                                        ))

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        self.L = data['L'].to(self.device)
        if need_H:
            self.H = data['H'].to(self.device)
        if 'residual' in data:
            self.Residuals= data['residual']
            
            if data['residual']!=None:self.Residuals= self.Residuals.to(self.device)
        else:
            self.Residuals= None

    # ----------------------------------------
    # feed L to netG and get E
    # ----------------------------------------
    
    #def optical_encoder(self, complex_img_batch): #complex_img_batch: (b, 1, img_size, img_size)
    #    input_field= complex_img_batch[:, 0]
    #    output_field = self.optical_encoder_model(input_field)
    #    output_detections = (output_field.abs().float())**2
    #    return output_detections.unsqueeze(dim= 1)
    
    def netG_forward(self):
        #self.E = self.netG(self.optical_encoder(self.L))
        self.E = self.netG(self.L, self.Residuals)

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step, other_optimizers=None):
        # ------------------------------------
        # optimize G
        # ------------------------------------
        for p in self.netD.parameters():
            p.requires_grad = False

        self.G_optimizer.zero_grad()
        self.netG_forward()
        loss_G_total = 0

        if current_step % self.D_update_ratio == 0 and current_step > self.D_init_iters:  # updata D first
            if self.opt_train['G_lossfn_weight'] > 0:
                G_loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.H)
                loss_G_total += G_loss                 # 1) pixel loss
                
                self.log_G_loss = G_loss
            if self.opt_train['F_lossfn_weight'] > 0:
                F_loss = self.F_lossfn_weight * self.F_lossfn(self.E, self.H)
                loss_G_total += F_loss                 # 2) VGG feature loss

                self.log_F_loss = F_loss
            if self.opt['train']['gan_type'] in ['gan', 'lsgan', 'wgan', 'softplusgan']:
                pred_g_fake = self.netD(self.E)
                D_loss = self.D_lossfn_weight * self.D_lossfn(pred_g_fake, True)
            elif self.opt['train']['gan_type'] == 'ragan':
                pred_d_real = self.netD(self.H).detach()
                pred_g_fake = self.netD(self.E)
                D_loss = self.D_lossfn_weight * (
                        self.D_lossfn(pred_d_real - torch.mean(pred_g_fake, 0, True), False) +
                        self.D_lossfn(pred_g_fake - torch.mean(pred_d_real, 0, True), True)) / 2
            loss_G_total += D_loss                    # 3) GAN loss
            self.log_D_loss = D_loss
            
            loss_G_total.backward()
            self.G_optimizer.step()
            
            if other_optimizers!=None:other_optimizers.step()

        # ------------------------------------
        # optimize D
        # ------------------------------------
        for p in self.netD.parameters():
            p.requires_grad = True

        self.D_optimizer.zero_grad()
        if self.opt_train['gan_type'] in ['gan', 'lsgan', 'wgan', 'softplusgan']:
            # real
            pred_d_real = self.netD(self.H)                # 1) real data
            l_d_real = self.D_lossfn(pred_d_real, True)
            l_d_real.backward()
            # fake
            pred_d_fake = self.netD(self.E.detach().clone()) # 2) fake data, detach to avoid BP to G
            l_d_fake = self.D_lossfn(pred_d_fake, False)
            l_d_fake.backward()
        elif self.opt_train['gan_type'] == 'ragan':
            # real
            pred_d_fake = self.netD(self.E).detach()       # 1) fake data, detach to avoid BP to G
            pred_d_real = self.netD(self.H)                # 2) real data
            l_d_real = 0.5 * self.D_lossfn(pred_d_real - torch.mean(pred_d_fake, 0, True), True)
            l_d_real.backward()
            # fake
            pred_d_fake = self.netD(self.E.detach())
            l_d_fake = 0.5 * self.D_lossfn(pred_d_fake - torch.mean(pred_d_real.detach(), 0, True), False)
            l_d_fake.backward()

        self.D_optimizer.step()


        #self.log_dict['l_d_real'] = l_d_real.item()
        #self.log_dict['l_d_fake'] = l_d_fake.item()
        self.log_dict['D_real'] = torch.mean(pred_d_real.detach())
        self.log_dict['D_fake'] = torch.mean(pred_d_fake.detach())

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])


    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.netG_forward()
        self.netG.train()
        
        
class swinIR_classical_model(ModelBase):
    """Train with pixel loss"""        
    def __init__(self, opt_file_path= 'swinIR_support_files/opt.yaml', prev_cfg= None):
        with open(opt_file_path, 'r') as file:
            opt = yaml.load(file)
        opt = utils_options.parse(opt, is_train=True)
        opt.update(prev_cfg)
        super(swinIR_classical_model, self).__init__(opt)
        
        self.device = prev_cfg['device']

        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt['train']    # training option
        self.netG = define_G(opt)
        self.netG = self.model_to_device(self.netG)
        if self.opt_train['E_decay'] > 0:
            self.netE = define_G(opt).to(self.device).eval()

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.load()                           # load model
        self.netG.train()                     # set training mode,for BN
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.load_optimizers()                # load optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log

    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    def load(self):
        load_path_G = self.opt['path']['pretrained_netG']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict=self.opt_train['G_param_strict'], param_key='params')
        load_path_E = self.opt['path']['pretrained_netE']
        if self.opt_train['E_decay'] > 0:
            if load_path_E is not None:
                print('Loading model for E [{:s}] ...'.format(load_path_E))
                self.load_network(load_path_E, self.netE, strict=self.opt_train['E_param_strict'], param_key='params_ema')
            else:
                print('Copying model for E ...')
                self.update_E(0)
            self.netE.eval()

    # ----------------------------------------
    # load optimizer
    # ----------------------------------------
    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, save_dir):
        iter_label= 'NA'
        self.save_network(save_dir, self.netG, 'G', iter_label)
        if self.opt_train['E_decay'] > 0:
            self.save_network(save_dir, self.netE, 'E', iter_label)
        self.save_optimizer(save_dir, self.G_optimizer, 'optimizerG', iter_label)

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        G_lossfn_type = self.opt_train['G_lossfn_type']
        if G_lossfn_type == 'l1':
            self.G_lossfn = nn.L1Loss().to(self.device)
        elif G_lossfn_type == 'l2':
            self.G_lossfn = nn.MSELoss().to(self.device)
        elif G_lossfn_type == 'l2sum':
            self.G_lossfn = nn.MSELoss(reduction='sum').to(self.device)
        elif G_lossfn_type == 'ssim':
            self.G_lossfn = SSIMLoss().to(self.device)
        elif G_lossfn_type == 'charbonnier':
            self.G_lossfn = CharbonnierLoss(self.opt_train['G_charbonnier_eps']).to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
        self.G_lossfn_weight = self.opt_train['G_lossfn_weight']

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        if self.opt_train['G_optimizer_type'] == 'adam':
            self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'],
                                    betas=self.opt_train['G_optimizer_betas'],
                                    weight_decay=self.opt_train['G_optimizer_wd'])
        else:
            raise NotImplementedError

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        if self.opt_train['G_scheduler_type'] == 'MultiStepLR':
            self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                            self.opt_train['G_scheduler_milestones'],
                                                            self.opt_train['G_scheduler_gamma']
                                                            ))
        elif self.opt_train['G_scheduler_type'] == 'CosineAnnealingWarmRestarts':
            self.schedulers.append(lr_scheduler.CosineAnnealingWarmRestarts(self.G_optimizer,
                                                            self.opt_train['G_scheduler_periods'],
                                                            self.opt_train['G_scheduler_restart_weights'],
                                                            self.opt_train['G_scheduler_eta_min']
                                                            ))
        else:
            raise NotImplementedError

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        self.L = data['L'].to(self.device)
        if need_H:
            self.H = data['H'].to(self.device)

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self):
        self.E = self.netG(self.L)

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()
        self.netG_forward()
        G_loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.H)
        G_loss.backward()

        self.log_G_loss = G_loss
        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'], norm_type=2)

        self.G_optimizer.step()

        # ------------------------------------
        # regularizer
        # ------------------------------------
        G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train['G_regularizer_orthstep'] else 0
        if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_orth)
        G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train['G_regularizer_clipstep'] else 0
        if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_clip)

        # self.log_dict['G_loss'] = G_loss.item()/self.E.size()[0]  # if `reduction='sum'`
        self.log_dict['G_loss'] = G_loss.item()

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.netG_forward()
        self.netG.train()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H image
    # ----------------------------------------
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach()[0].float().cpu()
        out_dict['E'] = self.E.detach()[0].float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach()[0].float().cpu()
        return out_dict

    # ----------------------------------------
    # get L, E, H batch images
    # ----------------------------------------
    def current_results(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach().float().cpu()
        out_dict['E'] = self.E.detach().float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach().float().cpu()
        return out_dict

    """
    # ----------------------------------------
    # Information of netG
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netG)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg