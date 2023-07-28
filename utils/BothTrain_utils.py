import numpy as np
import torch
import matplotlib.pyplot as plt

from ignite.metrics import SSIM as SSIM_IGNITE
from ignite.metrics import PSNR


def get_similarity_model(weights_model1, weights_model2):
    sim= []
    assert len(weights_model1)==len(weights_model2), 'Weights list lengths should be same !!!'
    for i in range(len(weights_model1)):
        sim.append((weights_model1[i]==weights_model2[i]).astype('float').mean())
    sim= np.round(np.array(sim).mean(), 4)
    return sim

def freeze_params(model, requires_grad= True):
    for name, param in model.named_parameters():
        param.requires_grad= requires_grad
        #print(f'freeze/ unfreeze weights of SwinIR.netG ---> {name} : {param.requires_grad}')
        
def load_models(sr_opt_dir, swinir_saved_dir):
    decoder_name= optical_state_dict_dir.split('(')[-1].split(')')[0]
    #cfg['filter_circular']= False
    optical_model = final_model(cfg, decoder_name).to(device)
    optical_model.load_state_dict(torch.load(optical_state_dict_dir, map_location= device))

    sr_model = swinIR_generative_model(sr_opt_dir, cfg)
    sr_model.init_train()
    sr_model_info= torch.load(swinir_saved_dir, map_location= device)
    sr_model.load_state_dict(sr_model_info['SR_state_dict'])
    return optical_model, sr_model
  
def ssim_grayscale(X_hat, X, k= 11): # shape: BCHW
    metric = SSIM_IGNITE(data_range = 1.0, kernel_size= (k,k))
    metric.update((X_hat, X))
    return metric.compute().item()

def psnr_grayscale(X_hat, X): # shape: BCHW
    metric = PSNR(data_range= 1.0) #(data_range= 235- 16= 219) ->  https://scikit-image.org/docs/dev/api/skimage.color.html#rgb2ycbcr 
    metric.update((X_hat, X))
    return metric.compute().item()

def show_results_BothTraining(gt_angle, optical_out, decoder_out, losses, psnrs, ssims, l1s, description, more_title, save_dir):
    fraction= 0.03
    plt.figure(figsize= (12,6))
    plt.subplot(2,4,1)
    plt.imshow(gt_angle[0].detach().cpu(), vmin=0, vmax=1)
    plt.title('gt (angle)')
    plt.colorbar(fraction=fraction)
    plt.subplot(2,4,2)
    plt.imshow(optical_out[0].detach().cpu(), vmin=0, vmax=1)
    plt.title('optical out')
    plt.colorbar(fraction=fraction)
    plt.subplot(2,4,3)
    plt.imshow(decoder_out[0].detach().cpu(), vmin=0, vmax=1)
    plt.title('final out')
    plt.colorbar(fraction=fraction)
    plt.subplot(2,4,5)
    plt.plot(losses)
    plt.title(f'loss : {np.round(losses[-1], 4)}')
    plt.subplot(2,4,6)
    plt.plot(psnrs)
    plt.title(f'psnr : {np.round(psnrs[-1], 4)}')
    plt.subplot(2,4,7)
    plt.plot(ssims)
    plt.title(f'ssim : {np.round(ssims[-1], 4)}')
    plt.subplot(2,4,8)
    plt.plot(l1s)
    plt.title(f'l1 : {np.round(l1s[-1], 4)}')
    plt.suptitle(description + '|' + more_title)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{description}.pdf', bbox_inches='tight')
    plt.show()