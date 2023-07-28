import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt

def do_downsample(img_batch, upscale): #img_batch: (b, 1, 32, 32)
    n_pool = int(np.log2(upscale))
    for _ in range(n_pool):
        img_batch = F.avg_pool2d(img_batch, kernel_size= 2)
    return img_batch

def show_results_SwinIRTraining(gt_angle, optical_out, decoder_out, loss, description, save_dir):
    vmin= min(gt_angle[0].detach().cpu().min(), decoder_out[0].detach().cpu().min())
    vmax= max(gt_angle[0].detach().cpu().max(), decoder_out[0].detach().cpu().max())

    
    plt.figure(figsize= (14,3))
    plt.subplot(1,4,1)
    plt.imshow(gt_angle[0].detach().cpu(), vmin= vmin, vmax= vmax)
    plt.title('gt(angle)')
    plt.colorbar()
    
    plt.subplot(1,4,2)
    plt.imshow(optical_out[0].detach().cpu())
    plt.title('out(optical)')
    plt.colorbar()
    
    plt.subplot(1,4,3)
    plt.imshow(decoder_out[0].detach().cpu(), vmin= vmin, vmax= vmax)
    plt.title('out(final:SR)')
    plt.colorbar()
    
    plt.subplot(1,4,4)
    plt.plot(list(range(1, len(loss)+1)), loss)
    plt.title(f'loss (final: {np.round(loss[-1], 4)})')
    
    plt.suptitle(description, y= 1.03)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{description}.pdf', bbox_inches='tight')
    plt.show()