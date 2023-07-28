import numpy as np
import matplotlib.pyplot as plt

def show_results_AllOptical(gt, out_final, losses, description, save_dir):
    out_final_abs= out_final.abs()[0].detach().cpu()
    in_angle_normalized= (gt.angle()[0].detach().cpu()%(2*np.pi))/(2*np.pi)
    
    vmin= min(out_final_abs.min(), in_angle_normalized.min())
    vmax= max(out_final_abs.max(), in_angle_normalized.max())

    plt.figure(figsize= (12, 7))
    plt.subplot(2,3,1)
    plt.imshow(gt.abs()[0].detach().cpu())
    plt.title('gt: abs')
    plt.colorbar()
    
    plt.subplot(2,3,2)
    plt.imshow(in_angle_normalized, vmin= vmin, vmax= vmax)
    plt.title('gt: angle')
    plt.colorbar()

    plt.subplot(2,3,3)
    plt.imshow(gt.angle()[0].detach().cpu())
    plt.title('gt (input): angle')
    plt.colorbar()

    plt.subplot(2,3,4)
    plt.imshow(out_final_abs, vmin= vmin, vmax= vmax)
    plt.title('output: abs')
    plt.colorbar()
    
    plt.subplot(2,3,5)
    plt.imshow(out_final.angle()[0].detach().cpu())
    plt.title('output: angle')
    plt.colorbar()

    plt.subplot(2,3,6)
    plt.plot(losses)
    plt.title(f'loss -> final_loss: {np.round(losses[-1], 3)}')
    
    plt.suptitle(f'{description} : output', y= 1.01)
    plt.tight_layout()

    plt.savefig(f'{save_dir}/{description}.pdf',bbox_inches='tight')
    plt.show()