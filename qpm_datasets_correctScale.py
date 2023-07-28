import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import glob
import numpy as np
from torchvision import transforms
import random
import torch


class qpm_np_v4_CorrectedScale(Dataset):
    def __init__(self, data_dir, type_= 'train', crop_size= 32, remove_inverted= False):
        self.np_image_paths = sorted(glob.glob(f'{data_dir}/{type_}/*.npy'))
        
        if remove_inverted:
            inverted= [13, 72, 83, 127, 243, 273, 277, 312, 461, 14, 15, 242]
            not_inverted = [int(x.split('/')[-1][:-4]) not in inverted for x in self.np_image_paths]
            self.np_image_paths = np.array(self.np_image_paths)[not_inverted]
            
            print_text = '.npy, '.join(list(map(str, inverted)))+'.npy'
            print(f'Removed inverted samples : {print_text}')
        
        random.Random(4).shuffle(self.np_image_paths)
        
        self.crop_size= crop_size
        
    def transform(self, amp_img, phase_img):
        # Transform to tensor
        amp_img = TF.to_tensor(amp_img)
        phase_img = TF.to_tensor(phase_img)
        
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            amp_img, output_size=(self.crop_size, self.crop_size))
        amp_img = TF.crop(amp_img, i, j, h, w)
        phase_img = TF.crop(phase_img, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            amp_img = TF.hflip(amp_img)
            phase_img = TF.hflip(phase_img)

        # Random vertical flipping
        if random.random() > 0.5:
            amp_img = TF.vflip(amp_img)
            phase_img = TF.vflip(phase_img)
            
        return amp_img, phase_img

    def __getitem__(self, index):
        
        amp_img, phase_img = np.load(self.np_image_paths[index])
        amp_img, phase_img = self.transform(amp_img, phase_img)
        
        phase_img= torch.clip(phase_img, 0, 2*np.pi- 0.00001)
        out_img= amp_img.float() * torch.exp(1j* phase_img.float())
        return out_img

    def __len__(self):
        return len(self.np_image_paths)