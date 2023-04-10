from __future__ import print_function, absolute_import
import os
from PIL import Image
import numpy
import os.path as osp
import torch
from torch.utils.data import Dataset

def read_image(img_path):
    got_img = False
    
    if not osp.exists(img_path):
        raise IOError("{} dose not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except  IOError:
            print('dose not read image')
            pass
    return img

class ImageDataset(Dataset):
    def __init__(self,dataset, transform = None):
        self.dataset = dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
    

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid ,camid
        
            





if __name__ == '__main__':
    import data_manager
    dataset = data_manager.Market1501(root = 'data')
    train_loader = ImageDataset(dataset.train)
    # from IPython import embed
         
            
