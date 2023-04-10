from __future__ import absolute_import

from torchvision.transforms import *
from PIL import Image
import random
import torchvision.transforms as transforms

class Random2DTranslation(object):  #数据增广：先增加1/8，再随机裁剪
    def __init__(self, height, width, p = 0.5, interpolation = Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation
        
    def __call__(self, img):
        if random.random() < self.p:
            return img.resize((self.width,self.height),self.interpolation)
        new_width, new_height = int(round(self.width*1.125)), int(round(self.height*1.125))
        resize_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        
        cropped_img = resize_img.crop((x1, y1, x1+self.width, y1+self.height))
        
        return cropped_img
    
    
    
# #测试
# if __name__ == '__main__':
#     img = Image.open('data/Market-1501-v15.09.15/bounding_box_test/-1_c1s1_000401_03.jpg')
#     transforms = Random2DTranslation(256,128,0.5)
#     img_t = transforms(img)
#     import matplotlib.pyplot as plt
    
    
# plt.figure(12)
# plt.subplot(121)
# plt.imshow(img)
# plt.subplot(122)
# plt.imshow(img_t)
# plt.show()
    
    
    
    