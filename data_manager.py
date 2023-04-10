import os
import os.path as osp
import numpy as np
import glob
# from IPython import embed
import re

class Market1501(object):
    
    dataset_dir = 'Market-1501-v15.09.15'
    
    def __init__(self, root = 'data',**kwargs):
        self.dataset_dir = osp.join(root,self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir,'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir,'query')
        self.gallery_dir = osp.join(self.dataset_dir,'bounding_box_test')
        
        self._check_before_run()
    
        # 文件路径，标注信息（ID,CAMID）,图片数量
        train,num_train_pids,num_train_imgs = self._process_dir(self.train_dir, relabel=True)
        query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir)
        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs
        
        
        print("=> Market1501 laoded")
        print("Dataset statistics:")
        print(" -------------------------------")
        print(" subset    |  #ids  |  # images")
        print(" -------------------------------")
        print(" train     |  {:5d} |  {:8d}".format(num_train_pids,num_train_imgs))
        print(" query     |  {:5d} |  {:8d}".format(num_query_pids,num_query_imgs))
        print(" gallery   |  {:5d} |  {:8d}".format(num_gallery_pids,num_gallery_imgs))
        print(" ------------------------------")
        print(" total     |  {:5d} |  {:8d}".format(num_total_pids,num_total_imgs))
        print(" ------------------------------")
        
        self.train = train
        self.query = query
        self.gallery = gallery
        
        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids
        
        
    def _check_before_run(self):
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}'is not available!".format(self.dataset_dir))  
        if not os.path.exists(self.train_dir):
            raise RuntimeError("'{}'is not available!".format(self.train_dir))
        if not os.path.exists(self.query_dir):
            raise RuntimeError("'{}'is not available!".format(self.query_dir))
        if not os.path.exists(self.gallery_dir):
            raise RuntimeError("'{}'is not available!".format(self.gallery_dir))
       
    def _process_dir(self,dir_path,relabel = False):
        img_paths = glob.glob(osp.join(dir_path,"*.jpg"))   #拿出所有.jpg文件
        pattern = re.compile(r'([-\d]+)_c(\d)')
        pid_container = set()      #set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())   #这里的map将字符串转换成int型，然后只取第一个也就是id
            if pid == -1:
                continue
            assert 0<=pid<=1501
            assert 1<=camid<=6
            camid -= -1
            pid_container.add(pid)
            pid2label = {pid:label for label, pid in enumerate(pid_container)}  #enumerate同时列出数据和数据下标
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))   #把图片路径，人id，摄像机id加入dataset数组
            
        num_pids = len(pid_container)
        num_imgs = len(img_paths)
            
        
        
        return dataset,num_pids,num_imgs
        
        
            
        
if __name__ == '__main__':
    data = Market1501(root = 'data')