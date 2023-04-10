from __future__ import absolute_import
import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler

import ResNet
import losses
import data_manager
import transforms as T
from dataset_loader import ImageDataset
from utils import Logger
from utils import AverageMeter, Logger, save_checkpoint
from eval_metrics import evaluate
from optimizers import init_optim

# from IPython import embed
import torchvision.transforms.functional as f

parser = argparse.ArgumentParser(description='Train AlignedReID with cross entropy loss and triplet hard loss')
# Datasets
parser.add_argument('--root', type=str, default='data', help="root path to data directory")
# parser.add_argument('-d', '--dataset', type=str, default='market1501',
#                      choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 128)")
parser.add_argument('--split-id', type=int, default=0, help="split index")

parser.add_argument('--use-metric-cuhk03', action='store_true',
                    help="whether to use cuhk03-metric (default: False)")

parser.add_argument('--labelsmooth', action='store_true', help="label smooth")
parser.add_argument('--optim', type=str, default='adam', help="optimization algorithm (see optimizers.py)")
parser.add_argument('--max-epoch', default=300, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=32, type=int, help="test batch size")
parser.add_argument('--lr', '--learning-rate', default=0.0002, type=float,
                    help="initial learning rate")
parser.add_argument('--stepsize', default=150, type=int,
                    help="stepsize to decay learning rate (>0 means this is enabled)")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
# triplet hard loss
parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")
parser.add_argument('--num-instances', type=int, default=4,
                    help="number of instances per identity")
parser.add_argument('--htri-only', action='store_true', default=False,
                    help="if this is True, only htri loss is used in training")

# parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=ResNet.get_names())

parser.add_argument('--print-freq', type=int, default=10, help="print frequency")
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--eval-step', type=int, default=20,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start-eval', type=int, default=0, help="start to evaluate after specific epoch")
parser.add_argument('--save-dir', type=str, default='model_save')
parser.add_argument('--use_cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--reranking',action= 'store_true', help= 'result re_ranking')

parser.add_argument('--test_distance',type = str, default='global', help= 'test distance type')
parser.add_argument('--unaligned',action= 'store_true', help= 'test local feature with unalignment')

args = parser.parse_args()



def main():
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: 
        use_gpu = False
    if use_gpu:
        pin_memory = True
    else: 
        pin_memory = False
        
    if not args.evaluate:  #如果不是测试模式，存为log_train.txt
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:                 #否则存为log_test.txt
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")
        
    dataset = data_manager.Market1501(root = 'data')
        
    transform_train = T.Compose([
        T.Random2DTranslation(args.height, args.width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225]),    #归一化
    ])
    transform_test = T.Compose([
        T.Resize([args.height, args.width]),
        # T.Resize(args.height, args.width,interpolation=f._interpolation_modes_from_int(0)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225]),    #归一化
    ])
        
    trainloader = DataLoader(
        ImageDataset(dataset.train, transform = transform_train),
        shuffle=True,
        batch_size = args.train_batch, 
        num_workers=args.workers,
        pin_memory = pin_memory, 
        drop_last=True,
    )
        
    queryloader = DataLoader(
        ImageDataset(dataset.query, transform = transform_test),
        shuffle=False,
        batch_size = args.test_batch, num_workers=args.workers,
        pin_memory = pin_memory, 
        drop_last=False,
    )
        
    galleryloader = DataLoader(
        ImageDataset(dataset.gallery, transform = transform_test),
        shuffle=False,
        batch_size = args.test_batch, num_workers=args.workers,
        pin_memory = pin_memory, 
        drop_last=False,
    )
        
    
    print("Initializing model: {}".format('market_1501'))
    model = ResNet.ResNet50(num_classes=751)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))
        
    criterion_class = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr,  weight_decay = args.weight_decay)     #如果只修改其中某几层参数就这么写nn.Sequential([ResNet.conv1, ResNet.conv2])
    
    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma = args.gamma)
    
    start_epoch = args.start_epoch
    
    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        
    if use_gpu:
        # model = nn.DataParallel(model).cuda()   #多GPU训练
        model = model.cuda()
        
        
    if args.evaluate:
        print('Evaluate only!')
        model.test(model, queryloader, galleryloader, use_gpu)
        return 0
    
    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    
    print('start training')
        
    for epoch in range(start_epoch, args.max_epoch):
        start_train_time = time.time()
        train(epoch, model, criterion_class, optimizer, trainloader, use_gpu)
        
        # model._save_to_state_dict()    #保存模型
        
        train_time += round(time.time() - start_train_time)
        
        if args.stepsize > 0:scheduler.step()     #学习率衰减
        
        if (epoch + 1) > args.start_eval and args.eval_step > 0 and (epoch + 1) % args.eval_step == 0 or (epoch + 1) == args.max_epoch:
            
            print("===> Test")
            rank1 = test(model, queryloader, galleryloader, use_gpu)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch+1
            
            
            if use_gpu:
                state_dict = model.state_dict()
            else:
                state_dict = model.state_dict()
            save_checkpoint({
                'state_dict': state_dict,
                'rank1': rank1,
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))
    print("==>Best Rank-1:{:.1%},chieved at epoch{}".format(best_rank1, best_epoch))
    
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds = elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished Total elapsed time (H:M:S):{}. Training time (H:M:S):{}.".format(elapsed,train_time))
    
def train(epoch, model, criterion_class, optimizer, trainloader, use_gpu):
    model.train()
    
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    end = time.time()
    
    for batch_idx, (imgs, pids, _) in enumerate(trainloader):
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()
        data_time.update(time.time()-end)
        
        outputs = model(imgs)
        loss = criterion_class(outputs, pids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_time.update(time.time() - end)
        end = time.time()
        losses.update(loss.item(), pids.size(0))
        
        if (batch_idx+1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch+1, batch_idx+1, len(trainloader), batch_time=batch_time,data_time=data_time,
                   loss=losses))
        
        
        

def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20]):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids, lqf = [], [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features, local_features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            local_features = local_features.data.cpu()
            qf.append(features)
            lqf.append(local_features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        lqf = torch.cat(lqf,0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids, lgf = [], [], [], []
        end = time.time()
        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features, local_features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cuda()
            local_features = local_features.data.cuda()
            gf.append(features)
            lgf.append(local_features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        lgf = torch.cat(lgf,0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)


        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch))
    # feature normlization
    qf = (1. * qf / (torch.norm(qf, 2, dim = -1, keepdim=True).expand_as(qf) + 1e-12)).cuda()
    gf = (1. * gf / (torch.norm(gf, 2, dim = -1, keepdim=True).expand_as(gf) + 1e-12)).cuda()
    m, n = torch.tensor(qf.size(0)).cuda(), torch.tensor(gf.size(0)).cuda()
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    distmat = distmat.cpu().numpy()

    if not args.test_distance== 'global':
        print("Only using global branch")
        from distance import low_memory_local_dist
        lqf = lqf.permute(0,2,1)
        lgf = lgf.permute(0,2,1)
        local_distmat = low_memory_local_dist(lqf.numpy(),lgf.numpy(),aligned= not args.unaligned)
        if args.test_distance== 'local':
            print("Only using local branch")
            distmat = local_distmat
        if args.test_distance == 'global_local':
            print("Using global and local branches")
            distmat = local_distmat+distmat
            
    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")
    
    
    return cmc[0]





   

if __name__ == '__main__':
    main()
            