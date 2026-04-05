import argparse
import os
import sys
import json
import logging
import warnings
from tqdm import tqdm

from sklearn import metrics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from dataset import IQON_FITB
from torch.utils.data import dataloader
import model

from torch.cuda.amp import autocast as autocast, GradScaler

torch.set_num_threads(4)

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='PS-OCM FITB result')
parser.add_argument('--datadir', default='../data/', type=str,
                    help='directory of the IQON3000 outfits dataset')
parser.add_argument('--imgpath', type=str, default='/home/share/wangchun/KGAT-pytorch-master/other/IQON3000/IQON3000')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='input batch size for training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--num_workers', type=int, default=0)

parser.add_argument('--save_dir', type=str, default='./')
args = parser.parse_args()


def compute_fitb_acc(predicted, label):
    label = np.zeros_like(label)
    predicted_max_index = np.argmax(predicted, axis=0)
    total_num = predicted.shape[1]
    correct_num = np.where(predicted_max_index == label, 1, 0)

    return np.sum(correct_num) / float(total_num)


def test():
    model = torch.load(os.path.join('../Comp/result/','model.pt')).cuda()
    model.eval()

    testset = IQON_FITB(args, split='test',
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.Resize(256),
                                torchvision.transforms.CenterCrop(224),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])
                            ]))

    testloader = dataloader.DataLoader(testset, 
                             batch_size = args.batch_size,
                             shuffle = False,
                             drop_last = True,
                             num_workers=args.num_workers,
                             collate_fn= lambda i :i)    

    all_predicted_score = [[],[],[],[]]
    all_labels = []

    with torch.no_grad():
        with tqdm(total = 4 * len(testloader)) as t:
            for i,data in enumerate(testloader):    
                data0 = [t[0] for t in data]
                data1 = [t[1] for t in data]
                data2 = [t[2] for t in data]
                data3 = [t[3] for t in data]

                data_current = [data0, data1, data2, data3]
                for _, d in enumerate(data_current):

                    img = [t['img'] for t in d]
                    att_label = [t['att_label'] for t in d]
                    att_mask = [t['att_mask'] for t in d]
                    target = [t['target'] for t in d]
                    partial_mask = [t['partial_mask'] for t in d]
                    
                    att_mask = torch.tensor(att_mask).unsqueeze(3).cuda() 
                    att_label = torch.tensor(att_label).cuda()
                    partial_mask = torch.tensor(partial_mask).unsqueeze(3).cuda()
                    predicted_score = model(img,att_mask,att_label,partial_mask)[0] 

                    labels = torch.tensor(target).squeeze(1).cuda()
                    predicted_y = F.softmax(predicted_score, dim=1)
                    predicted_y = predicted_y[:,1]
                    all_predicted_socre[_] += [predicted_y.data.cpu().numpy()]
                    if _ == 0:
                        all_labels += [labels.data.cpu().numpy()]

                    t.update()

    all_predicted_socre = [np.concatenate(i) for i in all_predicted_socre]
    all_predicted_socre = np.stack(all_predicted_socre)

    all_labels = np.concatenate(all_labels)
    acc = compute_fitb_acc(all_predicted_socre, all_labels)

    saved_acc = os.path.join(args.save_dir, "fitb_acc_result.txt")
    with open(saved_acc, 'w') as f:
        f.write('fitb_acc: %s ' % (str(acc)))

    print(acc)


if __name__ == '__main__':

    print('Arguments:')
    for k in args.__dict__.keys():
        print('    ', k, ':', str(args.__dict__[k]))
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  # Numpy module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True 

    test()
