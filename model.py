import torch
from torch import nn
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms
from torch.optim.optimizer import Optimizer, required

import os
import math
import cv2
from skimage import measure
from PIL import Image
import numpy as np
from tqdm import tqdm_notebook as tqdm
from sklearn import metrics

import pretrainedmodels

def scale_radius(src, img_size, padding=False):
    x = src[src.shape[0] // 2, ...].sum(axis=1)
    r = (x > x.mean() / 10).sum() // 2
    yx = src.sum(axis=2)
    region_props = measure.regionprops((yx > yx.mean() / 10).astype('uint8'))
    yc, xc = np.round(region_props[0].centroid).astype('int')
    x1 = max(xc - r, 0)
    x2 = min(xc + r, src.shape[1] - 1)
    y1 = max(yc - r, 0)
    y2 = min(yc + r, src.shape[0] - 1)
    dst = src[y1:y2, x1:x2]
    dst = cv2.resize(dst, dsize=None, fx=img_size/(2*r), fy=img_size/(2*r))
    if padding:
        pad_x = (img_size - dst.shape[1]) // 2
        pad_y = (img_size - dst.shape[0]) // 2
        dst = np.pad(dst, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), 'constant')
    return dst

class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, labels, transform=None, img_size=576, save_img=True):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
        self.img_size = img_size
        self.save_img = save_img

    def __getitem__(self, index):
        img_path, label = self.img_paths[index], self.labels[index]
        
        if 'train' in img_path:
            img = cv2.imread('../input/aptos2019-dataset/images_288_scaled/images_288_scaled/%s' %os.path.basename(img_path))
        
        else:
            if os.path.exists('processed/%s' %os.path.basename(img_path)):
                img = cv2.imread('processed/%s' %os.path.basename(img_path))

            else:
                img = cv2.imread(img_path)
                try:
                    img = scale_radius(img, img_size=self.img_size, padding=False)
                except Exception as e:
                    img = img
                if self.save_img:
                    cv2.imwrite('processed/%s' %os.path.basename(img_path), img)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, label


    def __len__(self):
        return len(self.img_paths)


def get_model(model_name='resnet18', num_outputs=None, pretrained=True,
              freeze_bn=False, dropout_p=0, **kwargs):

    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.__dict__[model_name](num_classes=1000,
                                                  pretrained=pretrained)

    if 'dpn' in model_name:
        in_channels = model.last_linear.in_channels
        model.last_linear = nn.Conv2d(in_channels, num_outputs,
                                      kernel_size=1, bias=True)
    else:
        if 'resnet' in model_name:
            model.avgpool = nn.AdaptiveAvgPool2d(1)
        else:
            model.avg_pool = nn.AdaptiveAvgPool2d(1)
        in_features = model.last_linear.in_features
        if dropout_p == 0:
            model.last_linear = nn.Linear(in_features, num_outputs)
        else:
            model.last_linear = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(in_features, num_outputs),
            )

    if freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    return model


class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def quadratic_weighted_kappa(y_pred, y_true):
    if torch.is_tensor(y_pred):
        y_pred = y_pred.data.cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.data.cpu().numpy()
    if y_pred.shape[1] == 1:
        y_pred = y_pred[:, 0]
    else:
        y_pred = np.argmax(y_pred, axis=1)
    return metrics.cohen_kappa_score(y_pred, y_true, weights='quadratic')


def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    scores = AverageMeter()

    model.train()

    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input = input.cuda()
        target = target.cuda()

        output = model(input)
        loss = criterion(output.view(-1), target.float())

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        thrs = [0.5, 1.5, 2.5, 3.5]
        output[output < thrs[0]] = 0
        output[(output >= thrs[0]) & (output < thrs[1])] = 1
        output[(output >= thrs[1]) & (output < thrs[2])] = 2
        output[(output >= thrs[2]) & (output < thrs[3])] = 3
        output[output >= thrs[3]] = 4
        
        target[target < thrs[0]] = 0
        target[(target >= thrs[0]) & (target < thrs[1])] = 1
        target[(target >= thrs[1]) & (target < thrs[2])] = 2
        target[(target >= thrs[2]) & (target < thrs[3])] = 3
        target[target >= thrs[3]] = 4
        
        score = quadratic_weighted_kappa(output, target)

        losses.update(loss.item(), input.size(0))
        scores.update(score, input.size(0))

    return losses.avg, scores.avg


def validate(val_loader, model, criterion):
    losses = AverageMeter()
    scores = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            loss = criterion(output.view(-1), target.float())
        
            thrs = [0.5, 1.5, 2.5, 3.5]
            output[output < thrs[0]] = 0
            output[(output >= thrs[0]) & (output < thrs[1])] = 1
            output[(output >= thrs[1]) & (output < thrs[2])] = 2
            output[(output >= thrs[2]) & (output < thrs[3])] = 3
            output[output >= thrs[3]] = 4
            score = quadratic_weighted_kappa(output, target)

            losses.update(loss.item(), input.size(0))
            scores.update(score, input.size(0))

    return losses.avg, scores.avg