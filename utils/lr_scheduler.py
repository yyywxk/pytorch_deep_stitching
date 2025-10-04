#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   lr_scheduler.py
@Time    :   2024/11/22 17:14:46
@Author  :   yyywxk
@Email   :   qiulinwei@buaa.edu.cn
'''


import torch.optim as optim


def LR_Scheduler(mode, optimizer, T_max=20, eta_min=0, lr_step=30, decay_rate=0.97, last_epoch=-1):
    print('Using {} LR Scheduler!'.format(mode))
    if mode == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, lr_step, decay_rate, last_epoch)
    elif mode == 'poly':
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, decay_rate, last_epoch)
    elif mode == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max, eta_min, last_epoch)
    else:
        raise ValueError(
            'Learning scheduler {} is not supported!'.format(mode))
    return scheduler
