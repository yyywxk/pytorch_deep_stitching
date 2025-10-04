#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   build_optimizer.py
@Time    :   2024/11/22 17:09:51
@Author  :   yyywxk
@Email   :   qiulinwei@buaa.edu.cn
'''

import torch.optim as optim


def build_optimizer(args, model):
    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay,
                              nesterov=args.nesterov)
    elif args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(
        ), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999), eps=1e-08)
    else:
        raise AssertionError
    return optimizer
