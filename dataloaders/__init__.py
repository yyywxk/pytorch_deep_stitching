#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py
@Time    :   2024/11/21 15:57:45
@Author  :   yyywxk
@Email   :   qiulinwei@buaa.edu.cn
'''

import torch
from torch.utils.data import DataLoader
import os


def my_path(path):
    '''
    To set the dataset path
    '''
    if path == 'UDIS-D':
        return '../../dataset/UDIS-D/'
    elif path == 'MS-COCO':
        return '../../dataset/Warped MS-COCO/'
    elif path == 'UDAIS-D':
        return '../../dataset/UDAIS-D/'
    elif path == 'UDAIS-D+':
        return '../../dataset/UDAIS-D+/'
    else:
        print('Dataset {} not available.'.format(path))
        raise NotImplementedError


def make_align_data_loader(args, mode='train', **kwargs):
    if args.dataset == 'UDIS-D' or args.dataset == 'MS-COCO' or args.dataset == 'UDAIS-D' or args.dataset == 'UDAIS-D+':
        if args.dataset == 'UDIS-D' or args.dataset == 'MS-COCO':
            from .datasets.udis_d import TrainAlignDataset, TestAlignDataset
        elif args.dataset == 'UDAIS-D' or args.dataset == 'UDAIS-D+':
            from .datasets.udais_d import TrainAlignDataset, TestAlignDataset
        if mode == 'train':
            # set dataset path
            print('Init data, please wait!')
            args.train_path = my_path(args.dataset) + 'training/'
            # args.train_path = my_path(args.dataset) + 'testing/'
            train_data = TrainAlignDataset(
                data_path=args.train_path, width=args.width, height=args.height)
            train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size,
                                      num_workers=args.workers, shuffle=True, drop_last=False)
            print('Init data successfully!')
            return train_loader
        elif mode == 'test':
            # set dataset path
            if args.test_path == None:
                # args.test_path = my_path(args.dataset) + 'training/'
                args.test_path = my_path(args.dataset) + 'testing/'
            test_data = TestAlignDataset(
                data_path=args.test_path, width=args.width, height=args.height, resize_flag=True)
            test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size,
                                     num_workers=args.workers, shuffle=False, drop_last=False)
            print('Init data successfully!')
            return test_loader
        elif mode == 'test_output':
            # set dataset path
            if args.test_path == None:
                args.test_path = os.path.join(my_path(args.dataset), args.get_warp_path)
            test_data = TestAlignDataset(
                data_path=args.test_path, width=args.width, height=args.height)
            test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size,
                                     num_workers=args.workers, shuffle=False, drop_last=False)
            print('Init data successfully!')
            return test_loader
        elif mode == 'test_other':
            # set dataset path
            if args.test_path == None:
                raise ValueError(
                    '=> no input image, please set the input path')
            test_data = TestAlignDataset(
                data_path=args.test_path, width=args.width, height=args.height)
            test_loader = DataLoader(dataset=test_data, batch_size=1,
                                     num_workers=args.workers, shuffle=False, drop_last=False)
            print('Init data successfully!')
            return test_loader
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

def make_compo_data_loader(args, mode='train', **kwargs):
    if args.dataset == 'UDIS-D' or args.dataset == 'MS-COCO' or args.dataset == 'UDAIS-D' or args.dataset == 'UDAIS-D+':
        if args.dataset == 'UDIS-D' or args.dataset == 'MS-COCO':
            from .datasets.udis_d import TrainCompoDataset, TestCompoDataset
        elif args.dataset == 'UDAIS-D' or args.dataset == 'UDAIS-D+':
            from .datasets.udais_d import TrainCompoDataset, TestCompoDataset
        if mode == 'train':
            print('Init data, please wait!')
            train_data = TrainCompoDataset(
                data_path=args.train_path, width=args.width, height=args.height, resize_flag=args.resize)
            train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size,
                                      num_workers=args.workers, shuffle=True, drop_last=False)
            print('Init data successfully!')
            return train_loader
        elif mode == 'test_output':
            test_data = TestCompoDataset(data_path=args.test_path)
            test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size,
                                     num_workers=args.workers, shuffle=False, drop_last=False)
            print('Init data successfully!')
            return test_loader
        elif mode == 'test':
            test_data = TestCompoDataset(data_path=args.test_path)
            test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size,
                                     num_workers=args.workers, shuffle=False, drop_last=False)
            print('Init data successfully!')
            return test_loader
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
