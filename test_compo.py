#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test_compo.py
@Time    :   2024/11/23 15:00:04
@Author  :   yyywxk
@Email   :   qiulinwei@buaa.edu.cn
'''

import warnings
import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
import time

from dataloaders import make_compo_data_loader
from models import compo_model_select

import torch
from torch import nn
from thop import clever_format, profile

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # set the GPUs
os.environ['TORCH_HOME'] = './pretrained_models/'
warnings.filterwarnings('ignore')


def parse_args():
    '''
    To define parameters and settings
    '''
    parser = argparse.ArgumentParser(
        description='PyTorch Codes for Unsupervised Deep Image Stitching Testing')
    # --------------------------------- Base Settings ----------------------------
    parser.add_argument('--dataset', type=str, default='UDIS-D',
                        choices=['UDIS-D', 'UDAIS-D', 'UDAIS-D+', 'MS-COCO'],
                        help='dataset name (default: UDIS)')
    parser.add_argument('--test_path', type=str,
                        default='./Warp/UDAIS-D/UDIS2/align/testing/',
                        help='your data path for testing after stage 1')
    parser.add_argument('--model', type=str, default='UDIS2',
                        choices=['UDIS', 'UDIS2'],
                        help='model name (default: UDIS2)')
    parser.add_argument('--mode', type=str, default='test_output',
                        choices=['test', 'test_output'],
                        help='mode status (default: output)')
    parser.add_argument('--workers', type=int, default=1, metavar='N', help='dataloader threads')
    parser.add_argument('--model_path', type=str,
                        default='./run_com/UDIS-D/UDIS2/experiment_0/epoch50_model.pth', 
                        help='load your model')
    parser.add_argument('-b', '--batch_size', default=1, type=int,
                        help='batch size (default: 1)')
    parser.add_argument('--gpu-id', type=str, default='0',
                        help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
    parser.add_argument('--no-cuda', action='store_true',
                        default=False, help='disables CUDA training')
    parser.add_argument("--profile", dest='do_profiling', action='store_true', default=False,
                        help='Calculate amount of params and FLOPs. ')
    parser.add_argument('--color', action='store_true', default=False,
                        help='draw composition images with different colors')
    # model
    parser.add_argument('--nclasses', default=1, type=int,
                        help='class for model (default: 1)')
    # dataset
    # define the image resolution
    parser.add_argument('--height', type=int, default=512,
                        help='height of input images (UDIS: 640, UDIS2/UDAIS-D/UDAIS-D+: 512: 512)')
    parser.add_argument('--width', type=int, default=512,
                        help='width of input images (UDIS: 640, UDIS2/UDAIS-D/UDAIS-D+: 512: 512)')

    parser.add_argument('--save_path', type=str,
                        default='./Final/',
                        help='save your prediction data')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            torch.cuda.empty_cache()
        except ValueError:
            raise ValueError(
                'Argument --gpu_ids must be a comma-separated list of integers only')

    if args.model_path is None:
        raise ValueError('=> no checkpoint, please set the model path')
    
    os.makedirs(args.save_path, exist_ok=True)

    return args   
        

class Tester(object):
    def __init__(self, args):
        self.args = args
        # Define Dataloader
        self.test_loader = make_compo_data_loader(args, mode=args.mode)

        # Define network
        self.net = compo_model_select(args, args.model)
        if args.cuda:
            # self.net = torch.nn.DataParallel(self.net, device_ids=self.args.gpu_id)
            # self.net = torch.nn.DataParallel(self.net)
            self.net = self.net.cuda()
        checkpoint = torch.load(args.model_path)
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.model_path, checkpoint['epoch']))
        self.net.load_state_dict(checkpoint['model'])
        # Weights calculation
        pytorch_total_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print("Total_params: {}".format(pytorch_total_params))

        # Calculate amount of params and FLOPs.
        if args.do_profiling:
            pass

        self.net.eval()

    def test(self):
        print("=================== Test mode! ==================")
        print("No metric here!")

    def test_output(self):
        print("=================== Test output mode! ==================")
        # create folders if it dose not exist
        self.args.path_learn_mask1 = os.path.join(args.save_path, 'learn_mask1/')
        self.args.path_learn_mask2 = os.path.join(args.save_path, 'learn_mask2/')
        self.args.path_composition = os.path.join(args.save_path, 'composition/')
        if args.color:
            self.args.path_color = os.path.join(args.save_path, 'color/')
            os.makedirs(self.args.path_color, exist_ok=True)
        os.makedirs(self.args.path_learn_mask1, exist_ok=True)
        os.makedirs(self.args.path_learn_mask2, exist_ok=True)
        os.makedirs(self.args.path_composition, exist_ok=True)
        tbar = tqdm(self.test_loader, desc='Testing Images')
        for i, batch_value in enumerate(tbar):
            warp1_tensor = batch_value[0].float()
            warp2_tensor = batch_value[1].float()
            mask1_tensor = batch_value[2].float()
            mask2_tensor = batch_value[3].float()
            name = batch_value[4][0]

            if self.args.cuda:
                warp1_tensor = warp1_tensor.cuda()
                warp2_tensor = warp2_tensor.cuda()
                mask1_tensor = mask1_tensor.cuda()
                mask2_tensor = mask2_tensor.cuda()

            with torch.no_grad():
                if self.args.model == 'UDIS2':
                    batch_out = self.net.build_model(warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor)
                elif self.args.model == 'UDIS':
                    from models.UDIS.net import build_compo_model
                    # (-1,1)->(0,1)
                    # tanh is used in UDIS codes, and here is sigmoid
                    warp1_tensor = (warp1_tensor + 1.0)/2.0
                    warp2_tensor = (warp2_tensor + 1.0)/2.0
                    batch_out = build_compo_model(self.net, warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor)

            stitched_image = batch_out['stitched_image']
            learned_mask1 = batch_out['learned_mask1']
            learned_mask2 = batch_out['learned_mask2']

            # (optional) draw composition images with different colors like our paper
            if self.args.color:
                if self.args.cuda:
                    s1 = ((warp1_tensor[0]+1)*127.5 * learned_mask1[0]).cpu().detach().numpy().transpose(1,2,0)
                    s2 = ((warp2_tensor[0]+1)*127.5 * learned_mask2[0]).cpu().detach().numpy().transpose(1,2,0)
                else:
                    s1 = ((warp1_tensor[0]+1)*127.5 * learned_mask1[0]).detach().numpy().transpose(1,2,0)
                    s2 = ((warp2_tensor[0]+1)*127.5 * learned_mask2[0]).detach().numpy().transpose(1,2,0)
                fusion = np.zeros((warp1_tensor.shape[2],warp1_tensor.shape[3],3), np.uint8)
                fusion[...,0] = s2[...,0]
                fusion[...,1] = s1[...,1]*0.5 +  s2[...,1]*0.5
                fusion[...,2] = s1[...,2]
                path = self.args.path_color + name
                cv2.imwrite(path, fusion)

            if self.args.cuda:
                stitched_image = ((stitched_image[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
                learned_mask1 = (learned_mask1[0]*255).cpu().detach().numpy().transpose(1,2,0)
                learned_mask2 = (learned_mask2[0]*255).cpu().detach().numpy().transpose(1,2,0)
            else:
                stitched_image = ((stitched_image[0]+1)*127.5).detach().numpy().transpose(1,2,0)
                learned_mask1 = (learned_mask1[0]*255).detach().numpy().transpose(1,2,0)
                learned_mask2 = (learned_mask2[0]*255).detach().numpy().transpose(1,2,0)

            path = self.args.path_learn_mask1 + name
            cv2.imwrite(path, learned_mask1)
            path = self.args.path_learn_mask2 + name
            cv2.imwrite(path, learned_mask2)
            path = self.args.path_composition + name
            cv2.imwrite(path, stitched_image)


def main(args):
    print('Start testing!')
    print(args)

    tester = Tester(args)
    if args.mode == 'test':
        tester.test()
    elif args.mode == 'test_output':
        tester.test_output()
    else:
        raise NotImplementedError


if __name__ == '__main__':
    since = time.time()
    args = parse_args()
    main(args)
    time_elapsed = time.time() - since
    print('Finish testing!Totally cost: {:.0f}m {:.5f}s'.format(time_elapsed // 60, time_elapsed % 60))
