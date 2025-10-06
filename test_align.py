#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test_align.py
@Time    :   2024/11/22 21:50:37
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
import datetime
import copy

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse

from dataloaders import make_align_data_loader
from models import align_model_select

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
                        help='dataset name (default: UDIS-D)')
    parser.add_argument('--model', type=str, default='UDIS2',
                        choices=['UDIS', 'UDIS2'],
                        help='model name (default: UDIS2)')
    parser.add_argument('--mode', type=str, default='test_output',
                        choices=['test', 'test_output', 'test_other'],
                        help='mode status (default: output)')
    parser.add_argument('--workers', type=int, default=1,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--model_path', type=str,
                        default='./run_align/UDIS-D/UDIS2/experiment_0/epoch100_model.pth',
                        help='load your model')
    parser.add_argument('-b', '--batch_size', default=1, type=int,
                        help='batch size (default: 1)')
    parser.add_argument('--gpu-id', type=str, default='0',
                        help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
    parser.add_argument('--no-cuda', action='store_true',
                        default=False, help='disables CUDA training')
    parser.add_argument("--profile", dest='do_profiling', action='store_true', default=False,
                        help='Calculate amount of params and FLOPs. ')
    # dataset
    # define the image resolution
    parser.add_argument('--height', type=int, default=512,
                        help='height of input images (UDIS: 128, UDIS2/UDAIS-D/UDAIS-D+: 512)')
    parser.add_argument('--width', type=int, default=512,
                        help='width of input images (UDIS: 128, UDIS2/UDAIS-D/UDAIS-D+: 512)')
    # define the mesh resolution
    parser.add_argument('--grid_w', default=12, type=int,
                        help='control points number of width (default: 12)')
    parser.add_argument('--grid_h', default=12, type=int,
                        help='control points number of height (default: 12)')
    parser.add_argument('--save_path', type=str,
                        default='./Warp/',
                        help='save your prediction data')
    # inference
    parser.add_argument('--input_path', type=str,
                        default=None,
                        # default='input/',
                        help='put the path to predict: "input1" and "input2" folders are needed')
    # test_other mode settings
    parser.add_argument('--optim', default='adam',
                        choices=['adam', 'sgd'], help='optimizer')
    parser.add_argument('--lr', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate (default: 1e-4)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='SGD momentum')
    parser.add_argument('--nesterov', action='store_true',
                        default=False, help='To use nesterov or not.')
    parser.add_argument('--weight-decay', '--wd', default=0.0, type=float,
                        metavar='WD', help='weight decay (default: 0.0)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'], help='lr scheduler mode: (default: poly)')
    parser.add_argument('--decay-rate', '--dr', default=0.97, type=float,
                        metavar='DR', help='decay rate (default: 0.96 for step, 0.97 for poly)')
    parser.add_argument('--max_iter', type=int, default=50)
    # test_output mode settings
    parser.add_argument('--get_warp_path', type=str,
                        default='training',
                        choices=['training', 'testing'],
                        help='put the data path to get_warp: training or testing')
    parser.add_argument('--save_wrong', action='store_true',
                        default=False, help='using original inputs as wrong results')
    # test_other setting
    parser.add_argument('--threshold', default=1e-4,
                        type=float, help='threshold for ft (default: 1e-4)')
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')

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

    if args.mode == 'test_output' or args.mode == 'test_other':
        os.makedirs(args.save_path, exist_ok=True)

    return args


class Tester(object):
    def __init__(self, args):
        if args.input_path:
            args.test_path = args.input_path
            print('Test on input images!\n')
        else:
            args.test_path = None
        self.args = args
        # Define Dataloader
        self.test_loader = make_align_data_loader(args, mode=args.mode)

        # Define network
        self.net = align_model_select(args, args.model)
        if args.cuda:
            # self.net = torch.nn.DataParallel(self.net, device_ids=self.args.gpu_id)
            # self.net = torch.nn.DataParallel(self.net)
            self.net = self.net.cuda()
        checkpoint = torch.load(args.model_path)
        print("=> loaded checkpoint '{}' (epoch {}, iter {})"
              .format(args.model_path, checkpoint['epoch'], checkpoint['glob_iter']))
        self.net.load_state_dict(checkpoint['model'])

        if args.mode == 'test_other' and args.ft:
            from utils.build_optimizer import build_optimizer
            from utils.lr_scheduler import LR_Scheduler  
            # Define Optimizer
            self.optimizer = build_optimizer(args, self.net)

            # Define lr scheduler
            self.scheduler = LR_Scheduler(args.lr_scheduler,
                                          self.optimizer, decay_rate=args.decay_rate)

            if args.model == 'UDIS2':
                from loss.align_loss import cal_lp_loss2
                self.criterion = cal_lp_loss2
            else:
                raise RuntimeError("=> Loss {} not supported".format(self.args.model))

            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.args.start_epoch = checkpoint['epoch']
            self.scheduler.last_epoch = self.args.start_epoch

        # Weights calculation
        pytorch_total_params = sum(p.numel()
                                   for p in self.net.parameters() if p.requires_grad)
        print("Total_params: {}".format(pytorch_total_params))
        # Calculate amount of params and FLOPs.
        if args.do_profiling:
            pass

        self.crashed = 0
        self.crashed_name = []
        self.net.eval()
        
    def test_output(self):
        print("=================== Test output mode! ==================")
        # create folders if it dose not exist
        self.args.path_ave_fusion = os.path.join(args.save_path, 'ave_fusion/')
        self.args.path_warp1 = os.path.join(args.save_path, 'warp1/')
        self.args.path_warp2 = os.path.join(args.save_path, 'warp2/')
        self.args.path_mask1 = os.path.join(args.save_path, 'mask1/')
        self.args.path_mask2 = os.path.join(args.save_path, 'mask2/')
        os.makedirs(self.args.path_ave_fusion, exist_ok=True)
        os.makedirs(self.args.path_warp1, exist_ok=True)
        os.makedirs(self.args.path_warp2, exist_ok=True)
        os.makedirs(self.args.path_mask1, exist_ok=True)
        os.makedirs(self.args.path_mask2, exist_ok=True)

        tbar = tqdm(self.test_loader, desc='Testing Images')
        if args.model == 'UDIS2':
            from models.UDIS2.net_warp import build_output_model
        elif args.model == 'UDIS':
            from models.UDIS.net import build_output_model

        for i, batch_value in enumerate(tbar):
            input1_tensor = batch_value[0].float()
            input2_tensor = batch_value[1].float()
            name = batch_value[2][0]
            if self.args.cuda:
                input1_tensor = input1_tensor.cuda()
                input2_tensor = input2_tensor.cuda()
            with torch.no_grad():
                batch_out = build_output_model(
                    self.net, input1_tensor, input2_tensor)

            final_warp1 = batch_out['final_warp1']
            final_warp1_mask = batch_out['final_warp1_mask']
            final_warp2 = batch_out['final_warp2']
            final_warp2_mask = batch_out['final_warp2_mask']
            # final_mesh1 = batch_out['mesh1']
            # final_mesh2 = batch_out['mesh2']

            # '3894_o11.png' in UDIS
            if batch_out['mesh1'].sum() == 0:
                self.crashed += 1
                self.crashed_name.append(name)
                print('ERROR: Warp Cashed on Image {}!'.format(name))
                if args.save_wrong:
                    final_warp1 = input1_tensor
                    final_warp2 = input2_tensor
                    final_warp1_mask = torch.ones_like(input1_tensor)
                    final_warp2_mask = torch.ones_like(input2_tensor)
                else:   
                    torch.cuda.empty_cache()
                    continue
            
            if self.args.cuda:
                final_warp1 = (
                    (final_warp1[0]+1)*127.5).cpu().detach().numpy().transpose(1, 2, 0)
                final_warp2 = (
                    (final_warp2[0]+1)*127.5).cpu().detach().numpy().transpose(1, 2, 0)
                final_warp1_mask = final_warp1_mask[0].cpu(
                ).detach().numpy().transpose(1, 2, 0)
                final_warp2_mask = final_warp2_mask[0].cpu(
                ).detach().numpy().transpose(1, 2, 0)
                # final_mesh1 = final_mesh1[0].cpu().detach().numpy()
                # final_mesh2 = final_mesh2[0].cpu().detach().numpy()
            else:
                final_warp1 = (
                    (final_warp1[0]+1)*127.5).detach().numpy().transpose(1, 2, 0)
                final_warp2 = (
                    (final_warp2[0]+1)*127.5).detach().numpy().transpose(1, 2, 0)
                final_warp1_mask = final_warp1_mask[0].detach(
                ).numpy().transpose(1, 2, 0)
                final_warp2_mask = final_warp2_mask[0].detach(
                ).numpy().transpose(1, 2, 0)
                # final_mesh1 = final_mesh1[0].detach().numpy()
                # final_mesh2 = final_mesh2[0].detach().numpy()

            path = self.args.path_warp1 + name
            cv2.imwrite(path, final_warp1)
            path = self.args.path_warp2 + name
            cv2.imwrite(path, final_warp2)
            path = self.args.path_mask1 + name
            cv2.imwrite(path, final_warp1_mask*255)
            path = self.args.path_mask2 + name
            cv2.imwrite(path, final_warp2_mask*255)

            ave_fusion = final_warp1 * (final_warp1 / (final_warp1+final_warp2+1e-6)) + \
                final_warp2 * (final_warp2 / (final_warp1+final_warp2+1e-6))
            path = self.args.path_ave_fusion + name
            cv2.imwrite(path, ave_fusion)

            torch.cuda.empty_cache()
        if self.crashed != 0:
            print('Warp Cashed on {} images!\n'.format(self.crashed))
            print(self.crashed_name)

    def test(self):
        print("=================== Test mode! ==================")
        psnr_list = []
        ssim_list = []
        mse_list = []

        tbar = tqdm(self.test_loader, desc='Testing Images')
        num = len(self.test_loader)
        if args.model == 'UDIS2':
            from models.UDIS2.net_warp import build_model
        elif args.model == 'UDIS':
            from models.UDIS.net import build_model

        for i, batch_value in enumerate(tbar):
            input1_tensor = batch_value[0].float()
            input2_tensor = batch_value[1].float()
            name = batch_value[2][0]
            if self.args.cuda:
                input1_tensor = input1_tensor.cuda()
                input2_tensor = input2_tensor.cuda()
            with torch.no_grad():
                batch_out = build_model(
                    self.net, input1_tensor, input2_tensor, is_training=False)

            warp_mesh_mask = batch_out['warp_mesh_mask']
            warp_mesh = batch_out['warp_mesh']
            # test homo
            # warp_mesh_mask = batch_out['output_H'][:, 3:6, ...]
            # warp_mesh = batch_out['output_H'][:, 0:3, ...]

            if warp_mesh_mask.sum() == 0:
                '''
                if crashed, use I3x3 to warp, only for UDIS
                '''
                self.crashed += 1
                self.crashed_name.append(name)
                print('ERROR: Warp Cashed on Image {}!'.format(name))
                # psnr_list.append(0)
                # ssim_list.append(0)
                # continue
                warp_mesh = torch.clone(input2_tensor)
                warp_mesh_mask = torch.ones_like(input1_tensor)

            warp_mesh_np = (
                (warp_mesh[0]+1)*127.5).cpu().detach().numpy().transpose(1, 2, 0)
            warp_mesh_mask_np = warp_mesh_mask[0].cpu(
            ).detach().numpy().transpose(1, 2, 0)
            inpu1_np = (
                (input1_tensor[0]+1)*127.5).cpu().detach().numpy().transpose(1, 2, 0)

            # calculate psnr/ssim
            psnr = compare_psnr(inpu1_np*warp_mesh_mask_np,
                                warp_mesh_np*warp_mesh_mask_np, data_range=255)
            ssim = compare_ssim(inpu1_np*warp_mesh_mask_np, warp_mesh_np *
                                warp_mesh_mask_np, data_range=255, channel_axis=-1)
            mse = compare_mse(inpu1_np*warp_mesh_mask_np,
                              warp_mesh_np*warp_mesh_mask_np)

            tbar.set_description('PSNR: {:.5f}'.format(psnr))
            # print('Image [{}/{}] {}: PSNR = {:.6f}'.format(i+1, len(self.test_loader), name, psnr))
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            mse_list.append(mse)
            torch.cuda.empty_cache()

        print("===================Results Analysis==================")
        print("PSNR")
        psnr_list.sort(reverse=True)
        # psnr_list_30 = psnr_list[0: 331]
        # psnr_list_60 = psnr_list[331: 663]
        # psnr_list_100 = psnr_list[663: -1]
        psnr_list_30 = psnr_list[0: int(0.3 * num)]
        psnr_list_60 = psnr_list[int(0.3 * num): int(0.6 * num)]
        psnr_list_100 = psnr_list[int(0.6 * num): -1]
        print("top 30%", np.mean(psnr_list_30))
        print("top 30~60%", np.mean(psnr_list_60))
        print("top 60~100%", np.mean(psnr_list_100))
        print('average psnr:', np.mean(psnr_list))

        print("SSIM")
        ssim_list.sort(reverse=True)
        ssim_list_30 = ssim_list[0: int(0.3 * num)]
        ssim_list_60 = ssim_list[int(0.3 * num): int(0.6 * num)]
        ssim_list_100 = ssim_list[int(0.6 * num): -1]
        print("top 30%", np.mean(ssim_list_30))
        print("top 30~60%", np.mean(ssim_list_60))
        print("top 60~100%", np.mean(ssim_list_100))
        print('average ssim:', np.mean(ssim_list))

        print("MSE")
        mse_list.sort(reverse=False)
        mse_list_30 = mse_list[0: int(0.3 * num)]
        mse_list_60 = mse_list[int(0.3 * num): int(0.6 * num)]
        mse_list_100 = mse_list[int(0.6 * num): -1]
        print("top 30%", np.mean(mse_list_30))
        print("top 30~60%", np.mean(mse_list_60))
        print("top 60~100%", np.mean(mse_list_100))
        print('average mse:', np.mean(mse_list))

        if self.crashed != 0:
            print('Warp Cashed on {} images!\n'.format(self.crashed))
            print(self.crashed_name)

    def test_other(self):
        '''
        finetune on every image
        '''
        print("=================== Test other mode! ==================")
        # create folders if it dose not exist
        self.args.path_stitched = os.path.join(args.save_path, 'stitched/')
        self.args.path_mesh = os.path.join(args.save_path, 'mesh/')
        self.args.path_warp1 = os.path.join(args.save_path, 'warp1/')
        self.args.path_warp2 = os.path.join(args.save_path, 'warp2/')
        self.args.path_mask1 = os.path.join(args.save_path, 'mask1/')
        self.args.path_mask2 = os.path.join(args.save_path, 'mask2/')
        os.makedirs(self.args.path_stitched, exist_ok=True)
        os.makedirs(self.args.path_mesh, exist_ok=True)
        os.makedirs(self.args.path_warp1, exist_ok=True)
        os.makedirs(self.args.path_warp2, exist_ok=True)
        os.makedirs(self.args.path_mask1, exist_ok=True)
        os.makedirs(self.args.path_mask2, exist_ok=True)

        if self.args.ft:
            print("=================== Start finetune!  ===================")
        else:
            print("=================== Start inference!  ===================")
        for i, batch_value in enumerate(self.test_loader):
            start_time = time.time()
            input1_tensor = batch_value[0].float()
            input2_tensor = batch_value[1].float()
            name = batch_value[2][0]
            print('===> Name:{}  {}|{}\n'.format(
                name, i+1, len(self.test_loader)))

            if args.ft:
                self.ft(input1_tensor, input2_tensor, name)
            else:
                self.inference(input1_tensor, input2_tensor, name)
            used_time = time.time() - start_time
            total_training_time = time.time() - since
            eta = used_time * (len(self.test_loader) - i + 1)
            eta = str(datetime.timedelta(seconds=int(eta)))
            print('Total training time: {:.4f}s, {:.4f} s/image, Eta: {}\n'.format(
                total_training_time, used_time, eta))

        if self.crashed != 0:
            print('Warp Cashed on {} images!\n'.format(self.crashed))
            print(self.crashed_name)

    def ft(self, input1_tensor, input2_tensor, name):
        torch.cuda.empty_cache()
        net = copy.deepcopy(self.net)
        optimizer = copy.deepcopy(self.optimizer)
        scheduler = copy.deepcopy(self.scheduler)
        start_epoch = copy.deepcopy(self.args.start_epoch)

        if self.args.cuda:
            input1_tensor = input1_tensor.cuda()
            input2_tensor = input2_tensor.cuda()

        input1_tensor_512 = net.resize_512(input1_tensor)
        input2_tensor_512 = net.resize_512(input2_tensor)

        loss_list = []
        print("=================== Start iteration  ===================")
        if args.model == 'UDIS2':
            from models.UDIS2.net_warp import build_new_ft_model, get_stitched_result
        else:
            raise RuntimeError("=> Model {} not supported".format(self.args.model))

        tbar = tqdm(range(start_epoch, start_epoch + self.args.max_iter))
        for epoch in tbar:
            net.train()

            optimizer.zero_grad()

            batch_out = build_new_ft_model(
                net, input1_tensor_512, input2_tensor_512)
            warp_mesh = batch_out['warp_mesh']
            warp_mesh_mask = batch_out['warp_mesh_mask']
            rigid_mesh = batch_out['rigid_mesh']
            mesh = batch_out['mesh']

            total_loss = self.criterion(
                input1_tensor_512, warp_mesh, warp_mesh_mask)
            total_loss.backward()
            # clip the gradient
            torch.nn.utils.clip_grad_norm_(
                net.parameters(), max_norm=3, norm_type=2)
            optimizer.step()

            current_iter = epoch - start_epoch + 1
            tbar.set_description("Training: Iteration[{:0>3}/{:0>3}] Total Loss: {:.4f} lr={:.8f}".format(
                current_iter, self.args.max_iter, total_loss.item(), optimizer.state_dict()['param_groups'][0]['lr']))

            loss_list.append(total_loss.item())

            net.eval()
            torch.cuda.empty_cache()
            if current_iter == 1:
                with torch.no_grad():
                    output = get_stitched_result(
                        input1_tensor, input2_tensor, rigid_mesh, mesh)
                path = self.args.path_stitched + \
                    name.split('.')[0] + "-iter-" + \
                    str(0).zfill(3) + '.' + name.split('.')[-1]
                if self.args.cuda:
                    cv2.imwrite(path, output['stitched'][0].cpu(
                    ).detach().numpy().transpose(1, 2, 0))
                else:
                    cv2.imwrite(
                        path, output['stitched'][0].detach().numpy().transpose(1, 2, 0))
                path = self.args.path_mesh + \
                    name.split('.')[0] + "-iter-" + \
                    str(0).zfill(3) + '.' + name.split('.')[-1]
                cv2.imwrite(path, output['stitched_mesh'])

            if current_iter >= 4:
                if abs(loss_list[current_iter-4]-loss_list[current_iter-3]) <= self.args.threshold and abs(loss_list[current_iter-3]-loss_list[current_iter-2]) <= self.args.threshold \
                        and abs(loss_list[current_iter-2]-loss_list[current_iter-1]) <= self.args.threshold:
                    
                    with torch.no_grad():
                        output = get_stitched_result(
                            input1_tensor, input2_tensor, rigid_mesh, mesh)

                    path = self.args.path_stitched + \
                        name.split('.')[
                            0] + "-iter-" + str(current_iter).zfill(3) + '.' + name.split('.')[-1]
                    if self.args.cuda:
                        cv2.imwrite(path, output['stitched'][0].cpu(
                        ).detach().numpy().transpose(1, 2, 0))
                        warp1 = output['warp1'][0].cpu(
                        ).detach().numpy().transpose(1, 2, 0)
                        warp2 = output['warp2'][0].cpu(
                        ).detach().numpy().transpose(1, 2, 0)
                        warp1_mask = output['mask1'][0].cpu(
                        ).detach().numpy().transpose(1, 2, 0)
                        warp2_mask = output['mask2'][0].cpu(
                        ).detach().numpy().transpose(1, 2, 0)
                    else:
                        cv2.imwrite(
                            path, output['stitched'][0].detach().numpy().transpose(1, 2, 0))
                        warp1 = output['warp1'][0].detach(
                        ).numpy().transpose(1, 2, 0)
                        warp2 = output['warp2'][0].detach(
                        ).numpy().transpose(1, 2, 0)
                        warp1_mask = output['mask1'][0].detach(
                        ).numpy().transpose(1, 2, 0)
                        warp2_mask = output['mask2'][0].detach(
                        ).numpy().transpose(1, 2, 0)
                    path = self.args.path_mesh + \
                        name.split('.')[
                            0] + "-iter-" + str(current_iter).zfill(3) + '.' + name.split('.')[-1]
                    cv2.imwrite(path, output['stitched_mesh'])
                    path = self.args.path_warp1 + name
                    cv2.imwrite(path, warp1)
                    path = self.args.path_warp2 + name
                    cv2.imwrite(path, warp2)
                    path = self.args.path_mask1 + name
                    cv2.imwrite(path, warp1_mask)
                    path = self.args.path_mask2 + name
                    cv2.imwrite(path, warp2_mask)
                    print('fine-tune enough! early stop!')
                    break

            if current_iter == args.max_iter:
                with torch.no_grad():
                    output = get_stitched_result(
                        input1_tensor, input2_tensor, rigid_mesh, mesh)

                path = self.args.path_stitched + \
                    name.split('.')[0] + "-iter-" + \
                    str(current_iter).zfill(3) + '.' + name.split('.')[-1]
                if self.args.cuda:
                    cv2.imwrite(path, output['stitched'][0].cpu(
                    ).detach().numpy().transpose(1, 2, 0))
                    warp1 = output['warp1'][0].cpu(
                    ).detach().numpy().transpose(1, 2, 0)
                    warp2 = output['warp2'][0].cpu(
                    ).detach().numpy().transpose(1, 2, 0)
                    warp1_mask = output['mask1'][0].cpu(
                    ).detach().numpy().transpose(1, 2, 0)
                    warp2_mask = output['mask2'][0].cpu(
                    ).detach().numpy().transpose(1, 2, 0)
                else:
                    cv2.imwrite(
                        path, output['stitched'][0].detach().numpy().transpose(1, 2, 0))
                    warp1 = output['warp1'][0].detach(
                    ).numpy().transpose(1, 2, 0)
                    warp2 = output['warp2'][0].detach(
                    ).numpy().transpose(1, 2, 0)
                    warp1_mask = output['mask1'][0].detach(
                    ).numpy().transpose(1, 2, 0)
                    warp2_mask = output['mask2'][0].detach(
                    ).numpy().transpose(1, 2, 0)
                path = self.args.path_mesh + \
                    name.split('.')[0] + "-iter-" + \
                    str(current_iter).zfill(3) + '.' + name.split('.')[-1]
                cv2.imwrite(path, output['stitched_mesh'])
                path = self.args.path_warp1 + name
                cv2.imwrite(path, warp1)
                path = self.args.path_warp2 + name
                cv2.imwrite(path, warp2)
                path = self.args.path_mask1 + name
                cv2.imwrite(path, warp1_mask)
                path = self.args.path_mask2 + name
                cv2.imwrite(path, warp2_mask)

            scheduler.step()

        print("=================== End iteration  ===================")

    def inference(self, input1_tensor, input2_tensor, name):
        torch.cuda.empty_cache()
        if self.args.cuda:
            input1_tensor = input1_tensor.cuda()
            input2_tensor = input2_tensor.cuda()

        input1_tensor_512 = self.net.resize_512(input1_tensor)
        input2_tensor_512 = self.net.resize_512(input2_tensor)

        if args.model == 'UDIS2':
            from models.UDIS2.net_warp import build_new_ft_model, get_stitched_result
        else:
            raise RuntimeError("=> Model {} not supported".format(self.args.model))


        self.net.eval()
        with torch.no_grad():
            batch_out = build_new_ft_model(
                self.net, input1_tensor_512, input2_tensor_512)
            rigid_mesh = batch_out['rigid_mesh']
            mesh = batch_out['mesh']
            output = get_stitched_result(
                    input1_tensor, input2_tensor, rigid_mesh, mesh)
            
            if output['stitched_mesh'].sum() == 0:
                self.crashed += 1
                self.crashed_name.append(name)
                print('ERROR: Warp Cashed on Image {}!'.format(name))
                torch.cuda.empty_cache()
                return

            path = self.args.path_stitched + name
            if self.args.cuda:
                cv2.imwrite(path, output['stitched'][0].cpu(
                    ).detach().numpy().transpose(1, 2, 0))
                warp1 = output['warp1'][0].cpu(
                    ).detach().numpy().transpose(1, 2, 0)
                warp2 = output['warp2'][0].cpu(
                    ).detach().numpy().transpose(1, 2, 0)
                warp1_mask = output['mask1'][0].cpu(
                    ).detach().numpy().transpose(1, 2, 0)
                warp2_mask = output['mask2'][0].cpu(
                    ).detach().numpy().transpose(1, 2, 0)
            else:
                cv2.imwrite(
                        path, output['stitched'][0].detach().numpy().transpose(1, 2, 0))
                warp1 = output['warp1'][0].detach(
                    ).numpy().transpose(1, 2, 0)
                warp2 = output['warp2'][0].detach(
                    ).numpy().transpose(1, 2, 0)
                warp1_mask = output['mask1'][0].detach(
                    ).numpy().transpose(1, 2, 0)
                warp2_mask = output['mask2'][0].detach(
                    ).numpy().transpose(1, 2, 0)
 
            path = self.args.path_mesh + name
            cv2.imwrite(path, output['stitched_mesh'])
            path = self.args.path_warp1 + name
            cv2.imwrite(path, warp1)
            path = self.args.path_warp2 + name
            cv2.imwrite(path, warp2)
            path = self.args.path_mask1 + name
            cv2.imwrite(path, warp1_mask)
            path = self.args.path_mask2 + name
            cv2.imwrite(path, warp2_mask)


def main(args):
    print('Start testing!')
    print(args)

    tester = Tester(args)
    if args.mode == 'test':
        tester.test()
    elif args.mode == 'test_output':
        tester.test_output()
    elif args.mode == 'test_other' and args.model != 'UDIS':
        tester.test_other()
    else:
        raise NotImplementedError


if __name__ == '__main__':
    since = time.time()
    args = parse_args()
    main(args)
    time_elapsed = time.time() - since
    print('Finish testing!Totally cost: {:.0f}m {:.5f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
