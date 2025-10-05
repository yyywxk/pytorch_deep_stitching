#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train_align.py
@Time    :   2024/11/22 15:10:28
@Author  :   yyywxk
@Email   :   qiulinwei@buaa.edu.cn
'''


import warnings
import argparse
import os
import numpy as np
from tqdm import tqdm
import time
import datetime
import sys
from collections import defaultdict

from dataloaders import make_align_data_loader
from models import align_model_select

from loss import AlignLosses
from utils.saver import Saver, make_log, myprint
from utils.build_optimizer import build_optimizer
from utils.lr_scheduler import LR_Scheduler

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
import tensorboardX

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # set the GPUs
os.environ['TORCH_HOME'] = './pretrained_models/'
warnings.filterwarnings('ignore')


def parse_args():
    '''
    To define parameters and settings
    '''
    parser = argparse.ArgumentParser(
        description='PyTorch Codes for Unsupervised Deep Image Stitching Training')
    # --------------------------------- Base Settings ----------------------------
    # training settings
    parser.add_argument('--fun_main', type=str, default='train_align.py',
                        help='main function name')
    parser.add_argument('--dataset', type=str, default='UDIS-D',
                        choices=['UDIS-D', 'UDAIS-D', 'UDAIS-D+', 'MS-COCO'],
                        help='dataset name (default: UDIS-D)')
    parser.add_argument('--model', type=str, default='UDIS2',
                        choices=['UDIS', 'UDIS2'],
                        help='model name (default: UDIS2)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='Workers',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=100, type=int, metavar='Epoch',
                        help='number of total epochs to run (MS-COCO: 150, UDIS: 50, UDIS2: 100)')
    parser.add_argument('-b', '--batch_size', default=4, type=int,
                        help='batch size (default: 4)')
    parser.add_argument('--gpu-id', type=str, default='0',
                        help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
    parser.add_argument('--no-cuda', action='store_true',
                        default=False, help='disables CUDA training')
    parser.add_argument('--loss-type', type=str, default='UDIS2',
                        choices=['UDIS', 'UDIS2'],
                        help='loss func type (default: UDIS2)')
    parser.add_argument('--freq_record', default=600, type=int,
                        help='number of iteration to record images (default: 600)')
    parser.add_argument('--freq_save', default=10, type=int,
                        help='number of epoch to save model (default: 10)')

    # optimizer params
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

    # dataset
    # define the image resolution
    parser.add_argument('--height', type=int, default=512,
                        help='height of input images (UDIS: 128, UDIS2/UDAIS-D/UDAIS-D+: 512)')
    parser.add_argument('--width', type=int, default=512,
                        help='width of input images (UDIS: 128, UDIS2/UDAIS-D/UDAIS-D+: 512)')

    # checking point
    parser.add_argument('--resume', type=str, 
                        default=None,
                        # default='./run_align/UDIS-D/UDIS2/experiment_0/epoch100_model.pth',
                        help='put the path to resuming file if needed')
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    parser.add_argument('--run_path', type=str,
                        default='./run_align/',
                        help='save your experiments')

    # define the mesh resolution
    parser.add_argument('--grid_w', default=12, type=int,
                        help='control points number of width (default: 12)')
    parser.add_argument('--grid_h', default=12, type=int,
                        help='control points number of height (default: 12)')
    # define the weight in the loss
    # for UDIS2
    parser.add_argument('--lam_lp1', default=3.0, type=float,
                        help='weight of pixel loss 1 (UDIS 16.0; UDIS2: 3.0)')
    parser.add_argument('--lam_lp2', default=1.0, type=float,
                        help='weight of pixel loss 2 (UDIS 4.0; UDIS2: 1.0)')
    parser.add_argument('--lam_grid', default=10.0, type=float,
                        help='weight of grid loss (default:10)')
    # for UDIS
    parser.add_argument('--lam_lp3', default=1.0, type=float,
                        help='weight of grid loss (default:1.0)')

    args = parser.parse_args()
    args.func_main = sys.argv[0]
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_id = [int(s) for s in args.gpu_id.split(',')]
        except ValueError:
            raise ValueError(
                'Argument --gpu_ids must be a comma-separated list of integers only')

    if args.ft and args.resume is None:
        raise ValueError('=> no checkpoint, please set the resume path')

    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        # Define Dataloader
        self.train_loader = make_align_data_loader(args, mode='train')

        # Define network
        self.net = align_model_select(args, args.model)
        if args.cuda:
            # self.net = torch.nn.DataParallel(self.net, device_ids=self.args.gpu_id)
            # self.net = torch.nn.DataParallel(self.net)
            self.net = self.net.cuda()

        # Define Optimizer
        self.optimizer = build_optimizer(args, self.net)

        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler,
                                      self.optimizer, decay_rate=args.decay_rate)

        self.criterion = AlignLosses(
            mode=args.loss_type, height=args.height, width=args.width, grid_h=args.grid_h, grid_w=args.grid_w, cuda_flag=args.cuda).build_loss()

        # Define Saver
        self.saver = Saver(args)
        # Define Tensorboard Summary
        self.writer = tensorboardX.SummaryWriter(self.saver.experiment_dir)
        self.logging = make_log(self.saver.experiment_dir)

        pytorch_total_params = sum(
            p.numel() for p in self.net.parameters() if p.requires_grad)
        myprint(self.logging, "Total_params: {}".format(pytorch_total_params))
        self.saver.save_experiment_config(pytorch_total_params)
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError(
                    "=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            self.net.load_state_dict(checkpoint['model'])
            if args.ft:
                # Clear start epoch if fine-tuning
                checkpoint['epoch'] = 0
                self.args.start_epoch = 0
                self.args.glob_iter = 0
                print('Fine-tune!')
            else:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.args.start_epoch = checkpoint['epoch']
                self.args.glob_iter = checkpoint['glob_iter']
                self.scheduler.last_epoch = self.args.start_epoch
        else:
            self.args.start_epoch = 0
            self.args.glob_iter = 0
            print('training from stratch!')
        
    def _log_training_info(self, epoch, loss_dict):
        """
        Log training metrics to tensorboard and console
        Args:
            epoch: Current epoch number
            loss_dict: Dictionary containing different types of losses
        """
        # Calculate mean values once to avoid repeated computation
        mean_losses = {name: np.mean(values) for name, values in loss_dict.items()}
        
        # Log to console
        myprint(self.logging, 'Training: Epoch[{:0>3}/{:0>3}] Global loss: {:.5f}'.format(
            epoch + 1, self.args.epochs, mean_losses['total_loss']))
        
        # Log individual losses
        for name, mean_value in mean_losses.items():
            if name != 'total_loss':
                myprint(self.logging, f'  {name}: {mean_value:.5f}')
            # Log to tensorboard
            self.writer.add_scalar(f'Epoch/{name}', mean_value, epoch)

    def _log_iteration(self, input1_tensor, input2_tensor, batch_out):
        """Log intermediate results to tensorboard during training"""
        # Log input images
        self.writer.add_image("input1", (input1_tensor[0]+1.)/2., self.args.glob_iter)
        self.writer.add_image("input2", (input2_tensor[0]+1.)/2., self.args.glob_iter)
        
        # Log model specific outputs
        if self.args.model == 'UDIS':
            self.writer.add_image(
                "train_warp2_H3", 
                (batch_out[0]+1.)/2., 
                self.args.glob_iter
            )
        elif self.args.model == 'UDIS2':
            self.writer.add_image(
                "warp_H", 
                (batch_out['output_H'][0, 0:3, :, :]+1.)/2., 
                self.args.glob_iter
            )
            self.writer.add_image(
                "warp_mesh", 
                (batch_out['warp_mesh'][0]+1.)/2., 
                self.args.glob_iter
            )

    def _log_losses_to_tensorboard(self, losses, names):
        """Log individual losses to tensorboard"""
        # Log learning rate
        self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], self.args.glob_iter)

        for name, loss in zip(names, losses):
            self.writer.add_scalar(
                f'Iter/{name}_iter', 
                loss.item(), 
                self.args.glob_iter
            )

    def _save_checkpoint(self, epoch):
        """Save model checkpoint"""
        filename = f'epoch{epoch+1}_model.pth'
        model_save_path = os.path.join(self.saver.experiment_dir, filename)
        state = {
            'model': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch+1,
            "glob_iter": self.args.glob_iter
        }
        torch.save(state, model_save_path)
        myprint(self.logging, 'Saving model!\n')            

    def train(self, epoch):
        # Initialize loss tracking
        loss_dict = defaultdict(list)
        
        # Log learning rate
        myprint(self.logging, f'=>Epoches [{epoch+1:0>3}/{self.args.epochs:0>3}], learning rate = {self.optimizer.param_groups[0]["lr"]:.8f}', True)
        
        self.net.train()
        tbar = tqdm(self.train_loader)

        for i, batch_value in enumerate(tbar):
            # Prepare input tensors
            input1_tensor = batch_value[0].float()
            input2_tensor = batch_value[1].float()
            
            if self.args.cuda:
                input1_tensor = input1_tensor.cuda()
                input2_tensor = input2_tensor.cuda()

            self.optimizer.zero_grad()

            # Forward pass based on model type
            if self.args.model == 'UDIS':
                imgs = torch.cat((input1_tensor, input2_tensor), 1)
                pred_warp = self.net(imgs)
                total_loss, *losses, batch_out = self.criterion(pred_warp, imgs,
                    self.args.lam_lp1, self.args.lam_lp2, self.args.lam_lp3)
                loss_names = ['l1_loss1', 'l1_loss2', 'l1_loss3']
            elif self.args.model == 'UDIS2':
                # UDIS2
                from models.UDIS2.net_warp import build_model
                batch_out = build_model(self.net, input1_tensor, input2_tensor)
                total_loss, *losses = self.criterion(
                        input1_tensor, input2_tensor,
                        batch_out['output_H'], batch_out['output_H_inv'],
                        batch_out['warp_mesh'], batch_out['warp_mesh_mask'],
                        batch_out['overlap'], batch_out['mesh2'],
                        self.args.lam_lp1, self.args.lam_lp2, self.args.lam_grid
                    )
                loss_names = ['overlap_loss1', 'overlap_loss2', 'nonoverlap_loss']
            else:
                raise RuntimeError("=> Model {} not supported".format(self.args.model))

            if torch.isnan(total_loss):
                myprint(self.logging, f"NaN loss detected at epoch {epoch+1}")
                self._save_checkpoint(epoch)
                exit()
            # elif (total_loss.item() > 1000.0) and (epoch > 10):
            #     myprint(self.logging, f"Loss has suddenly increased at epoch {epoch+1}")
            #     self._save_checkpoint(epoch)
            #     exit()

            # Backward pass
            total_loss.backward()
            clip_grad_norm_(self.net.parameters(), max_norm=3, norm_type=2)
            self.optimizer.step()

            # Track losses
            loss_dict['total_loss'].append(total_loss.item())
            for name, loss in zip(loss_names, losses):
                loss_dict[name].append(loss.item())

            # Update progress bar
            tbar.set_description(f'Train loss: {np.mean(loss_dict["total_loss"]):.5f}')
            self._log_losses_to_tensorboard(losses, loss_names)

            # Log to tensorboard
            if i % self.args.freq_record == 0:
                self._log_iteration(input1_tensor, input2_tensor, batch_out)

            self.args.glob_iter += 1

        # Log epoch metrics
        self._log_training_info(epoch, loss_dict)
        
        # Update scheduler and save checkpoint
        self.scheduler.step()
        if epoch == 0 or (epoch+1) % self.args.freq_save == 0 or (epoch+1) == self.args.epochs:
            self._save_checkpoint(epoch)


def main(args):
    print(args)
    print('--------------------------------- Start training {} --------------------------------'.format(args.model))
    trainer = Trainer(args)
    myprint(trainer.logging, 'Starting Epoch: {}'.format(
        trainer.args.start_epoch))
    myprint(trainer.logging, 'Total Epoches:'.format(trainer.args.epochs))
    since = time.time()

    myprint(trainer.logging, 'Start training!\n')
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        start_time = time.time()
        trainer.train(epoch)

        used_time = time.time() - start_time
        total_training_time = time.time() - since
        eta = used_time * (args.epochs - epoch - 1)
        eta = str(datetime.timedelta(seconds=int(eta)))
        myprint(trainer.logging, 'Total training time: {:.4f}s, {:.4f} s/epoch, Eta: {}\n'.format(
            total_training_time, used_time, eta))

    trainer.writer.close()
    myprint(trainer.logging, 'Finish training!')
    time_elapsed = time.time() - since
    myprint(trainer.logging, 'Totally cost: {:.0f}m {:.5f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':
    args = parse_args()
    main(args)
