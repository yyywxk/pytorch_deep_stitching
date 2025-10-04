#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train_compo.py
@Time    :   2024/11/23 13:29:50
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

from dataloaders import make_compo_data_loader
from models import compo_model_select
from loss import CompoLosses
from utils.saver import Saver, make_log, myprint
from utils.build_optimizer import build_optimizer
from utils.lr_scheduler import LR_Scheduler

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
import tensorboardX

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # set the GPUs
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
    parser.add_argument('--fun_main', type=str, default='train_compo.py',
                        help='main function name')
    parser.add_argument('--dataset', type=str, default='UDIS-D',
                        choices=['UDIS-D', 'UDAIS-D', 'UDAIS-D+', 'MS-COCO'],
                        help='dataset name (default: UDIS)')
    parser.add_argument('--train_path', type=str,
                        default='./Warp/UDIS-D/UDIS2/align/training/',
                        help='your data path for training after stage 1')
    parser.add_argument('--model', type=str, default='UDIS2',
                        choices=['UDIS', 'UDIS2'],
                        help='model name (default: UDIS2)')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='Workers',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=30, type=int, metavar='Epoch',
                        help='number of total epochs to run(UDIS: 30, UDIS2:  50)')
    parser.add_argument('-b', '--batch_size', default=4, type=int,
                        help='batch size (default: 4, no resize: 1)')
    parser.add_argument('--gpu-id', type=str, default='0',
                        help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
    parser.add_argument('--no-cuda', action='store_true',
                        default=False, help='disables CUDA training')
    parser.add_argument('--loss-type', type=str, default='UDIS2',
                        choices=['UDIS', 'UDIS2', 'l1'],
                        help='loss func type (default: UDIS2)')
    parser.add_argument('--freq_record', default=300, type=int,
                        help='number of iteration to record images (default: 300)')
    parser.add_argument('--freq_save', default=10, type=int,
                        help='number of epoch to save model (default: 10)')
    # model
    parser.add_argument('--nclasses', default=1, type=int,
                        help='class for model (default: 1)')

    # dataset
    # define the image resolution
    parser.add_argument('--height', type=int, default=512,
                        help='height of input images (UDIS: 640, UDIS2/UDAIS-D/UDAIS-D+: 512)')
    parser.add_argument('--width', type=int, default=512,
                        help='width of input images (UDIS: 640, UDIS2/UDAIS-D/UDAIS-D+: 512)')
    parser.add_argument('--resize', action='store_true', default=True,
                        help='training using resized input for more big batch size')

    # optimizer params
    parser.add_argument('--optim', default='adam',
                        choices=['adam', 'sgd'], help='optimizer')
    parser.add_argument('--lr', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate (default: 1e-4)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='SGD momentum')
    parser.add_argument('--nesterov', action='store_true',
                        default=False, help='To use nesterov or not.')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='WD', help='weight decay (default: 0)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'], help='lr scheduler mode: (default: poly)')
    parser.add_argument('--decay-rate', '--dr', default=0.97, type=float,
                        metavar='DR', help='decay rate (default: 0.96 for step, 0.97 for poly)')

    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        # default='./run_com/UDIS-D/UDIS2/experiment_0/epoch30_model.pth',
                        help='put the path to resuming file if needed')
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    parser.add_argument('--run_path', type=str,
                        default='./run_com/',
                        help='save your experiments')

    # define the weight in the loss
    # for UDIS2
    parser.add_argument('--lam_bt', default=10000, type=float,
                        help='weight of boundary term (default: 10000)')
    parser.add_argument('--lam_st1', default=1000, type=float,
                        help='weight of smooth term on stitched image (default: 1000)')
    parser.add_argument('--lam_st2', default=1000, type=float,
                        help='weight of smooth term on different images (default: 1000)')
    # for UDIS
    parser.add_argument('--lam_lr', default=100.0, type=float,
                        help='weight of LR branch (default: 100.0)')
    parser.add_argument('--lam_hr', default=1.0, type=float,
                        help='weight of HR branch (default: 1.0)')
    parser.add_argument('--lam_consistency', default=1.0, type=float,
                        help='weight of consistency (default: 1.0)')
    parser.add_argument('--lam_cont_lr', default=.000001, type=float,
                        help='content loss weight of LR branch (default: .000001)')
    parser.add_argument('--lam_seam_lr', default=2.0, type=float,
                        help='seam loss weight of LR branch (default: 2.0)')
    parser.add_argument('--lam_cont_hr', default=.000001, type=float,
                        help='content loss weight of HR branch (default: .000001)')
    parser.add_argument('--lam_seam_hr', default=2.0, type=float,
                        help='seam loss weight of HR branch (default: 2.0)')

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

    if not os.path.exists(args.train_path):
        raise RuntimeError(
            "=> no file found at '{}'".format(args.train_path))

    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        # Define Dataloader
        self.train_loader = make_compo_data_loader(args, mode='train')

        # Define network
        self.net = compo_model_select(args, args.model)
        if args.cuda:
            # self.net = torch.nn.DataParallel(self.net, device_ids=self.args.gpu_id)
            # self.net = torch.nn.DataParallel(self.net)
            self.net = self.net.cuda()

        # Define Optimizer
        self.optimizer = build_optimizer(args, self.net)

        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler,
                                      self.optimizer, decay_rate=args.decay_rate)

        self.criterion = CompoLosses(
            mode=args.loss_type, cuda_flag=args.cuda).build_loss()

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

    def _log_iteration(self, warp1_tensor, warp2_tensor, batch_out):
        """Log intermediate results to tensorboard during training"""
        # Log learning rate
        self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], self.args.glob_iter)

        # Log model specific outputs
        if self.args.model == 'UDIS':
            # Log input images
            self.writer.add_image("input1", 
                                  warp1_tensor[0], self.args.glob_iter)
            self.writer.add_image("input2", 
                                  warp2_tensor[0], self.args.glob_iter)
            self.writer.add_image(
                "train_hr_stitched", 
                batch_out["train_hr_stitched"][0], 
                self.args.glob_iter
            )
            self.writer.add_image(
                "train_lr_stitched", 
                batch_out["train_lr_stitched"][0], 
                self.args.glob_iter
            )
            self.writer.add_image(
                "hr_seam_mask1", 
                batch_out["hr_seam_mask1"][0], 
                self.args.glob_iter
            )
            self.writer.add_image(
                "hr_seam_mask2", 
                batch_out['hr_seam_mask2'][0], 
                self.args.glob_iter
            )
        else:
            # Log input images
            self.writer.add_image("input1", (warp1_tensor[0]+1.)/2., self.args.glob_iter)
            self.writer.add_image("input2", (warp2_tensor[0]+1.)/2., self.args.glob_iter)
            if self.args.model == 'UDIS2':
                self.writer.add_image(
                    "stitched_image", 
                    (batch_out["stitched_image"][0]+1.)/2., 
                    self.args.glob_iter
                )
                self.writer.add_image(
                    "learned_mask1", 
                    (batch_out["learned_mask1"][0]+1.)/2., 
                    self.args.glob_iter
                )
                self.writer.add_image(
                    "boundary_mask1", 
                    (batch_out['boundary_mask1'][0]+1.)/2., 
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
        """Unified training function for both modes"""
        loss_dict = defaultdict(list)
        
        # Print training info
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        myprint(self.logging, f'=>Epoches [{epoch+1:0>3}/{self.args.epochs:0>3}], learning rate = {lr:.8f}', True)
        
        self.net.train()
        tbar = tqdm(self.train_loader)

        for i, batch_value in enumerate(tbar):
            # Process inputs
            warp1_tensor = batch_value[0].float()
            warp2_tensor = batch_value[1].float()
            mask1_tensor = batch_value[2].float()
            mask2_tensor = batch_value[3].float()

            if self.args.model == 'UDIS':
                # (-1,1)->(0,1)
                # tanh is used in UDIS codes, and here is sigmoid
                warp1_tensor = (warp1_tensor + 1.0)/2.0
                warp2_tensor = (warp2_tensor + 1.0)/2.0

            if self.args.cuda:
                warp1_tensor = warp1_tensor.cuda()
                warp2_tensor = warp2_tensor.cuda()
                mask1_tensor = mask1_tensor.cuda()
                mask2_tensor = mask2_tensor.cuda()

            # Forward pass
            self.optimizer.zero_grad()

            if self.args.model == 'UDIS':
                imgs = torch.cat((warp1_tensor, mask1_tensor, warp2_tensor, mask2_tensor), 1)
                batch_out = self.net(imgs)
                total_loss, *losses, loss_out = self.criterion(
                    batch_out, imgs, 
                    self.args.lam_lr, self.args.lam_hr, self.args.lam_consistency,
                    self.args.lam_cont_lr, self.args.lam_seam_lr, 
                    self.args.lam_cont_hr, self.args.lam_seam_hr
                )
                loss_names = ['content_lr_loss', 'seam_lr_loss', 'content_hr_loss', 'seam_hr_loss', 'consistency_loss']
            elif self.args.model == 'UDIS2':
                batch_out = self.net.build_model(warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor)
                total_loss, *losses, boundary_mask1 = self.criterion(
                        warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor,
                        batch_out['stitched_image'], batch_out['learned_mask1'],
                        self.args.lam_bt, self.args.lam_st1, self.args.lam_st2
                    )
                loss_names = ['boundary_loss', 'smooth1_loss', 'smooth2_loss']
                batch_out['boundary_mask1'] = boundary_mask1

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
                if self.args.model == 'UDIS':
                    self._log_iteration(warp1_tensor, warp2_tensor, loss_out)
                elif self.args.model == 'UDIS2' or self.args.model == 'UDRSIS' or self.args.model == 'UDRSIS2':
                    self._log_iteration(warp1_tensor, warp2_tensor, batch_out)

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
