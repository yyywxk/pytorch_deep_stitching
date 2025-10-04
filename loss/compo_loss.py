#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   compo_loss.py
@Time    :   2024/11/22 19:42:30
@Author  :   yyywxk
@Email   :   qiulinwei@buaa.edu.cn
'''

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


#  UDIS2
def l_num_loss(img1, img2, l_num=1):
    return torch.mean(torch.abs((img1 - img2)**l_num))


def boundary_extraction(mask):

    ones = torch.ones_like(mask)
    zeros = torch.zeros_like(mask)
    # define kernel
    in_channel = 1
    out_channel = 1
    kernel = [[1, 1, 1],
              [1, 1, 1],
              [1, 1, 1]]
    kernel = torch.FloatTensor(kernel).expand(out_channel, in_channel, 3, 3)
    if torch.cuda.is_available():
        kernel = kernel.cuda()
        ones = ones.cuda()
        zeros = zeros.cuda()
    weight = nn.Parameter(data=kernel, requires_grad=False)

    # dilation
    x = F.conv2d(1-mask, weight, stride=1, padding=1)
    x = torch.where(x < 1, zeros, ones)
    x = F.conv2d(x, weight, stride=1, padding=1)
    x = torch.where(x < 1, zeros, ones)
    x = F.conv2d(x, weight, stride=1, padding=1)
    x = torch.where(x < 1, zeros, ones)
    x = F.conv2d(x, weight, stride=1, padding=1)
    x = torch.where(x < 1, zeros, ones)
    x = F.conv2d(x, weight, stride=1, padding=1)
    x = torch.where(x < 1, zeros, ones)
    x = F.conv2d(x, weight, stride=1, padding=1)
    x = torch.where(x < 1, zeros, ones)
    x = F.conv2d(x, weight, stride=1, padding=1)
    x = torch.where(x < 1, zeros, ones)

    return x*mask


def cal_boundary_term(inpu1_tesnor, inpu2_tesnor, mask1_tesnor, mask2_tesnor, stitched_image):
    boundary_mask1 = mask1_tesnor * boundary_extraction(mask2_tesnor)
    boundary_mask2 = mask2_tesnor * boundary_extraction(mask1_tesnor)

    loss1 = l_num_loss(inpu1_tesnor*boundary_mask1,
                       stitched_image*boundary_mask1, 1)
    loss2 = l_num_loss(inpu2_tesnor*boundary_mask2,
                       stitched_image*boundary_mask2, 1)

    return loss1+loss2, boundary_mask1


def cal_smooth_term_stitch(stitched_image, learned_mask1):
    delta = 1
    dh_mask = torch.abs(
        learned_mask1[:, :, 0:-1*delta, :] - learned_mask1[:, :, delta:, :])
    dw_mask = torch.abs(
        learned_mask1[:, :, :, 0:-1*delta] - learned_mask1[:, :, :, delta:])
    dh_diff_img = torch.abs(
        stitched_image[:, :, 0:-1*delta, :] - stitched_image[:, :, delta:, :])
    dw_diff_img = torch.abs(
        stitched_image[:, :, :, 0:-1*delta] - stitched_image[:, :, :, delta:])

    dh_pixel = dh_mask * dh_diff_img
    dw_pixel = dw_mask * dw_diff_img

    loss = torch.mean(dh_pixel) + torch.mean(dw_pixel)

    return loss


def cal_smooth_term_diff(img1, img2, learned_mask1, overlap):

    diff_feature = torch.abs(img1-img2)**2 * overlap

    delta = 1
    dh_mask = torch.abs(
        learned_mask1[:, :, 0:-1*delta, :] - learned_mask1[:, :, delta:, :])
    dw_mask = torch.abs(
        learned_mask1[:, :, :, 0:-1*delta] - learned_mask1[:, :, :, delta:])
    dh_diff_img = torch.abs(
        diff_feature[:, :, 0:-1*delta, :] + diff_feature[:, :, delta:, :])
    dw_diff_img = torch.abs(
        diff_feature[:, :, :, 0:-1*delta] + diff_feature[:, :, :, delta:])

    dh_pixel = dh_mask * dh_diff_img
    dw_pixel = dw_mask * dw_diff_img

    loss = torch.mean(dh_pixel) + torch.mean(dw_pixel)

    return loss


    """Calculate illumination balance loss across seam regions
    
    Args:
        img1: First input image
        img2: Second input image
        mask1: Mask for first image
        mask2: Mask for second image
        seam_width: Width of the region to consider on each side of seam (default: 5)
        
    Returns:
        Illumination balance loss between regions on both sides of the seam
    """
    # Get seam region
    seam = mask2 * boundary_extraction(learned_mask1)  # shape: (B,1,H,W)
    
    # Expand seam region
    kernel = torch.ones(1, 1, seam_width, seam_width)
    if torch.cuda.is_available():
        kernel = kernel.cuda()
    
    # Get regions on both sides of seam
    left_region = F.conv2d(seam * mask1, kernel, padding=seam_width//2) > 0
    right_region = F.conv2d(seam * mask2, kernel, padding=seam_width//2) > 0
    
    # Calculate average intensity for both regions
    left_intensity = torch.mean(img1 * left_region, dim=1, keepdim=True)
    right_intensity = torch.mean(img2 * right_region, dim=1, keepdim=True)
    
    # Calculate L1 loss between intensities
    illumination_loss = F.l1_loss(left_intensity, right_intensity)
    
    return illumination_loss

# UDIS
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, cuda_flag=True, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        # vgg_features = torchvision.models.vgg16(pretrained=True).features.eval().to(device)
        self.cuda_flag = cuda_flag
        vgg_features = torchvision.models.vgg19(
            pretrained=True).features.eval().cuda() if cuda_flag else torchvision.models.vgg19(
            pretrained=True).features.eval()
        
        # vgg16: 4, 9, 16, 23, 30; vgg19: 4, 9, 18, 27, 36
        # blocks = [vgg_features[:16], vgg_features[16:30]]
        blocks = [vgg_features[:18], vgg_features[18:36]]
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)

        self.resize = torch.nn.functional.interpolate if resize else None
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])

    def forward(self, x, y, mask=None):
        if mask is None:
            mask = torch.ones_like(x[:, :1, :, :])
        x, y = x * mask, y * mask
        if self.resize:
            x = self.resize(x, mode='bilinear', size=(
                224, 224), align_corners=False)
            y = self.resize(y, mode='bilinear', size=(
                224, 224), align_corners=False)
        x = self.normalize(x).float()
        y = self.normalize(y).float()

        losses = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            losses.append(F.mse_loss(x, y))

            # msk = self.resize(mask, mode='nearest', size=x.shape[-2:]) > 0
            # ch = x.shape[1]
            # msk = msk.expand(-1, ch, -1, -1)
            # losses.append(F.mse_loss(x[msk], y[msk]))

        return losses


class SeamMaskExtractor(object):
    def __init__(self, cuda_flag=True):
        self.cuda_flag = cuda_flag
        sobel_x = np.array([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]], dtype=np.float32)
        sobel_y = np.array([[-1., -2., -1.],
                            [0., 0., 0.],
                            [1., 2., 1.]], dtype=np.float32)
        ones = np.ones((3, 3), dtype=np.float32)
        kernels = []
        for kernel in [sobel_x, sobel_y, ones]:
            kernel = np.reshape(kernel, (1, 1, 3, 3))
            kernels.append(torch.from_numpy(kernel).cuda() if self.cuda_flag else torch.from_numpy(kernel)) 
        self.edge_kernel_x, self.edge_kernel_y, self.seam_kernel = kernels

    @torch.no_grad()
    def __call__(self, mask):
        # shape(b,1,h,w)
        assert isinstance(mask, torch.Tensor) and len(
            mask.shape) == 4 and mask.size(1) == 1
        if self.edge_kernel_x.dtype != mask.dtype:
            self.edge_kernel_x = self.edge_kernel_x.type_as(mask)
            self.edge_kernel_y = self.edge_kernel_y.type_as(mask)
            self.seam_kernel = self.seam_kernel.type_as(mask)

        mask_dx = F.conv2d(mask, self.edge_kernel_x,
                           bias=None, stride=1, padding=1).abs()
        mask_dy = F.conv2d(mask, self.edge_kernel_y,
                           bias=None, stride=1, padding=1).abs()
        edge = (mask_dx + mask_dy).clamp_(0, 1)
        for _ in range(3):  # dilate
            edge = F.conv2d(edge, self.seam_kernel, bias=None,
                            stride=1, padding=1).clamp_(0, 1)

        return edge


