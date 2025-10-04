#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   align_loss.py
@Time    :   2024/11/22 19:41:59
@Author  :   yyywxk
@Email   :   qiulinwei@buaa.edu.cn
'''

from matplotlib.pyplot import grid
import torch
import torch.nn.functional as F


def l_num_loss(img1, img2, l_num=1):
    return torch.mean(torch.abs((img1 - img2)**l_num))


def cal_lp_loss(input1, input2, output_H, output_H_inv, warp_mesh, warp_mesh_mask):
    _, _, img_h, img_w = input1.size()

    # part one: sym homo loss with color balance
    delta1 = (torch.sum(output_H[:, 0:3, :, :], [2, 3]) - torch.sum(
        input1*output_H[:, 3:6, :, :], [2, 3])) / torch.sum(output_H[:, 3:6, :, :], [2, 3])
    input1_balance = input1 + \
        delta1.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

    delta2 = (torch.sum(output_H_inv[:, 0:3, :, :], [2, 3]) - torch.sum(
        input2*output_H_inv[:, 3:6, :, :], [2, 3])) / torch.sum(output_H_inv[:, 3:6, :, :], [2, 3])
    input2_balance = input2 + \
        delta2.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

    lp_loss_1 = l_num_loss(input1_balance*output_H[:, 3:6, :, :], output_H[:, 0:3, :, :], 1) + l_num_loss(
        input2_balance*output_H_inv[:, 3:6, :, :], output_H_inv[:, 0:3, :, :], 1)

    # part two: tps loss with color balance
    delta3 = (torch.sum(warp_mesh, [2, 3]) - torch.sum(input1 *
              warp_mesh_mask, [2, 3])) / torch.sum(warp_mesh_mask, [2, 3])
    input1_newbalance = input1 + \
        delta3.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

    lp_loss_2 = l_num_loss(input1_newbalance*warp_mesh_mask, warp_mesh, 1)

    lp_loss = 3. * lp_loss_1 + 1. * lp_loss_2

    return lp_loss


def cal_lp_loss1(input1, input2, output_H, output_H_inv):
    _, _, img_h, img_w = input1.size()

    # part one: sym homo loss with color balance
    delta1 = (torch.sum(output_H[:, 0:3, :, :], [2, 3]) - torch.sum(
        input1*output_H[:, 3:6, :, :], [2, 3])) / torch.sum(output_H[:, 3:6, :, :], [2, 3])
    input1_balance = input1 + \
        delta1.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

    if output_H_inv is not None:
        delta2 = (torch.sum(output_H_inv[:, 0:3, :, :], [2, 3]) - torch.sum(
            input2*output_H_inv[:, 3:6, :, :], [2, 3])) / torch.sum(output_H_inv[:, 3:6, :, :], [2, 3])
        input2_balance = input2 + \
        delta2.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

        lp_loss_1 = l_num_loss(input1_balance*output_H[:, 3:6, :, :], output_H[:, 0:3, :, :], 1) + l_num_loss(
            input2_balance*output_H_inv[:, 3:6, :, :], output_H_inv[:, 0:3, :, :], 1)
    else:
        lp_loss_1 = l_num_loss(input1_balance*output_H[:, 3:6, :, :], output_H[:, 0:3, :, :], 1)

    return lp_loss_1


def cal_lp_loss2(input1, warp_mesh, warp_mesh_mask):
    _, _, img_h, img_w = input1.size()

    # part two: tps loss with color balance
    delta3 = (torch.sum(warp_mesh, [2, 3]) - torch.sum(input1 *
              warp_mesh_mask, [2, 3])) / torch.sum(warp_mesh_mask, [2, 3])
    input1_newbalance = input1 + \
        delta3.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

    lp_loss_2 = l_num_loss(input1_newbalance*warp_mesh_mask, warp_mesh, 1)

    return lp_loss_2


def inter_grid_loss(overlap, mesh, grid_h=12, grid_w=12):

    ##############################
    # compute horizontal edges
    w_edges = mesh[:, :, 0:grid_w, :] - mesh[:, :, 1:grid_w+1, :]
    # compute angles of two successive horizontal edges
    cos_w = torch.sum(w_edges[:, :, 0:grid_w-1, :] * w_edges[:, :, 1:grid_w, :], 3) / (torch.sqrt(torch.sum(w_edges[:, :, 0:grid_w-1, :]
                                                                                                            * w_edges[:, :, 0:grid_w-1, :], 3))*torch.sqrt(torch.sum(w_edges[:, :, 1:grid_w, :]*w_edges[:, :, 1:grid_w, :], 3)))
    # horizontal angle-preserving error for two successive horizontal edges
    delta_w_angle = 1 - cos_w
    # horizontal angle-preserving error for two successive horizontal grids
    delta_w_angle = delta_w_angle[:, 0:grid_h,
                                  :] + delta_w_angle[:, 1:grid_h+1, :]
    ##############################

    ##############################
    # compute vertical edges
    h_edges = mesh[:, 0:grid_h, :, :] - mesh[:, 1:grid_h+1, :, :]
    # compute angles of two successive vertical edges
    cos_h = torch.sum(h_edges[:, 0:grid_h-1, :, :] * h_edges[:, 1:grid_h, :, :], 3) / (torch.sqrt(torch.sum(h_edges[:, 0:grid_h-1, :, :]
                                                                                                            * h_edges[:, 0:grid_h-1, :, :], 3))*torch.sqrt(torch.sum(h_edges[:, 1:grid_h, :, :]*h_edges[:, 1:grid_h, :, :], 3)))
    # vertical angle-preserving error for two successive vertical edges
    delta_h_angle = 1 - cos_h
    # vertical angle-preserving error for two successive vertical grids
    delta_h_angle = delta_h_angle[:, :, 0:grid_w] + \
        delta_h_angle[:, :, 1:grid_w+1]
    ##############################

    # on overlapping regions
    depth_diff_w = (1-torch.abs(overlap[:, :, 0:grid_w-1] -
                    overlap[:, :, 1:grid_w])) * overlap[:, :, 0:grid_w-1]
    error_w = depth_diff_w * delta_w_angle
    # on overlapping regions
    depth_diff_h = (1-torch.abs(overlap[:, 0:grid_h-1, :] -
                    overlap[:, 1:grid_h, :])) * overlap[:, 0:grid_h-1, :]
    error_h = depth_diff_h * delta_h_angle

    return torch.mean(error_w) + torch.mean(error_h)


# intra-grid constraint
def intra_grid_loss(pts, height=512, width=512, grid_h=12, grid_w=12):

    max_w = width/grid_w * 2
    max_h = height/grid_h * 2

    delta_x = pts[:, :, 1:grid_w+1, 0] - pts[:, :, 0:grid_w, 0]
    delta_y = pts[:, 1:grid_h+1, :, 1] - pts[:, 0:grid_h, :, 1]

    loss_x = F.relu(delta_x - max_w)
    loss_y = F.relu(delta_y - max_h)
    loss = torch.mean(loss_x) + torch.mean(loss_y)

    return loss


