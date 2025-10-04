#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py
@Time    :   2024/11/21 15:57:28
@Author  :   yyywxk
@Email   :   qiulinwei@buaa.edu.cn
'''

import torch.nn.functional as F
import torch


class AlignLosses(object):
    def __init__(self, mode='UDIS2', height=512, width=512, grid_h=12, grid_w=12, cuda_flag=False):
        self.mode = mode
        self.cuda_flag = cuda_flag
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.height = height
        self.width = width

    def build_loss(self):
        if self.mode == 'UDIS':
            self.intensity_loss = F.l1_loss
            return self.udisLoss
        elif self.mode == 'UDIS2':
            from .align_loss import cal_lp_loss1, cal_lp_loss2, inter_grid_loss, intra_grid_loss
            self.cal_lp_loss1 = cal_lp_loss1
            self.cal_lp_loss2 = cal_lp_loss2
            self.inter_grid_loss = inter_grid_loss
            self.intra_grid_loss = intra_grid_loss

            return self.udis2Loss
        else:
            raise RuntimeError("=> Loss {} not supported".format(self.mode))

    def udis2Loss(self, input1_tensor, input2_tensor, output_H, output_H_inv, warp_mesh, warp_mesh_mask,
                  overlap, mesh2,
                  lam_lp1=3.0, lam_lp2=1.0, lam_grid=10):

        lp_loss_1 = self.cal_lp_loss1(
            input1_tensor, input2_tensor, output_H, output_H_inv)
        lp_loss_2 = self.cal_lp_loss2(input1_tensor, warp_mesh, warp_mesh_mask)
        nonoverlap_loss = self.inter_grid_loss(overlap, mesh2, self.grid_h, self.grid_w) + \
            self.intra_grid_loss(mesh2, self.height,
                                 self.width, self.grid_h, self.grid_w)
        g_loss = lam_lp1 * lp_loss_1 + lam_lp2 * lp_loss_2 + lam_grid * nonoverlap_loss

        return g_loss, lp_loss_1, lp_loss_2, nonoverlap_loss

    def udisLoss(self, pred, images, lam_lp1=16.0, lam_lp2=4.0, lam_lp3=1.0):
        eps = 0.01
        warped_imgs, warped_ones = pred[1:]
        target_image, target_mask = images[:, :3, ...], images[:, 3:, ...]
        target_mask = (target_mask > eps)

        warped_mask1 = (warped_ones[0] > eps)
        overlap_mask1 = target_mask & warped_mask1
        # lp_loss_1 = self.intensity_loss(
        #     warped_imgs[0][overlap_mask1], target_image[overlap_mask1])
        lp_loss_1 = self.intensity_loss(
            warped_imgs[0] * overlap_mask1, target_image * overlap_mask1)

        warped_mask2 = (warped_ones[1] > eps)
        overlap_mask2 = target_mask & warped_mask2
        # lp_loss_2 = self.intensity_loss(
        #     warped_imgs[1][overlap_mask2], target_image[overlap_mask2])
        lp_loss_2 = self.intensity_loss(
            warped_imgs[1] * overlap_mask2, target_image * overlap_mask2)

        warped_mask3 = (warped_ones[2] > eps)
        overlap_mask3 = target_mask & warped_mask3
        # lp_loss_3 = self.intensity_loss(
        #     warped_imgs[2][overlap_mask3], target_image[overlap_mask3])
        lp_loss_3 = self.intensity_loss(
            warped_imgs[2] * overlap_mask3, target_image * overlap_mask3)
        g_loss = lam_lp1 * lp_loss_1 + lam_lp2 * lp_loss_2 + lam_lp3 * lp_loss_3
        train_warp2_H3 = warped_imgs[2] * overlap_mask3

        return g_loss, lp_loss_1, lp_loss_2, lp_loss_3, train_warp2_H3



class CompoLosses(object):
    def __init__(self, mode='UDIS2', cuda_flag=True):
        self.mode = mode
        self.cuda_flag = cuda_flag

    def build_loss(self):
        if self.mode == 'UDIS':
            from .compo_loss import VGGPerceptualLoss, SeamMaskExtractor
            self.ploss = VGGPerceptualLoss(self.cuda_flag)
            self.seam_extractor = SeamMaskExtractor(self.cuda_flag)

            return self.udisLoss
        elif self.mode == 'UDIS2':
            from .compo_loss import cal_boundary_term, cal_smooth_term_stitch, cal_smooth_term_diff
            self.cal_boundary_term = cal_boundary_term
            self.cal_smooth_term_stitch = cal_smooth_term_stitch
            self.cal_smooth_term_diff = cal_smooth_term_diff

            return self.udis2Loss
        else:
            raise RuntimeError("=> Loss {} not supported".format(self.mode))

    def udis2Loss(self, warp1_tensor,  warp2_tensor, mask1_tensor, mask2_tensor,
                  stitched_image, learned_mask1,
                  lam_bt=10000, lam_st1=1000, lam_st2=1000):

        boundary_loss, boundary_mask1 = self.cal_boundary_term(
            warp1_tensor,  warp2_tensor, mask1_tensor, mask2_tensor, stitched_image)
        smooth1_loss = self.cal_smooth_term_stitch(
            stitched_image, learned_mask1)
        smooth2_loss = self.cal_smooth_term_diff(
            warp1_tensor,  warp2_tensor, learned_mask1, mask1_tensor*mask2_tensor)
        g_loss = lam_bt * boundary_loss + lam_st1 * \
            smooth1_loss + lam_st2 * smooth2_loss

        return g_loss, boundary_loss, smooth1_loss, smooth2_loss, boundary_mask1
    def udisLoss(self, stitched, images, lam_lr=100.0, lam_hr=1.0, lam_consistency=1.0,
                 lam_cont_lr=.000001, lam_seam_lr=2., lam_cont_hr=.000001, lam_seam_hr=2.):
        assert len(stitched) == 2
        eps = 0.01
        stitched_lr, stitched_hr = stitched
        stride_h = stitched_hr.shape[-2] / stitched_lr.shape[-2]
        stride_w = stitched_hr.shape[-1] / stitched_lr.shape[-1]

        image1, mask1, image2, mask2 = torch.split(
            images, [3, 1, 3, 1], dim=1)  # channel dimension
        mask1, mask2 = (mask1 > eps).int().type_as(
            images), (mask2 > eps).int().type_as(images)  # binarize
        seam1, seam2 = self.seam_extractor(mask1), self.seam_extractor(mask2)
        seam_mask1, seam_mask2 = (
            mask1 * seam2).expand(-1, 3, -1, -1), (mask2 * seam1).expand(-1, 3, -1, -1)
        image1_lr, image2_lr = self.downsample(image1, mode='bilinear', stride_h=stride_h, stride_w=stride_w), \
            self.downsample(image2, mode='bilinear',
                            stride_h=stride_h, stride_w=stride_w)
        mask1_lr, mask2_lr = self.downsample(mask1, mode='nearest', stride_h=stride_h, stride_w=stride_w), \
            self.downsample(mask2, mode='nearest', stride_h=stride_h,
                            stride_w=stride_w)  # TODO: nearest or bilinear?
        seam_mask1_lr, seam_mask2_lr = self.downsample(seam_mask1, mode='nearest', stride_h=stride_h, stride_w=stride_w), \
            self.downsample(seam_mask2, mode='nearest',
                            stride_h=stride_h, stride_w=stride_w)

        lcontent_lr = (self.ploss(stitched_lr, image1_lr, mask1_lr)[1] +
                       self.ploss(stitched_lr, image2_lr, mask2_lr)[1])
        lseam_lr = (F.l1_loss(stitched_lr[seam_mask1_lr > eps], image1_lr[seam_mask1_lr > eps]) +
                    F.l1_loss(stitched_lr[seam_mask2_lr > eps], image2_lr[seam_mask2_lr > eps]))

        lcontent_hr = (self.ploss(stitched_hr, image1, mask1)[0] +
                       self.ploss(stitched_hr, image2, mask2)[0])
        lseam_hr = (F.l1_loss(stitched_hr[seam_mask1 > eps], image1[seam_mask1 > eps]) +
                    F.l1_loss(stitched_hr[seam_mask2 > eps], image2[seam_mask2 > eps]))

        lconsistency = F.l1_loss(self.downsample(
            stitched_hr, mode='bilinear', stride_h=stride_h, stride_w=stride_w), stitched_lr)

        g_loss = lam_lr * (lam_cont_lr * lcontent_lr + lam_seam_lr * lseam_lr) + \
            lam_hr * (lam_cont_hr * lcontent_hr + lam_seam_hr * lseam_hr) + \
            lam_consistency * lconsistency

        out_dict = {}
        out_dict.update(train_hr_stitched=stitched_hr, train_lr_stitched=stitched_lr,
                        hr_seam_mask1=seam_mask1, hr_seam_mask2=seam_mask2)
        return g_loss, lcontent_lr, lseam_lr, lcontent_hr, lseam_hr, lconsistency, out_dict

    @staticmethod
    def downsample(x, mode='nearest', stride_h=4, stride_w=4):
        return F.interpolate(x, mode=mode, size=(int(round(x.shape[2] / stride_h)), int(round(x.shape[3] / stride_w))),
                             align_corners=False if mode == "bilinear" else None)

