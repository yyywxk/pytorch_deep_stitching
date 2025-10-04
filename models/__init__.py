#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py
@Time    :   2024/11/21 15:57:19
@Author  :   yyywxk
@Email   :   qiulinwei@buaa.edu.cn
'''

def align_model_select(args, model='UDIS2'):
    if model == 'UDIS':
        from .UDIS.net import Network
        net = Network('models/UDIS/align_origin.yaml', mode_align=True, cuda_flag=args.cuda, img_size=args.height)
        # net = Network('models/UDIS/align_variant.yaml', mode_align=True, cuda_flag=args.cuda, img_size=args.height)
        # net = Network('models/UDIS/align_yolo.yaml', mode_align=True, cuda_flag=args.cuda, img_size=args.height)
        return net
    elif model == 'UDIS2':
        from .UDIS2.net_warp import Network
        net = Network(args.height, args.width, args.grid_h, args.grid_w, args.cuda)
        return net
    else:
        raise NotImplementedError
    
def compo_model_select(args, model='UDIS2'):
    if model == 'UDIS':
        from .UDIS.net import Network
        net = Network('models/UDIS/fuse_origin.yaml', mode_align=False, cuda_flag=args.cuda, img_size=args.height)
        # net = Network('models/UDIS/fuse_variant.yaml', mode_align=False, cuda_flag=args.cuda, img_size=args.height)
        # net = Network('models/UDIS/fuse_yolo.yaml', mode_align=False, cuda_flag=args.cuda, img_size=args.height)
        return net
    elif model == 'UDIS2':
        from .UDIS2.net_compo import Network
        net = Network(args.nclasses)
        return net
    else:
        raise NotImplementedError
