from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import imp
from typing import no_type_check
from kornia.geometry.epipolar.projection import depth

import torchvision.models as models
import torch
import torch.nn as nn
import os

from .networks.msra_resnet import get_pose_net
from .networks.dlav0 import get_pose_net as get_dlav0
from .networks.pose_dla_dcn import get_pose_net as get_dla_dcn
from .networks.resnet_dcn import get_pose_net as get_pose_net_dcn
from .networks.large_hourglass import get_large_hourglass_net
# from .networks.multiagent_pose_dla_dcn import get_pose_net as get_multiagent_dla_dcn
from .networks.multiagent_pose_dla_dcn_noise import get_pose_net as get_multiagent_dla_dcn

_model_factory = {
    'res': get_pose_net,  # default Resnet with deconv
    'dlav0': get_dlav0,  # default DLAup
    'dla': get_dla_dcn,
    'resdcn': get_pose_net_dcn,
    'hourglass': get_large_hourglass_net,
    'multiagentdla': get_multiagent_dla_dcn
}


def create_model(arch, heads, head_conv, message_mode=0, trans_layer=[3], coord='Local', warp_mode='HW', depth_mode='Unique', feat_mode='inter', feat_shape=[192, 352], round=1, compress_flag=False, comm_thre=0.1, sigma=0.0):
    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
    arch = arch[:arch.find('_')] if '_' in arch else arch
    get_model = _model_factory[arch]
    if arch.startswith('multiagent'):
        print('Trans_layer: ', trans_layer)
        print('Coord: ', coord)
        model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv,\
                            message_mode=message_mode, trans_layer=trans_layer, coord=coord, warp_mode=warp_mode, depth_mode=depth_mode, \
                            feat_mode=feat_mode, feat_shape=feat_shape, round=round, compress_flag=compress_flag, comm_thre=comm_thre, sigma=sigma)
    else:
        model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
    return model


def load_model(model, model_path, optimizer=None, resume=False,
               lr=None, lr_step=None):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = 'If you see this, your model does not fully load the ' + \
          'pre-trained weight. Please make sure ' + \
          'you have correctly specified --arch xxx ' + \
          'or set the correct --num_classes for your own dataset.'
    ####################### Initial Self Attn With Cross Attn Parameter ##########
    # attn_params = [k for k in state_dict if 'attn' in k]
    # for k in attn_params:
    #     state_dict['self'+k[5:]] = state_dict[k]
    ####################### Initial Self Attn With Cross Attn Parameter ##########

    ###################### Initial Cross Attn K With Cross Attn Parameter ##########
    # for key_word in ['cross_attn', 'linear1_', 'linear2_', 'norm1_', 'norm2_', 'dropout0_', 'dropout1_', 'dropout2_']:
    #     attn_params = [k for k in state_dict if key_word in k]
    #     for k in attn_params:
    #         for i in range(1,3):
    #             state_dict[key_word+str(i)+k[len(key_word)+1:]] = state_dict[key_word+'0'+k[len(key_word)+1:]]
    #             # print(key_word+str(i)+k[len(key_word)+1:])
    ###################### Initial Self Attn With Cross Attn Parameter ##########

    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}. {}'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
            ###################### Initial Self Attn With Cross Attn Parameter ##########
            # last_checkpoint = torch.load('/GPFS/data/yhu/code/BEV_Det/CoDet/exp/multiagent_det/HW_QualityMapMessage_Translayer0_CommMask_Transformer_BandwidthAware_Repeat/model_86.pth', map_location=lambda storage, loc: storage)
            # last_checkpoint = torch.load('/GPFS/data/yhu/code/BEV_Det/CoDet/exp/multiagent_det/HW_QualityMapMessage_Translayer0_CommMask_Transformer_MultiRound_BandwidthAware/model_last.pth', map_location=lambda storage, loc: storage)
            # state_dict[k] = last_checkpoint['state_dict'][k]
            ###################### Initial Self Attn With Cross Attn Parameter ##########
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        for optim in optimizer:
            if optim in checkpoint:
                optimizer[optim].load_state_dict(checkpoint[optim])
                start_epoch = checkpoint['epoch']
                start_lr = lr
                for step in lr_step:
                    if start_epoch >= step:
                        start_lr *= 0.1
                for param_group in optimizer[optim].param_groups:
                    param_group['lr'] = start_lr
                print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model


def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    if not (optimizer is None):
        for optim in optimizer:
            data[optim] = optimizer[optim].state_dict()
    torch.save(data, path)
