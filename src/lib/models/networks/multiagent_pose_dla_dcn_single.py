'''
Author: yhu
Contact: phyllis1sjtu@outlook.com
LastEditTime: 2021-07-26 02:08:55
Description: 
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import imp

import os
import math
import logging
import ipdb
from kornia.geometry.epipolar.projection import depth
from kornia.geometry.transform.imgwarp import warp_affine
from matplotlib.colors import hsv_to_rgb
import numpy as np
from os.path import join
import math

import torch
from torch import nn
from torch._C import device
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import kornia
import matplotlib.pyplot as plt
import cv2

from .DCNv2.dcn_v2 import DCN, CF_DCN
import sys
sys.path.append('/GPFS/data/yhu/code/CoDet/src/lib/models/networks')
from .convolutional_rnn import Conv2dGRU

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        return y

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        # fc = self.fc
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(
            self.channels[-1], num_classes,
            kernel_size=1, stride=1, padding=0, bias=True)
        self.load_state_dict(model_weights)
        # self.fc = fc


def dla34(pretrained=True, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
    return model

class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DeformConv(nn.Module):
    def __init__(self, chi, cho, k_size=3, dilation=1, mode='DCN'):
        super(DeformConv, self).__init__()
        self.mode = mode
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = eval(mode)(chi, cho, kernel_size=(k_size,k_size), stride=1, padding=(k_size-1)//2, dilation=dilation, deformable_groups=1)

    def forward(self, x, return_offset=False):
        if self.mode == 'DCN':
            x = self.conv(x)
            x = self.actf(x)
            return x
        else:
            x, offset, mask = self.conv(x)
            x = self.actf(x)
            if return_offset:
                return x, offset, mask
            else:
                return x


class IDAUp(nn.Module):

    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])  
            proj = DeformConv(c, o)
            node = DeformConv(o, o)
     
            up = nn.ConvTranspose2d(o, o, f * 2, stride=f, 
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)
                 
        
    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            if layers[i].shape == layers[i - 1].shape:
                layers[i] = node(layers[i] + layers[i - 1])
            else:
                layers[i] = node(layers[i])



class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]] # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) -i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
        return x


class conv2DBatchNormRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        n_filters,
        k_size,
        stride,
        padding,
        bias=True,
        dilation=1,
        is_batchnorm=True,
    ):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if is_batchnorm:
            self.cbr_unit = nn.Sequential(
                conv_mod, nn.BatchNorm2d(int(n_filters)), nn.ReLU(inplace=True)
            )
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

class Saliency_Sampler(nn.Module):
    def __init__(self, input_size_h, input_size_w, input_channel):
        super(Saliency_Sampler, self).__init__()

        self.grid_size_h = input_size_h // 4 # 31 #
        self.grid_size_w = input_size_w // 4 # 57 #
        self.padding_size = 30 # 20 # 10 # 30
        self.global_size_h = self.grid_size_h+2*self.padding_size
        self.global_size_w = self.grid_size_w+2*self.padding_size
        self.input_size_net_h = input_size_h
        self.input_size_net_w = input_size_w
        self.conv_last = nn.Conv2d(input_channel+2,1,kernel_size=1,padding=0,stride=1)
        gaussian_weights = torch.FloatTensor(makeGaussian(2*self.padding_size+1, fwhm = 13))

        # Spatial transformer localization-network
        self.filter = nn.Conv2d(1, 1, kernel_size=(2*self.padding_size+1,2*self.padding_size+1),bias=False)
        self.filter.weight[0].data[:,:,:] = gaussian_weights

        self.P_basis = torch.zeros(2,self.grid_size_h+2*self.padding_size, self.grid_size_w+2*self.padding_size)
        for k in range(2):
            for i in range(self.global_size_h):
                for j in range(self.global_size_w):
                    self.P_basis[k,i,j] = k*(i-self.padding_size)/(self.grid_size_h-1.0)+(1.0-k)*(j-self.padding_size)/(self.grid_size_w-1.0)

    def create_grid(self, x):
        P = torch.autograd.Variable(torch.zeros(1,2,self.grid_size_h+2*self.padding_size, self.grid_size_w+2*self.padding_size).cuda(),requires_grad=False)
        P[0,:,:,:] = self.P_basis
        P = P.expand(x.size(0),2,self.grid_size_h+2*self.padding_size, self.grid_size_w+2*self.padding_size)

        x_cat = torch.cat((x,x),1)
        p_filter = self.filter(x)
        x_mul = torch.mul(P,x_cat).view(-1,1,self.global_size_h,self.global_size_w)
        all_filter = self.filter(x_mul).view(-1,2,self.grid_size_h,self.grid_size_w)

        x_filter = all_filter[:,0,:,:].contiguous().view(-1,1,self.grid_size_h,self.grid_size_w)
        y_filter = all_filter[:,1,:,:].contiguous().view(-1,1,self.grid_size_h,self.grid_size_w)

        x_filter = x_filter/p_filter
        y_filter = y_filter/p_filter

        xgrids = x_filter*2-1
        ygrids = y_filter*2-1
        xgrids = torch.clamp(xgrids,min=-1,max=1)
        ygrids = torch.clamp(ygrids,min=-1,max=1)


        xgrids = xgrids.view(-1,1,self.grid_size_h,self.grid_size_w)
        ygrids = ygrids.view(-1,1,self.grid_size_h,self.grid_size_w)

        grid = torch.cat((xgrids,ygrids),1)

        grid = nn.Upsample(size=(self.input_size_net_h, self.input_size_net_w), mode='bilinear')(grid)

        grid = torch.transpose(grid,1,2)
        grid = torch.transpose(grid,2,3)

        return grid

    def forward(self, x, images_warped):
        b, c, h, w = x.shape
        h_i = torch.arange(h).unsqueeze(-1).expand(-1,w).unsqueeze(0)/h
        w_i = torch.arange(w).unsqueeze(0).expand(h,-1).unsqueeze(0)/w
        im_i = torch.cat([h_i, w_i], dim=0).to(x.device) # (2, h, w)
        xs = F.relu(x)
        xs = torch.cat([xs, im_i.unsqueeze(0).expand(b,-1,-1,-1)], dim=1)  # (b, c+2, h, w)
        xs = self.conv_last(xs)
        xs = nn.Upsample(size=(self.grid_size_h,self.grid_size_w), mode='bilinear')(xs)
        xs = xs.view(-1,self.grid_size_h*self.grid_size_w)
        # xs = F.softmax(xs, dim=-1)
        xs = F.sigmoid(xs)
        xs = xs.view(-1,1,self.grid_size_h,self.grid_size_w)
        # print('xs: {}-{}'.format(xs.min(), xs.max()))
        image = xs[0].detach().cpu() * 255.
        # print('saliency: {}-{}'.format(image.min(), image.max()))
        image = kornia.tensor_to_image(image.byte())
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite('saliency.png', image)
        xs_hm = nn.ReplicationPad2d(self.padding_size)(xs)
        image = xs_hm[0].detach().cpu() * 255.
        # print('saliency_pad: {}-{}'.format(image.min(), image.max()))
        image = kornia.tensor_to_image(image.byte())
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite('saliency_pad.png', image)
        grid = self.create_grid(xs_hm)
        x_sampled = F.grid_sample(x, grid)
        image = x_sampled.max(dim=1)[0][0].detach().cpu() * 255.
        image = kornia.tensor_to_image(image.byte())
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite('saliency_sampled.png', image)
        
        image_sampled = F.grid_sample(images_warped, grid)
        image = images_warped[0].detach().cpu() * 255.
        image = kornia.tensor_to_image(image.byte())
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite('saliency_image.png', image)
        image = image_sampled[0].detach().cpu() * 255.
        image = kornia.tensor_to_image(image.byte())
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite('saliency_image_sampled.png', image)
        return x_sampled


class DLASeg(nn.Module):
    def __init__(self, base_name, heads, pretrained, down_ratio, final_kernel,
                 last_level, head_conv, out_channel=0, message_mode='NO_MESSAGE', trans_layer=[3], coord='Local', warp_mode='HW', depth_mode='Unique',
                 feat_mode='inter', feat_shape=[192, 352]):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.warp_mode = warp_mode
        self.depth_mode = depth_mode
        self.feat_mode = feat_mode

        if coord == 'Local' or self.feat_mode in ['inter', 'fused']:
            self.base = globals()[base_name](pretrained=pretrained)
            channels = self.base.channels
            scales = [2 ** i for i in range(len(channels[self.first_level:]))]
            self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)

        if coord in ['Global', 'Joint'] and self.feat_mode in ['early', 'fused']:
            self.base_early = globals()[base_name](pretrained=pretrained)
            channels = self.base_early.channels
            scales = [2 ** i for i in range(len(channels[self.first_level:]))]
            self.dla_up_early = DLAUp(self.first_level, channels[self.first_level:], scales)
        
        if out_channel == 0:
            out_channel = channels[self.first_level]
            # self.ida_up_early = IDAUp(out_channel, channels[self.first_level:self.last_level], 
            #                     [2 ** i for i in range(self.last_level - self.first_level)])

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level], 
                            [2 ** i for i in range(self.last_level - self.first_level)])
        
        if coord == 'Joint':
            self.ida_up_i = IDAUp(out_channel, channels[self.first_level:self.last_level], 
                                [2 ** i for i in range(self.last_level - self.first_level)])
            
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
              fc = nn.Sequential(
                  nn.Conv2d(channels[self.first_level], head_conv,
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(head_conv, classes, 
                    kernel_size=final_kernel, stride=1, 
                    padding=final_kernel // 2, bias=True))
              if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
              else:
                fill_fc_weights(fc)
            else:
              fc = nn.Conv2d(channels[self.first_level], classes, 
                  kernel_size=final_kernel, stride=1, 
                  padding=final_kernel // 2, bias=True)
              if 'hm' in head:
                fc.bias.data.fill_(-2.19)
              else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)
        
        # Message generator
        self.key_size = 128
        self.query_size = 32
        self.message_mode = message_mode
        self.coord = coord
        self.trans_layer = trans_layer

        if self.trans_layer[0] == -1:
            self.conv0 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
            self.conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        
        if self.warp_mode in ['LW', 'RLW']:
            hw0 = (800 * 448) // 16
            hw1 = (352 * 192)
            # self.fc0 = nn.Linear(hw0, hw1, bias=False)
            self.fc1 = nn.Linear(hw0//4, hw1//4, bias=False)
            self.fc2 = nn.Linear(hw0//16, hw1//16, bias=False)
            self.fc3 = nn.Linear(hw0//64, hw1//64, bias=False)
        elif self.warp_mode in ['DW', 'DADW']:
            k_size = 5 # 7, 3
            mode = 'CF_DCN' if self.warp_mode == 'DADW' else 'DCN' 
            self.DC0 = DeformConv(64, 64, k_size=k_size, mode=mode)
            self.DC1 = DeformConv(128, 128, k_size=k_size, mode=mode)
            self.DC2 = DeformConv(256, 256, k_size=k_size, mode=mode)
            self.DC3 = DeformConv(512, 512, k_size=k_size, mode=mode)
        elif self.warp_mode == 'SW':
            self.saliency0 = Saliency_Sampler(192, 352, 64)
            # self.saliency1 = Saliency_Sampler(96, 176, 128)
            # self.saliency2 = Saliency_Sampler(48, 88, 256)
            # self.saliency3 = Saliency_Sampler(24, 44, 512)
        elif self.warp_mode == 'SWU':
            self.saliency0 = Saliency_Sampler(448, 800, 64)
        if self.depth_mode == 'Weighted':
            self.conv0 = nn.Conv2d(64, 9, kernel_size=1, stride=1, padding=0)
            self.conv1 = nn.Conv2d(128, 9, kernel_size=1, stride=1, padding=0)
            self.conv2 = nn.Conv2d(256, 9, kernel_size=1, stride=1, padding=0)
            self.conv3 = nn.Conv2d(512, 9, kernel_size=1, stride=1, padding=0)
        
        self.feat_H, self.feat_W = feat_shape   # [192, 352], [96, 128]

    def LocalCoord_forward(self, images, trans_mats):
        # ------------------------ Image Feature Extraction ------------------- #
        b, num_agents, img_c, img_h, img_w = images.size()
        images = images.view(b*num_agents, img_c, img_h, img_w)
        
        # Encoder
        x = self.base(images)
        x = self.dla_up(x)  # list [(B, 64, 112, 200), (B, 128, 56, 100), (B, 256, 28, 50), (B, 512, 14, 25)] B = b * num_agents

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(y[-1]) # (b*num_agent, 2, 112, 200)

        return [z]


    def GlobalCoord_forward(self, images, trans_mats, shift_mats, map_scale):
        b, num_agents, img_c, img_h, img_w = images.size()
        images = images.view(b*num_agents, img_c, img_h, img_w)
        
        if self.feat_mode in ['early', 'fused']:
            # print('Warp image')
            cur_trans_mats = trans_mats[0].view(b*num_agents, 3, 3)
            worldgrid2worldcoord_mat = torch.Tensor(np.array([[map_scale, 0, 0], [0, map_scale, 0], [0, 0, 1]])).to(cur_trans_mats.device)
            feat_zoom_mats = torch.Tensor(np.array(np.diag([4, 4, 1]), dtype=np.float32)).to(cur_trans_mats.device)
            cur_trans_mats = feat_zoom_mats @ shift_mats[0].view(b*num_agents, 3, 3) @ torch.inverse(cur_trans_mats @ worldgrid2worldcoord_mat).contiguous()
            images_warped = kornia.warp_perspective(images, cur_trans_mats, dsize=(self.feat_H*4, self.feat_W*4))
            # image = images[0].detach().cpu() * 255.
            # image = kornia.tensor_to_image(image.byte())
            # cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # cv2.imwrite('warp_img.png', image)
            
            x_early = self.base_early(images_warped)
            x_early = self.dla_up_early(x_early)

        if self.feat_mode in ['inter', 'fused']:
            # Encoder
            x = self.base(images)
            x = self.dla_up(x)  # list [(B, 64, 112, 200), (B, 128, 56, 100), (B, 256, 28, 50), (B, 512, 14, 25)] B = b * num_agents

            scale = 1/map_scale

            if self.depth_mode == 'Weighted':
                # get depth weights
                depth_weighted_feat_maps = [[] for _ in range(len(trans_mats))]
                depth_weights_list = []
                for c_layer, feat_map in enumerate(x):
                    depth_weights = eval('self.conv'+str(c_layer))(feat_map)    # (B, D, h, w)
                    depth_weights = F.softmax(depth_weights, dim=1)
                    depth_weighted_feat_map = depth_weights.unsqueeze(2) * feat_map.unsqueeze(1)   # (B, D, C, h, w)
                    depth_weighted_feat_map = depth_weighted_feat_map.unbind(1)
                    for depth_layer in range(len(trans_mats)):
                        depth_weighted_feat_maps[depth_layer].append(depth_weighted_feat_map[depth_layer])
                    depth_weights_list.append(depth_weights)
            else:
                trans_mats = [trans_mats[0]]
                depth_weighted_feat_maps = [x]
            
            global_x_multi_depth = []
            if self.depth_mode == 'Weighted':
                global_depth_weights_list = []
                for depth_layer, init_trans_mats in enumerate(trans_mats):
                    cur_x = depth_weighted_feat_maps[depth_layer]
                    global_x_HW = []
                    warp_images_list = []
                    init_trans_mats = init_trans_mats.view(b*num_agents, 3, 3)

                    for c_layer, feat_map in enumerate(cur_x):
                        _, c, h, w = feat_map.size()
                        shift_mats[c_layer] = shift_mats[c_layer].view(b*num_agents, 3, 3)

                        worldgrid2worldcoord_mat = torch.Tensor(np.array([[2**c_layer*scale, 0, 0], [0, 2**c_layer*scale, 0], [0, 0, 1]])).to(init_trans_mats.device)
                        img_trans_mats = shift_mats[c_layer].view(b*num_agents, 3, 3).contiguous() @ torch.inverse(init_trans_mats @ worldgrid2worldcoord_mat).contiguous()
                        warp_images = kornia.warp_perspective(images, img_trans_mats, dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                        warp_images_list.append(warp_images)

                        #########################################################################
                        #                              Hard Warping                             #
                        #########################################################################
                        # uav_i --> global coord
                        worldgrid2worldcoord_mat = torch.Tensor(np.array([[2**c_layer*map_scale, 0, 0], [0, 2**c_layer*map_scale, 0], [0, 0, 1]])).to(init_trans_mats.device)
                        feat_zoom_mats = torch.Tensor(np.array(np.diag([2**(c_layer+2), 2**(c_layer+2), 1]), dtype=np.float32)).to(init_trans_mats.device)
                        cur_trans_mats = shift_mats[c_layer].contiguous() @ torch.inverse(init_trans_mats @ worldgrid2worldcoord_mat).contiguous() @ feat_zoom_mats
                        global_feat = kornia.warp_perspective(feat_map, cur_trans_mats, dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                        
                        # Two choice to classify z
                        if c_layer == 0:
                            # Choice1: supervise the sampe feature map warpped to multiple altitudes
                            if depth_layer == 0:
                                global_depth_weights = kornia.warp_perspective(depth_weights_list[c_layer], cur_trans_mats, dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                            # Choice2: supervise different feature map warpped to the corresponding altitudes
                            # global_depth_weights = kornia.warp_perspective(depth_weights_list[c_layer][:,depth_layer:depth_layer+1], cur_trans_mats, dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))                            
                            global_depth_weights_list.append(global_depth_weights)
                        
                        global_x_HW.append(global_feat)
                        
                    global_x_multi_depth.append(global_x_HW)
                
                global_x_HW = []
                for c_layer in range(len(x)):
                    cur_feat_map = [global_x_multi_depth[i][c_layer].unsqueeze(1) for i in range(len(global_x_multi_depth))]    # (B,D,C,H,W)
                    cur_feat_map = torch.cat(cur_feat_map, dim=1)
                    cur_feat_map = torch.mean(cur_feat_map, dim=1)
                    global_x_HW.append(cur_feat_map)
                
                global_x = []
                for c_layer, (feat_map, feat_map_HW) in enumerate(zip(x, global_x_HW)):
                    _, c, h, w = feat_map.size()
                    shift_mats[c_layer] = shift_mats[c_layer].view(b*num_agents, 3, 3)
                    # # 2. Get the value mat (trans feature to global coord)  # val_mat: (b, k_agents, q_agents, c, h, w)
                    if self.warp_mode in ['DW', 'DADW']:
                        #########################################################################
                        #                          Deformable Warping                           #
                        #########################################################################
                        # h_b, w_b = int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)
                        # grid_y, grid_x = torch.meshgrid(torch.arange(0, h_b), torch.arange(0, w_b))
                        # grid = torch.stack((grid_x, grid_y), 2).type_as(feat_map)  # (w, h, 2)
                        # grid.requires_grad = False
                        worldgrid2worldcoord_mat = torch.Tensor(np.array([[2**c_layer*scale, 0, 0], [0, 2**c_layer*scale, 0], [0, 0, 1]])).to(init_trans_mats.device)
                        feat_zoom_mats = torch.Tensor(np.array(np.diag([2**(c_layer+2), 2**(c_layer+2), 1]), dtype=np.float32)).to(init_trans_mats.device)
                        cur_trans_mats = shift_mats[c_layer] @ torch.inverse(init_trans_mats @ worldgrid2worldcoord_mat).contiguous() @ feat_zoom_mats
                        # cur_trans_mats = shift_mats[c_layer] @ torch.inverse(cur_trans_mats @ worldgrid2worldcoord_mat).contiguous() @ feat_zoom_mats
                        
                        ### Residual 
                        # global_feat_HW = kornia.warp_perspective(feat_map, cur_trans_mats, dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                        global_feat_res = kornia.warp_perspective(feat_map, cur_trans_mats, mode='nearest', dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                        global_feat_res = eval('self.DC'+str(c_layer))(global_feat_res, return_offset=False)
                        global_feat = feat_map_HW + global_feat_res
                        
                        ### Bilinear
                        # global_feat = kornia.warp_perspective(feat_map, cur_trans_mats, dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                        # global_feat, offset, mask = eval('self.DC'+str(c_layer))(global_feat, return_offset=True)
                        
                        ### Nearest
                        # global_feat = kornia.warp_perspective(feat_map, cur_trans_mats, mode='nearest', dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                        # global_feat, offset, mask = eval('self.DC'+str(c_layer))(global_feat, return_offset=True)
                        
                        # print(c_layer, global_feat.shape, offset.shape, mask.shape) # 0, (1, 50, 192, 352); 0, (96, 176); 0, (48, 88); 0, (24, 44)
                        # offset_list.append(offset)
                        # mask_list.append(mask)

                        # b, N, h, w = offset.shape
                        # offset_heatmap = offset.view(b, 2, N//2, h, w).cpu().numpy()
                        # fig, axes = plt.subplots(3, 3)
                        # for i, h_i in enumerate(range(50//(2**c_layer), h, 50//(2**c_layer))):
                        #     for j, w_j in enumerate(range(100//(2**c_layer), w, 100//(2**c_layer))):
                        #         axes[i,j].plot(w_j, h_i, 'bo')
                        #         axes[i,j].plot(offset_heatmap[0, 1, :, h_i, w_j], offset_heatmap[0, 0, :, h_i, w_j], 'r*')
                        #         axes[i,j].imshow((warp_images[0].detach().cpu().numpy().transpose(1,2,0) * 255.).astype('uint8'))
                        #         axes[i,j].set_xticks([])
                        #         axes[i,j].set_yticks([])
                        # plt.savefig('offset.png')
                        # plt.close()
                        # import ipdb; ipdb.set_trace()
                    else:
                        global_feat = feat_map_HW
                    global_x.append(global_feat)
            else:
                for depth_layer, init_trans_mats in enumerate(trans_mats):
                    cur_x = depth_weighted_feat_maps[depth_layer]
                    global_x = []
                    offset_list = []
                    mask_list = []
                    warp_images_list = []
                    trans_mats_list = []
                    init_trans_mats = init_trans_mats.view(b*num_agents, 3, 3)
                    # cur_trans_mats = init_trans_mats.view(b*num_agents, 3, 3)
                    # fig, axes = plt.subplots(5, 4)

                    for c_layer, feat_map in enumerate(cur_x):
                        _, c, h, w = feat_map.size()
                        shift_mats[c_layer] = shift_mats[c_layer].view(b*num_agents, 3, 3)

                        worldgrid2worldcoord_mat = torch.Tensor(np.array([[2**c_layer*scale, 0, 0], [0, 2**c_layer*scale, 0], [0, 0, 1]])).to(init_trans_mats.device)
                        img_trans_mats = shift_mats[c_layer].view(b*num_agents, 3, 3).contiguous() @ torch.inverse(init_trans_mats @ worldgrid2worldcoord_mat).contiguous()
                        warp_images = kornia.warp_perspective(images, img_trans_mats, dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                        warp_images_list.append(warp_images)

                        # # 2. Get the value mat (trans feature to global coord)  # val_mat: (b, k_agents, q_agents, c, h, w)
                        if self.warp_mode in ['LW', 'RLW']:
                            #########################################################################
                            #                             Learnt Warping                            #
                            #########################################################################
                            worldgrid2worldcoord_mat = torch.Tensor(np.array([[2**c_layer*scale, 0, 0], [0, 2**c_layer*scale, 0], [0, 0, 1]])).to(init_trans_mats.device)
                            feat_zoom_mats = torch.Tensor(np.array(np.diag([2**(c_layer+2), 2**(c_layer+2), 1]), dtype=np.float32)).to(init_trans_mats.device)
                            cur_trans_mats = shift_mats[c_layer] @ torch.inverse(init_trans_mats @ worldgrid2worldcoord_mat).contiguous() @ feat_zoom_mats
                            global_feat_HW = kornia.warp_perspective(feat_map, cur_trans_mats, dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                            
                            if c_layer == 0:
                                global_feat = global_feat_HW
                            else:
                                flat_feat_map = feat_map.view(-1, c, h*w).contiguous()
                                global_feat_res = eval('self.fc'+str(c_layer))(flat_feat_map)
                                global_feat_res = global_feat_res.view(-1, c, int(self.feat_H/2**c_layer), int(self.feat_W/2**c_layer))
                                if self.warp_mode == 'LW':
                                    global_feat = global_feat_res
                                else:
                                    global_feat = global_feat_HW + global_feat_res
                        elif self.warp_mode in ['DW', 'DADW']:
                            #########################################################################
                            #                          Deformable Warping                           #
                            #########################################################################
                            # h_b, w_b = int(192/2**c_layer*scale), int(352/2**c_layer*scale)
                            # grid_y, grid_x = torch.meshgrid(torch.arange(0, h_b), torch.arange(0, w_b))
                            # grid = torch.stack((grid_x, grid_y), 2).type_as(feat_map)  # (w, h, 2)
                            # grid.requires_grad = False
                            worldgrid2worldcoord_mat = torch.Tensor(np.array([[2**c_layer*scale, 0, 0], [0, 2**c_layer*scale, 0], [0, 0, 1]])).to(init_trans_mats.device)
                            feat_zoom_mats = torch.Tensor(np.array(np.diag([2**(c_layer+2), 2**(c_layer+2), 1]), dtype=np.float32)).to(init_trans_mats.device)
                            cur_trans_mats = shift_mats[c_layer] @ torch.inverse(init_trans_mats @ worldgrid2worldcoord_mat).contiguous() @ feat_zoom_mats
                            # cur_trans_mats = shift_mats[c_layer] @ torch.inverse(cur_trans_mats @ worldgrid2worldcoord_mat).contiguous() @ feat_zoom_mats
                            
                            ### Residual 
                            global_feat_HW = kornia.warp_perspective(feat_map, cur_trans_mats, dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                            global_feat_res = kornia.warp_perspective(feat_map, cur_trans_mats, mode='nearest', dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                            global_feat_res = eval('self.DC'+str(c_layer))(global_feat_res, return_offset=False)
                            global_feat = global_feat_HW + global_feat_res
                            
                            ### Bilinear
                            # global_feat = kornia.warp_perspective(feat_map, cur_trans_mats, dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                            # global_feat, offset, mask = eval('self.DC'+str(c_layer))(global_feat, return_offset=True)
                            
                            ### Nearest
                            # global_feat = kornia.warp_perspective(feat_map, cur_trans_mats, mode='nearest', dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                            # global_feat, offset, mask = eval('self.DC'+str(c_layer))(global_feat, return_offset=True)
                            
                            # print(c_layer, global_feat.shape, offset.shape, mask.shape) # 0, (1, 50, 192, 352); 0, (96, 176); 0, (48, 88); 0, (24, 44)
                            # offset_list.append(offset)
                            # mask_list.append(mask)

                            # b, N, h, w = offset.shape
                            # offset_heatmap = offset.view(b, 2, N//2, h, w).cpu().numpy()
                            # fig, axes = plt.subplots(3, 3)
                            # for i, h_i in enumerate(range(50//(2**c_layer), h, 50//(2**c_layer))):
                            #     for j, w_j in enumerate(range(100//(2**c_layer), w, 100//(2**c_layer))):
                            #         axes[i,j].plot(w_j, h_i, 'bo')
                            #         axes[i,j].plot(offset_heatmap[0, 1, :, h_i, w_j], offset_heatmap[0, 0, :, h_i, w_j], 'r*')
                            #         axes[i,j].imshow((warp_images[0].detach().cpu().numpy().transpose(1,2,0) * 255.).astype('uint8'))
                            #         axes[i,j].set_xticks([])
                            #         axes[i,j].set_yticks([])
                            # plt.savefig('offset.png')
                            # plt.close()
                            # import ipdb; ipdb.set_trace()
                        elif self.warp_mode == 'SW':
                            worldgrid2worldcoord_mat = torch.Tensor(np.array([[2**c_layer*scale, 0, 0], [0, 2**c_layer*scale, 0], [0, 0, 1]])).to(init_trans_mats.device)
                            feat_zoom_mats = torch.Tensor(np.array(np.diag([2**(c_layer+2), 2**(c_layer+2), 1]), dtype=np.float32)).to(init_trans_mats.device)
                            cur_trans_mats = shift_mats[c_layer] @ torch.inverse(init_trans_mats @ worldgrid2worldcoord_mat).contiguous() @ feat_zoom_mats
                            
                            ### Residual
                            # global_feat_HW = kornia.warp_perspective(feat_map, cur_trans_mats, dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                            # global_feat_res = kornia.warp_perspective(feat_map, cur_trans_mats, mode='nearest', dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                            # global_feat_res = eval('self.saliency'+str(c_layer))(global_feat_res)
                            # global_feat = global_feat_HW + global_feat_res

                            ### Bilinear
                            global_feat = kornia.warp_perspective(feat_map, cur_trans_mats, dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                            if c_layer == 0:
                                global_feat = eval('self.saliency'+str(c_layer))(global_feat, images_warped)
                            
                            ### Nearest
                            # global_feat = kornia.warp_perspective(feat_map, cur_trans_mats, mode='nearest', dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                            # global_feat = eval('self.saliency'+str(c_layer))(global_feat)
                        elif self.warp_mode == 'SWU':
                            worldgrid2worldcoord_mat = torch.Tensor(np.array([[2**c_layer*scale, 0, 0], [0, 2**c_layer*scale, 0], [0, 0, 1]])).to(init_trans_mats.device)
                            feat_zoom_mats = torch.Tensor(np.array(np.diag([2**(c_layer+2), 2**(c_layer+2), 1]), dtype=np.float32)).to(init_trans_mats.device)
                            cur_trans_mats = shift_mats[c_layer] @ torch.inverse(init_trans_mats @ worldgrid2worldcoord_mat).contiguous() @ feat_zoom_mats

                            ### Bilinear
                            if c_layer == 0:
                                feat_map = eval('self.saliency'+str(c_layer))(feat_map, images_warped)
                            global_feat = kornia.warp_perspective(feat_map, cur_trans_mats, dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                        else:
                            #########################################################################
                            #                              Hard Warping                             #
                            #########################################################################
                            # uav_i --> global coord
                            # worldgrid2worldcoord_mat = torch.Tensor(np.array([[500/(w*scale), 0, -200], [0, 500/(h*scale), -250], [0, 0, 1]])).to(init_trans_mats.device)
                            # worldgrid2worldcoord_mat = torch.Tensor(np.array([[1/(w*scale), 0, 0], [0, 1/(h*scale), 0], [0, 0, 1]])).to(init_trans_mats.device)
                            # worldgrid2worldcoord_mat = torch.Tensor(np.array([[1/(2**(c_layer+4)*scale), 0, 0], [0, 1/(2**(c_layer+4)*scale), 0], [0, 0, 1]])).to(init_trans_mats.device)
                            # worldgrid2worldcoord_mat = torch.Tensor(np.array([[1/(500/2**c_layer*scale), 0, 0], [0, 1/(500/2**c_layer*scale), 0], [0, 0, 1]])).to(init_trans_mats.device)
                            worldgrid2worldcoord_mat = torch.Tensor(np.array([[2**c_layer*map_scale, 0, 0], [0, 2**c_layer*map_scale, 0], [0, 0, 1]])).to(init_trans_mats.device)
                            feat_zoom_mats = torch.Tensor(np.array(np.diag([2**(c_layer+2), 2**(c_layer+2), 1]), dtype=np.float32)).to(init_trans_mats.device)
                            cur_trans_mats = shift_mats[c_layer].contiguous() @ torch.inverse(init_trans_mats @ worldgrid2worldcoord_mat).contiguous() @ feat_zoom_mats
                            global_feat = kornia.warp_perspective(feat_map, cur_trans_mats, dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                            # global_feat = kornia.warp_perspective(feat_map, cur_trans_mats, mode='nearest', dsize=(int(192/2**c_layer*scale), int(352/2**c_layer*scale)))
                            # global_feat = kornia.warp_perspective(feat_map, cur_trans_mats, dsize=(2**(c_layer+4)*scale, 2**(c_layer+4)*scale)) # (b*num_agents, c, h, w)
                            # global_feat = kornia.warp_perspective(feat_map, cur_trans_mats, dsize=(h*scale, w*scale)) # (b*num_agents, c, h, w)
                            # global_feat = kornia.resize(global_feat, size=(h*4, w*4))

                        global_x.append(global_feat)
                        
                        # val_mats = global_feat.max(dim=-3)[0].detach().cpu().numpy()
                        # for j in range(min(5, val_mats.shape[0])):
                        #     axes[j, c_layer].imshow((val_mats[j]*255.).astype('uint8'))
                        #     axes[j, c_layer].set_xticks([])
                        #     axes[j, c_layer].set_yticks([])

                    # plt.savefig('featmap.png')
                    # plt.close()
                    global_x_multi_depth.append(global_x)
            
                if len(global_x_multi_depth) > 1:
                    global_x = []
                    for c_layer in range(len(x)):
                        cur_feat_map = [global_x_multi_depth[i][c_layer].unsqueeze(1) for i in range(len(global_x_multi_depth))]    # (B,D,C,H,W)
                        cur_feat_map = torch.cat(cur_feat_map, dim=1)
                        cur_feat_map = torch.mean(cur_feat_map, dim=1)
                        global_x.append(cur_feat_map)        
                else:
                    global_x = global_x_multi_depth[0]
        
        #########################################################################
        #        Merge the feature of multi-agents (with bandwidth cost)        #
        #########################################################################
        global_x_fused = []
        if self.trans_layer[-1] == -2:
            if self.feat_mode == 'fused':
                for c_layer, feat_map in enumerate(x_early):
                    # Mean
                    # global_x[c_layer] = global_x[c_layer] + feat_map
                    # global_x_fused.append(torch.cat([global_x[c_layer].unsqueeze(1), feat_map.unsqueeze(1)], dim=1).mean(dim=1))
                    # Max
                    global_x_fused.append(torch.cat([global_x[c_layer].unsqueeze(1), feat_map.unsqueeze(1)], dim=1).max(dim=1)[0])
        else:
            for c_layer, feat_map in enumerate(global_x):
                _, c, h, w = feat_map.shape
                feat_map = feat_map.view(b, num_agents, c, h, w)
                global_x[c_layer] = feat_map
            
            for c_layer, feat_map in enumerate(global_x):
                b, num_agents, c, h, w = feat_map.shape
                feat_map = feat_map.view(b*num_agents, c, h, w)
                global_x[c_layer] = feat_map

        if self.feat_mode == 'early':
            y = []
            for i in range(self.last_level - self.first_level):
                y.append(x_early[i].clone())
            # self.ida_up_early(y, 0, len(y))
            self.ida_up(y, 0, len(y))
        elif self.feat_mode == 'inter':
            y = []
            for i in range(self.last_level - self.first_level):
                y.append(global_x[i].clone())
            self.ida_up(y, 0, len(y))
        elif self.feat_mode == 'fused':
            y = []
            y_early = []
            y_fused = []
            for i in range(self.last_level - self.first_level):
                y.append(global_x[i].clone())
                y_early.append(x_early[i].clone())
                y_fused.append(global_x_fused[i].clone())
            self.ida_up(y, 0, len(y))
            self.ida_up(y_early, 0, len(y_early))
            # self.ida_up_early(y_early, 0, len(y_early)
            self.ida_up(y_fused, 0, len(y_fused))

        _, _, h, w = y[-1].shape
        # trans_mats_inverse = torch.inverse(trans_mats_list[0]).view(b, num_agents, 3, 3).contiguous().view(b*num_agents, 3, 3).contiguous()
        global_z = {}
        z = {}
        for head in self.heads:
            global_z[head] = self.__getattr__(head)(y[-1]) # (b*num_agent, 2, 112, 200)
            if self.feat_mode == 'fused':
                global_z[head+'_early'] = self.__getattr__(head)(y_early[-1])
                global_z[head+'_fused'] = self.__getattr__(head)(y_fused[-1])
            # global_z[head] = self.__getattr__(head)(y_fused[-1])
            # global_z[head] = self.__getattr__(head)(y_early[-1])
            # z[head] = kornia.warp_perspective(global_z[head], trans_mats_inverse, dsize=(h, w)) # (b*num_agents*num_agents, c, h, w)
        # return [z]

        # index = torch.where(global_z['hm'].sigmoid()>0.3)
        # index = torch.cat([x.unsqueeze(-1) for x in index], dim=-1)
        # N_objs = index.shape[0]
        # ax_i = int(math.sqrt(N_objs))
        # ax_j = math.ceil(N_objs/ax_i)
        # # Find the most objectiveness areas
        # for layer_i, (warp_images, offset, mask) in enumerate(zip(warp_images_list, offset_list, mask_list)):
        #     b, N, h, w = offset.shape
        #     offset_heatmap = offset.view(b, 2, N//2, h, w).cpu().numpy()
        #     # fig, axes = plt.subplots(ax_i, ax_j)
        #     for index_i in range(index.shape[0]):
        #         fig = plt.figure()
        #         i, j = index_i//ax_j, index_i%ax_j
        #         h_i, w_j = index[index_i][-2:]
        #         h_i = int(h_i * h / 192)
        #         w_j = int(w_j * w / 352)
        #         plt.plot(w_j, h_i, 'bo')
        #         plt.plot(offset_heatmap[0, 1, :, h_i, w_j], offset_heatmap[0, 0, :, h_i, w_j], 'r*')
        #         plt.imshow((warp_images[0].detach().cpu().numpy().transpose(1,2,0) * 255.).astype('uint8'))
        #         plt.xticks([])
        #         plt.yticks([])
        #         # axes[i,j].plot(w_j, h_i, 'bo')
        #         # axes[i,j].plot(offset_heatmap[0, 1, :, h_i, w_j], offset_heatmap[0, 0, :, h_i, w_j], 'r*')
        #         # axes[i,j].imshow((warp_images[0].detach().cpu().numpy().transpose(1,2,0) * 255.).astype('uint8'))
        #         # axes[i,j].set_xticks([])
        #         # axes[i,j].set_yticks([])
        #         plt.savefig('offset/{}_{}.png'.format(index_i, layer_i))
        #         plt.close()
        if self.depth_mode == 'Weighted':
            global_z['z'] = global_depth_weights_list[0]
            # global_z['z'] = torch.cat(global_depth_weights_list, dim=1)
        return [global_z]
    
    def JointCoord_forward(self, images, trans_mats, shift_mats, map_scale):
        # ------------------------ Image Feature Extraction ------------------- #
        b, num_agents, img_c, img_h, img_w = images.size()
        images = images.view(b*num_agents, img_c, img_h, img_w)
        
        if self.feat_mode in ['early', 'fused']:
            # print('Warp image')
            cur_trans_mats = trans_mats[0].view(b*num_agents, 3, 3)
            worldgrid2worldcoord_mat = torch.Tensor(np.array([[map_scale, 0, 0], [0, map_scale, 0], [0, 0, 1]])).to(cur_trans_mats.device)
            feat_zoom_mats = torch.Tensor(np.array(np.diag([4, 4, 1]), dtype=np.float32)).to(cur_trans_mats.device)
            cur_trans_mats = feat_zoom_mats @ shift_mats[0].view(b*num_agents, 3, 3) @ torch.inverse(cur_trans_mats @ worldgrid2worldcoord_mat).contiguous()
            images_warped = kornia.warp_perspective(images, cur_trans_mats, dsize=(self.feat_H*4, self.feat_W*4))
            # image = images[0].detach().cpu() * 255.
            # image = kornia.tensor_to_image(image.byte())
            # cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # cv2.imwrite('warp_img.png', image)
            
            x_early = self.base_early(images_warped)
            x_early = self.dla_up_early(x_early)

        if self.feat_mode in ['inter', 'fused']:
            # Encoder
            x = self.base(images)
            x = self.dla_up(x)  # list [(B, 64, 112, 200), (B, 128, 56, 100), (B, 256, 28, 50), (B, 512, 14, 25)] B = b * num_agents

            scale = 1/map_scale

            if self.depth_mode == 'Weighted':
                # get depth weights
                depth_weighted_feat_maps = [[] for _ in range(len(trans_mats))]
                depth_weights_list = []
                for c_layer, feat_map in enumerate(x):
                    depth_weights = eval('self.conv'+str(c_layer))(feat_map)    # (B, D, h, w)
                    depth_weights = F.softmax(depth_weights, dim=1)
                    depth_weighted_feat_map = depth_weights.unsqueeze(2) * feat_map.unsqueeze(1)   # (B, D, C, h, w)
                    depth_weighted_feat_map = depth_weighted_feat_map.unbind(1)
                    for depth_layer in range(len(trans_mats)):
                        depth_weighted_feat_maps[depth_layer].append(depth_weighted_feat_map[depth_layer])
                    depth_weights_list.append(depth_weights)
            else:
                trans_mats = [trans_mats[0]]
                depth_weighted_feat_maps = [x]
            
            global_x_multi_depth = []
            if self.depth_mode == 'Weighted':
                global_depth_weights_list = []
                for depth_layer, init_trans_mats in enumerate(trans_mats):
                    cur_x = depth_weighted_feat_maps[depth_layer]
                    global_x_HW = []
                    warp_images_list = []
                    init_trans_mats = init_trans_mats.view(b*num_agents, 3, 3)

                    for c_layer, feat_map in enumerate(cur_x):
                        _, c, h, w = feat_map.size()
                        shift_mats[c_layer] = shift_mats[c_layer].view(b*num_agents, 3, 3)

                        worldgrid2worldcoord_mat = torch.Tensor(np.array([[2**c_layer*scale, 0, 0], [0, 2**c_layer*scale, 0], [0, 0, 1]])).to(init_trans_mats.device)
                        img_trans_mats = shift_mats[c_layer].view(b*num_agents, 3, 3).contiguous() @ torch.inverse(init_trans_mats @ worldgrid2worldcoord_mat).contiguous()
                        warp_images = kornia.warp_perspective(images, img_trans_mats, dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                        warp_images_list.append(warp_images)

                        #########################################################################
                        #                              Hard Warping                             #
                        #########################################################################
                        # uav_i --> global coord
                        worldgrid2worldcoord_mat = torch.Tensor(np.array([[2**c_layer*map_scale, 0, 0], [0, 2**c_layer*map_scale, 0], [0, 0, 1]])).to(init_trans_mats.device)
                        feat_zoom_mats = torch.Tensor(np.array(np.diag([2**(c_layer+2), 2**(c_layer+2), 1]), dtype=np.float32)).to(init_trans_mats.device)
                        cur_trans_mats = shift_mats[c_layer].contiguous() @ torch.inverse(init_trans_mats @ worldgrid2worldcoord_mat).contiguous() @ feat_zoom_mats
                        global_feat = kornia.warp_perspective(feat_map, cur_trans_mats, dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))

                        # Two choice to classify z
                        if c_layer == 0:
                            # Choice1: supervise the sampe feature map warpped to multiple altitudes
                            if depth_layer == 0:
                                global_depth_weights = kornia.warp_perspective(depth_weights_list[c_layer], cur_trans_mats, dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                            # Choice2: supervise different feature map warpped to the corresponding altitudes
                            # global_depth_weights = kornia.warp_perspective(depth_weights_list[c_layer][:,depth_layer:depth_layer+1], cur_trans_mats, dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))                            
                            global_depth_weights_list.append(global_depth_weights)
                        
                        global_x_HW.append(global_feat)
                        
                    global_x_multi_depth.append(global_x_HW)
                
                global_x_HW = []
                for c_layer in range(len(x)):
                    cur_feat_map = [global_x_multi_depth[i][c_layer].unsqueeze(1) for i in range(len(global_x_multi_depth))]    # (B,D,C,H,W)
                    cur_feat_map = torch.cat(cur_feat_map, dim=1)
                    cur_feat_map = torch.mean(cur_feat_map, dim=1)
                    global_x_HW.append(cur_feat_map)
                
                global_x = []
                for c_layer, (feat_map, feat_map_HW) in enumerate(zip(x, global_x_HW)):
                    _, c, h, w = feat_map.size()
                    shift_mats[c_layer] = shift_mats[c_layer].view(b*num_agents, 3, 3)
                    # # 2. Get the value mat (trans feature to global coord)  # val_mat: (b, k_agents, q_agents, c, h, w)
                    if self.warp_mode in ['DW', 'DADW']:
                        #########################################################################
                        #                          Deformable Warping                           #
                        #########################################################################
                        # h_b, w_b = int(192/2**c_layer*scale), int(352/2**c_layer*scale)
                        # grid_y, grid_x = torch.meshgrid(torch.arange(0, h_b), torch.arange(0, w_b))
                        # grid = torch.stack((grid_x, grid_y), 2).type_as(feat_map)  # (w, h, 2)
                        # grid.requires_grad = False
                        worldgrid2worldcoord_mat = torch.Tensor(np.array([[2**c_layer*scale, 0, 0], [0, 2**c_layer*scale, 0], [0, 0, 1]])).to(init_trans_mats.device)
                        feat_zoom_mats = torch.Tensor(np.array(np.diag([2**(c_layer+2), 2**(c_layer+2), 1]), dtype=np.float32)).to(init_trans_mats.device)
                        cur_trans_mats = shift_mats[c_layer] @ torch.inverse(init_trans_mats @ worldgrid2worldcoord_mat).contiguous() @ feat_zoom_mats
                        # cur_trans_mats = shift_mats[c_layer] @ torch.inverse(cur_trans_mats @ worldgrid2worldcoord_mat).contiguous() @ feat_zoom_mats
                        
                        ### Residual 
                        # global_feat_HW = kornia.warp_perspective(feat_map, cur_trans_mats, dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                        global_feat_res = kornia.warp_perspective(feat_map, cur_trans_mats, mode='nearest', dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                        global_feat_res = eval('self.DC'+str(c_layer))(global_feat_res, return_offset=False)
                        global_feat = feat_map_HW + global_feat_res
                        
                        ### Bilinear
                        # global_feat = kornia.warp_perspective(feat_map, cur_trans_mats, dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                        # global_feat, offset, mask = eval('self.DC'+str(c_layer))(global_feat, return_offset=True)
                        
                        ### Nearest
                        # global_feat = kornia.warp_perspective(feat_map, cur_trans_mats, mode='nearest', dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                        # global_feat, offset, mask = eval('self.DC'+str(c_layer))(global_feat, return_offset=True)
                        
                        # print(c_layer, global_feat.shape, offset.shape, mask.shape) # 0, (1, 50, 192, 352); 0, (96, 176); 0, (48, 88); 0, (24, 44)
                        # offset_list.append(offset)
                        # mask_list.append(mask)

                        # b, N, h, w = offset.shape
                        # offset_heatmap = offset.view(b, 2, N//2, h, w).cpu().numpy()
                        # fig, axes = plt.subplots(3, 3)
                        # for i, h_i in enumerate(range(50//(2**c_layer), h, 50//(2**c_layer))):
                        #     for j, w_j in enumerate(range(100//(2**c_layer), w, 100//(2**c_layer))):
                        #         axes[i,j].plot(w_j, h_i, 'bo')
                        #         axes[i,j].plot(offset_heatmap[0, 1, :, h_i, w_j], offset_heatmap[0, 0, :, h_i, w_j], 'r*')
                        #         axes[i,j].imshow((warp_images[0].detach().cpu().numpy().transpose(1,2,0) * 255.).astype('uint8'))
                        #         axes[i,j].set_xticks([])
                        #         axes[i,j].set_yticks([])
                        # plt.savefig('offset.png')
                        # plt.close()
                        # import ipdb; ipdb.set_trace()
                    else:
                        global_feat = feat_map_HW
                    global_x.append(global_feat)
            else:
                for depth_layer, init_trans_mats in enumerate(trans_mats):
                    cur_x = depth_weighted_feat_maps[depth_layer]
                    global_x = []
                    offset_list = []
                    mask_list = []
                    warp_images_list = []
                    trans_mats_list = []
                    init_trans_mats = init_trans_mats.view(b*num_agents, 3, 3)
                    # cur_trans_mats = init_trans_mats.view(b*num_agents, 3, 3)
                    # fig, axes = plt.subplots(5, 4)

                    for c_layer, feat_map in enumerate(cur_x):
                        _, c, h, w = feat_map.size()
                        shift_mats[c_layer] = shift_mats[c_layer].view(b*num_agents, 3, 3)

                        worldgrid2worldcoord_mat = torch.Tensor(np.array([[2**c_layer*scale, 0, 0], [0, 2**c_layer*scale, 0], [0, 0, 1]])).to(init_trans_mats.device)
                        img_trans_mats = shift_mats[c_layer].view(b*num_agents, 3, 3).contiguous() @ torch.inverse(init_trans_mats @ worldgrid2worldcoord_mat).contiguous()
                        warp_images = kornia.warp_perspective(images, img_trans_mats, dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                        warp_images_list.append(warp_images)

                        # # 2. Get the value mat (trans feature to global coord)  # val_mat: (b, k_agents, q_agents, c, h, w)
                        if self.warp_mode in ['LW', 'RLW']:
                            #########################################################################
                            #                             Learnt Warping                            #
                            #########################################################################
                            worldgrid2worldcoord_mat = torch.Tensor(np.array([[2**c_layer*scale, 0, 0], [0, 2**c_layer*scale, 0], [0, 0, 1]])).to(init_trans_mats.device)
                            feat_zoom_mats = torch.Tensor(np.array(np.diag([2**(c_layer+2), 2**(c_layer+2), 1]), dtype=np.float32)).to(init_trans_mats.device)
                            cur_trans_mats = shift_mats[c_layer] @ torch.inverse(init_trans_mats @ worldgrid2worldcoord_mat).contiguous() @ feat_zoom_mats
                            global_feat_HW = kornia.warp_perspective(feat_map, cur_trans_mats, dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                            
                            if c_layer == 0:
                                global_feat = global_feat_HW
                            else:
                                flat_feat_map = feat_map.view(-1, c, h*w).contiguous()
                                global_feat_res = eval('self.fc'+str(c_layer))(flat_feat_map)
                                global_feat_res = global_feat_res.view(-1, c, int(self.feat_H/2**c_layer), int(self.feat_W/2**c_layer))
                                if self.warp_mode == 'LW':
                                    global_feat = global_feat_res
                                else:
                                    global_feat = global_feat_HW + global_feat_res
                        elif self.warp_mode in ['DW', 'DADW']:
                            #########################################################################
                            #                          Deformable Warping                           #
                            #########################################################################
                            # h_b, w_b = int(192/2**c_layer*scale), int(352/2**c_layer*scale)
                            # grid_y, grid_x = torch.meshgrid(torch.arange(0, h_b), torch.arange(0, w_b))
                            # grid = torch.stack((grid_x, grid_y), 2).type_as(feat_map)  # (w, h, 2)
                            # grid.requires_grad = False
                            worldgrid2worldcoord_mat = torch.Tensor(np.array([[2**c_layer*scale, 0, 0], [0, 2**c_layer*scale, 0], [0, 0, 1]])).to(init_trans_mats.device)
                            feat_zoom_mats = torch.Tensor(np.array(np.diag([2**(c_layer+2), 2**(c_layer+2), 1]), dtype=np.float32)).to(init_trans_mats.device)
                            cur_trans_mats = shift_mats[c_layer] @ torch.inverse(init_trans_mats @ worldgrid2worldcoord_mat).contiguous() @ feat_zoom_mats
                            # cur_trans_mats = shift_mats[c_layer] @ torch.inverse(cur_trans_mats @ worldgrid2worldcoord_mat).contiguous() @ feat_zoom_mats
                            
                            ### Residual 
                            global_feat_HW = kornia.warp_perspective(feat_map, cur_trans_mats, dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                            global_feat_res = kornia.warp_perspective(feat_map, cur_trans_mats, mode='nearest', dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                            global_feat_res = eval('self.DC'+str(c_layer))(global_feat_res, return_offset=False)
                            global_feat = global_feat_HW + global_feat_res
                            
                            ### Bilinear
                            # global_feat = kornia.warp_perspective(feat_map, cur_trans_mats, dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                            # global_feat, offset, mask = eval('self.DC'+str(c_layer))(global_feat, return_offset=True)
                            
                            ### Nearest
                            # global_feat = kornia.warp_perspective(feat_map, cur_trans_mats, mode='nearest', dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                            # global_feat, offset, mask = eval('self.DC'+str(c_layer))(global_feat, return_offset=True)
                            
                            # print(c_layer, global_feat.shape, offset.shape, mask.shape) # 0, (1, 50, 192, 352); 0, (96, 176); 0, (48, 88); 0, (24, 44)
                            # offset_list.append(offset)
                            # mask_list.append(mask)

                            # b, N, h, w = offset.shape
                            # offset_heatmap = offset.view(b, 2, N//2, h, w).cpu().numpy()
                            # fig, axes = plt.subplots(3, 3)
                            # for i, h_i in enumerate(range(50//(2**c_layer), h, 50//(2**c_layer))):
                            #     for j, w_j in enumerate(range(100//(2**c_layer), w, 100//(2**c_layer))):
                            #         axes[i,j].plot(w_j, h_i, 'bo')
                            #         axes[i,j].plot(offset_heatmap[0, 1, :, h_i, w_j], offset_heatmap[0, 0, :, h_i, w_j], 'r*')
                            #         axes[i,j].imshow((warp_images[0].detach().cpu().numpy().transpose(1,2,0) * 255.).astype('uint8'))
                            #         axes[i,j].set_xticks([])
                            #         axes[i,j].set_yticks([])
                            # plt.savefig('offset.png')
                            # plt.close()
                            # import ipdb; ipdb.set_trace()
                        elif self.warp_mode == 'SW':
                            worldgrid2worldcoord_mat = torch.Tensor(np.array([[2**c_layer*scale, 0, 0], [0, 2**c_layer*scale, 0], [0, 0, 1]])).to(init_trans_mats.device)
                            feat_zoom_mats = torch.Tensor(np.array(np.diag([2**(c_layer+2), 2**(c_layer+2), 1]), dtype=np.float32)).to(init_trans_mats.device)
                            cur_trans_mats = shift_mats[c_layer] @ torch.inverse(init_trans_mats @ worldgrid2worldcoord_mat).contiguous() @ feat_zoom_mats
                            
                            ### Residual
                            # global_feat_HW = kornia.warp_perspective(feat_map, cur_trans_mats, dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                            # global_feat_res = kornia.warp_perspective(feat_map, cur_trans_mats, mode='nearest', dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                            # global_feat_res = eval('self.saliency'+str(c_layer))(global_feat_res)
                            # global_feat = global_feat_HW + global_feat_res

                            ### Bilinear
                            global_feat = kornia.warp_perspective(feat_map, cur_trans_mats, dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                            if c_layer == 0:
                                global_feat = eval('self.saliency'+str(c_layer))(global_feat, images_warped)
                            
                            ### Nearest
                            # global_feat = kornia.warp_perspective(feat_map, cur_trans_mats, mode='nearest', dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                            # global_feat = eval('self.saliency'+str(c_layer))(global_feat)
                        elif self.warp_mode == 'SWU':
                            worldgrid2worldcoord_mat = torch.Tensor(np.array([[2**c_layer*scale, 0, 0], [0, 2**c_layer*scale, 0], [0, 0, 1]])).to(init_trans_mats.device)
                            feat_zoom_mats = torch.Tensor(np.array(np.diag([2**(c_layer+2), 2**(c_layer+2), 1]), dtype=np.float32)).to(init_trans_mats.device)
                            cur_trans_mats = shift_mats[c_layer] @ torch.inverse(init_trans_mats @ worldgrid2worldcoord_mat).contiguous() @ feat_zoom_mats

                            ### Bilinear
                            if c_layer == 0:
                                feat_map = eval('self.saliency'+str(c_layer))(feat_map, images_warped)
                            global_feat = kornia.warp_perspective(feat_map, cur_trans_mats, dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                        else:
                            #########################################################################
                            #                              Hard Warping                             #
                            #########################################################################
                            # uav_i --> global coord
                            # worldgrid2worldcoord_mat = torch.Tensor(np.array([[500/(w*scale), 0, -200], [0, 500/(h*scale), -250], [0, 0, 1]])).to(init_trans_mats.device)
                            # worldgrid2worldcoord_mat = torch.Tensor(np.array([[1/(w*scale), 0, 0], [0, 1/(h*scale), 0], [0, 0, 1]])).to(init_trans_mats.device)
                            # worldgrid2worldcoord_mat = torch.Tensor(np.array([[1/(2**(c_layer+4)*scale), 0, 0], [0, 1/(2**(c_layer+4)*scale), 0], [0, 0, 1]])).to(init_trans_mats.device)
                            # worldgrid2worldcoord_mat = torch.Tensor(np.array([[1/(500/2**c_layer*scale), 0, 0], [0, 1/(500/2**c_layer*scale), 0], [0, 0, 1]])).to(init_trans_mats.device)
                            worldgrid2worldcoord_mat = torch.Tensor(np.array([[2**c_layer*map_scale, 0, 0], [0, 2**c_layer*map_scale, 0], [0, 0, 1]])).to(init_trans_mats.device)
                            feat_zoom_mats = torch.Tensor(np.array(np.diag([2**(c_layer+2), 2**(c_layer+2), 1]), dtype=np.float32)).to(init_trans_mats.device)
                            cur_trans_mats = shift_mats[c_layer].contiguous() @ torch.inverse(init_trans_mats @ worldgrid2worldcoord_mat).contiguous() @ feat_zoom_mats
                            global_feat = kornia.warp_perspective(feat_map, cur_trans_mats, dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                            # global_feat = kornia.warp_perspective(feat_map, cur_trans_mats, mode='nearest', dsize=(int(192/2**c_layer*scale), int(352/2**c_layer*scale)))
                            # global_feat = kornia.warp_perspective(feat_map, cur_trans_mats, dsize=(2**(c_layer+4)*scale, 2**(c_layer+4)*scale)) # (b*num_agents, c, h, w)
                            # global_feat = kornia.warp_perspective(feat_map, cur_trans_mats, dsize=(h*scale, w*scale)) # (b*num_agents, c, h, w)
                            # global_feat = kornia.resize(global_feat, size=(h*4, w*4))

                        global_x.append(global_feat)
                        
                        # val_mats = global_feat.max(dim=-3)[0].detach().cpu().numpy()
                        # for j in range(min(5, val_mats.shape[0])):
                        #     axes[j, c_layer].imshow((val_mats[j]*255.).astype('uint8'))
                        #     axes[j, c_layer].set_xticks([])
                        #     axes[j, c_layer].set_yticks([])

                    # plt.savefig('featmap.png')
                    # plt.close()
                    global_x_multi_depth.append(global_x)
            
                if len(global_x_multi_depth) > 1:
                    global_x = []
                    for c_layer in range(len(x)):
                        cur_feat_map = [global_x_multi_depth[i][c_layer].unsqueeze(1) for i in range(len(global_x_multi_depth))]    # (B,D,C,H,W)
                        cur_feat_map = torch.cat(cur_feat_map, dim=1)
                        cur_feat_map = torch.mean(cur_feat_map, dim=1)
                        global_x.append(cur_feat_map)        
                else:
                    global_x = global_x_multi_depth[0]
        
        #########################################################################
        #        Merge the feature of multi-agents (with bandwidth cost)        #
        #########################################################################
        global_x_fused = []
        if self.trans_layer[-1] == -2:
            if self.feat_mode == 'fused':
                for c_layer, feat_map in enumerate(x_early):
                    # Mean
                    # global_x[c_layer] = global_x[c_layer] + feat_map
                    # global_x_fused.append(torch.cat([global_x[c_layer].unsqueeze(1), feat_map.unsqueeze(1)], dim=1).mean(dim=1))
                    # Max
                    global_x_fused.append(torch.cat([global_x[c_layer].unsqueeze(1), feat_map.unsqueeze(1)], dim=1).max(dim=1)[0])
        else:
            for c_layer, feat_map in enumerate(global_x):
                _, c, h, w = feat_map.shape
                feat_map = feat_map.view(b, num_agents, c, h, w)
                global_x[c_layer] = feat_map
            
            for c_layer, feat_map in enumerate(global_x):
                b, num_agents, c, h, w = feat_map.shape
                feat_map = feat_map.view(b*num_agents, c, h, w)
                global_x[c_layer] = feat_map

        # --------------------------- UAV Detection --------------------------- #
        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up_i(y, 0, len(y))

        z = {}
        for head in self.heads:
            if '_i' in head:
                z[head] = self.__getattr__(head)(y[-1]) # (b*num_images, 2, 112, 200)
        
        # ---------------------------- BEV Detection -------------------------- #
        if self.feat_mode == 'early':
            global_y = []
            for i in range(self.last_level - self.first_level):
                global_y.append(x_early[i].clone())
            # self.ida_up_early(y, 0, len(y))
            self.ida_up(global_y, 0, len(global_y))
        elif self.feat_mode == 'inter':
            global_y = []
            for i in range(self.last_level - self.first_level):
                global_y.append(global_x[i].clone())
            self.ida_up(global_y, 0, len(global_y))
        elif self.feat_mode == 'fused':
            global_y = []
            global_y_early = []
            global_y_fused = []
            for i in range(self.last_level - self.first_level):
                global_y.append(global_x[i].clone())
                global_y_early.append(x_early[i].clone())
                global_y_fused.append(global_x_fused[i].clone())
            self.ida_up(global_y, 0, len(global_y))
            self.ida_up(global_y_early, 0, len(global_y_early))
            # self.ida_up_early(y_early, 0, len(y_early)
            self.ida_up(global_y_fused, 0, len(global_y_fused))

        global_z = {}
        for head in self.heads:
            if '_i' in head:
                continue
            global_z[head] = self.__getattr__(head)(global_y[-1]) # (b*num_agent, 2, 112, 200)
            if self.feat_mode == 'fused':
                global_z[head+'_early'] = self.__getattr__(head)(global_y_early[-1])
                global_z[head+'_fused'] = self.__getattr__(head)(global_y_fused[-1])
        global_z.update(z)
        if self.depth_mode == 'Weighted':
            global_z['z'] = global_depth_weights_list[0]
            # global_z['z'] = torch.cat(global_depth_weights_list, dim=1)
        return [global_z]
    
    def forward(self, images, trans_mats, shift_mats, map_scale=1.0):
        if self.coord == 'Global':
            return self.GlobalCoord_forward(images, trans_mats, shift_mats, map_scale)
        elif self.coord == 'Joint':
            return self.JointCoord_forward(images, trans_mats, shift_mats, map_scale)
        else:
            return self.LocalCoord_forward(images, trans_mats)
        

def get_pose_net(num_layers, heads, head_conv=256, down_ratio=4, message_mode='NO_MESSAGE_NOWARP', trans_layer=[3], coord='Local', warp_mode='HW', depth_mode='Unique', feat_mode='inter', feat_shape=[192, 352]):
    model = DLASeg('dla{}'.format(num_layers), heads,
                    pretrained=True,
                    down_ratio=down_ratio,
                    final_kernel=1,
                    last_level=5,
                    head_conv=head_conv,
                    message_mode=message_mode,
                    trans_layer=trans_layer,
                    coord=coord,
                    warp_mode=warp_mode,
                    depth_mode=depth_mode,
                    feat_mode=feat_mode,
                    feat_shape=feat_shape)
    return model

