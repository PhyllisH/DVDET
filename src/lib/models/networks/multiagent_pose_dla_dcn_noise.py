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
import time
import os
import math
import logging
import ipdb
from kornia.geometry.epipolar.projection import depth
from kornia.geometry.transform.imgwarp import warp_affine
from matplotlib.colors import hsv_to_rgb
from matplotlib.style import context
import numpy as np
from os.path import join
import math

import torch
from torch import nn, sigmoid
from torch._C import device
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import kornia
import matplotlib.pyplot as plt
import cv2

from .DCNv2.dcn_v2 import DCN, CF_DCN, EGOAWARE_DCN
import sys

from lib.models.networks.Compressor import compressor
sys.path.append('/GPFS/data/yhu/code/CoDet/src/lib/models/networks')
from .convolutional_rnn import Conv2dGRU
from .Compressor.compressor import ScaleHyperprior

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

class EGOAWAREDeformConv(nn.Module):
    def __init__(self, chi, cho, k_size=3, dilation=1, mode='EGOAWARE_DCN'):
        super(EGOAWAREDeformConv, self).__init__()
        self.mode = mode
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = eval(mode)(chi, cho, kernel_size=(k_size,k_size), stride=1, padding=(k_size-1)//2, dilation=dilation, deformable_groups=1)

    def forward(self, ego_x, x):
        x, offset, mask = self.conv(ego_x, x)
        x = self.actf(x)
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
            layers[i] = node(layers[i] + layers[i - 1])
            # if layers[i].shape == layers[i - 1].shape:
            #     layers[i] = node(layers[i] + layers[i - 1])
            # else:
            #     layers[i] = node(layers[i])



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


class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation
        
        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, input):
        """Forward function.
        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape and reshape back after sparsemax
        original_size = input.size()
        input = input.view(-1, input.size(self.dim))
        
        dim = 1
        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.range(start=1, end=number_of_logits, device=input.device).view(1, -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        output = self.output.view(original_size)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input

class km_generator(nn.Module):
    def __init__(self, out_size=128, input_feat_h=14, input_feat_w=25):
        super(km_generator, self).__init__()
        # self.n_feat = int(256 * (input_feat_h//4 + 1) * (input_feat_w//4 + 1))
        self.n_feat = int(256 * input_feat_h * input_feat_w)
        self.fc = nn.Sequential(
            nn.Linear(self.n_feat, 256), #            
            nn.ReLU(inplace=True),
            nn.Linear(256, 128), #             
            nn.ReLU(inplace=True),
            nn.Linear(128, out_size)) #            

    def forward(self, feat_map):
        outputs = self.fc(feat_map.view(-1, self.n_feat))
        return outputs
class policy_net4(nn.Module):
    def __init__(self, base_name, pretrained, down_ratio, last_level):
        super(policy_net4, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        # self.base = globals()[base_name](pretrained=pretrained)
        # channels = self.base.channels
        # scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        # self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)

        # Encoder
        # down 1 
        self.conv1 = conv2DBatchNormRelu(512, 512, k_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(512, 256, k_size=3, stride=1, padding=1)
        self.conv3 = conv2DBatchNormRelu(256, 256, k_size=3, stride=2, padding=1)

        # down 2
        self.conv4 = conv2DBatchNormRelu(256, 256, k_size=3, stride=1, padding=1)
        self.conv5 = conv2DBatchNormRelu(256, 256, k_size=3, stride=2, padding=1)

    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        outputs = self.conv5(outputs)
        return outputs
class MIMOGeneralDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, query_size, key_size, warp_flag=True, attn_dropout=0.1):
        super().__init__()
        self.sparsemax = Sparsemax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.linear = nn.Linear(query_size, key_size)
        self.warp_flag = warp_flag
        print('Msg size: ',query_size,'  Key size: ', key_size)

    def forward(self, qu, k, v, sparse=True):
        # qu (b, q_agents, query_size)
        # k (b, k_agents, key_size)
        # v (b, k_agents, q_agents, c, h, w)
        query = self.linear(qu)  # (b, q_agents, key_size)

        # normalization
        # query_norm = query.norm(p=2,dim=2).unsqueeze(2).expand_as(query)
        # query = query.div(query_norm + 1e-9)

        # k_norm = k.norm(p=2,dim=2).unsqueeze(2).expand_as(k)
        # k = k.div(k_norm + 1e-9)
        # generate the
        attn_orig = torch.bmm(k, query.transpose(2, 1))  # (b, k_agents, q_agents)  column: differnt keys and the same query

        # scaling [not sure]
        # scaling = torch.sqrt(torch.tensor(k.shape[2],dtype=torch.float32)).cuda()
        # attn_orig = attn_orig/ scaling # (b,5,5)  column: differnt keys and the same query

        attn_orig_softmax = self.softmax(attn_orig)  # (b, k_agents, q_agents)
        # attn_orig_softmax = self.sparsemax(attn_orig)

        attn_shape = attn_orig_softmax.shape
        bats, key_num, query_num = attn_shape[0], attn_shape[1], attn_shape[2]
        attn_orig_softmax_exp = attn_orig_softmax.view(bats, key_num, query_num, 1, 1, 1)

        if self.warp_flag:
            v_exp = v
        else:
            v_exp = torch.unsqueeze(v, 2)
            v_exp = v_exp.expand(-1, -1, query_num, -1, -1, -1)

        output = attn_orig_softmax_exp * v_exp  # (b, k_agents, q_agents, c, h, w)
        output_sum = output.sum(1)  # (b, q_agents, c, h, w)

        return output_sum, attn_orig_softmax


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3):
        super().__init__()

        nb_filter = [32, 64, 128]

        self.pool = nn.MaxPool2d(4, 4)
        self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])

        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_2 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))

        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, self.up(x1_1)], 1))

        output = self.final(x0_2)
        return output
class DLASeg(nn.Module):
    def __init__(self, base_name, heads, pretrained, down_ratio, final_kernel,
                 last_level, head_conv, out_channel=0, message_mode='NO_MESSAGE', trans_layer=[3], coord='Local', warp_mode='HW', depth_mode='Unique',
                 feat_mode='inter', feat_shape=[192, 352], round=1, compress_flag=False, comm_thre=0.1, sigma=0):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.warp_mode = warp_mode
        self.depth_mode = depth_mode
        self.feat_mode = feat_mode
        self.sigma = sigma

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
        self.round = round

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
        self.message_mode = message_mode.split('+')[0]
        if self.message_mode in ['When2com']:
            self.query_key_net = policy_net4(base_name, pretrained, down_ratio, last_level)
            self.key_net = km_generator(out_size=self.key_size, input_feat_h=self.feat_H//32, input_feat_w=self.feat_W//32)
            self.query_net = km_generator(out_size=self.query_size, input_feat_h=self.feat_H//32, input_feat_w=self.feat_W//32)
            self.attention_net = MIMOGeneralDotProductAttention(self.query_size, self.key_size)
        elif self.message_mode in ['V2V']:
            self.gnn_iter_num = 3
            for c_layer in self.trans_layer:
                if c_layer >= 0:
                    # convgru = Conv2dGRU(in_channels=128*2**c_layer+4,
                    convgru = Conv2dGRU(in_channels=128*2**c_layer,
                                            out_channels=64*2**c_layer,
                                            kernel_size=3,
                                            num_layers=1,
                                            bidirectional=False,
                                            dilation=1,
                                            stride=1)
                    self.__setattr__('convgru'+str(c_layer), convgru)
        elif self.message_mode in ['Pointwise']:
            for c_layer in self.trans_layer:
                if c_layer >= 0:
                    # weight_net = nn.Conv2d(128*2**c_layer+4, 1, kernel_size=1, stride=1, padding=0)
                    weight_net = nn.Conv2d(128*2**c_layer, 1, kernel_size=1, stride=1, padding=0)
                    self.__setattr__('weight_net'+str(c_layer), weight_net)
        elif self.message_mode in ['Concat']:
            for c_layer in self.trans_layer:
                if c_layer >= 0:
                    # weight_net = nn.Conv2d(128*2**c_layer+4, 1, kernel_size=1, stride=1, padding=0)
                    weight_net = nn.Conv2d(128*2**c_layer, 64*2**c_layer, kernel_size=1, stride=1, padding=0)
                    self.__setattr__('weight_net'+str(c_layer), weight_net)
        elif 'QualityMapTransformer' in self.message_mode:
            dropout = 0
            nhead = 8
            for c_layer in self.trans_layer:
                if c_layer >= 0:
                    d_model = 64*2**c_layer
                    cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
                    # Implementation of Feedforward model
                    linear1 = nn.Linear(d_model, d_model)
                    linear2 = nn.Linear(d_model, d_model)

                    norm1 = nn.LayerNorm(d_model)
                    norm2 = nn.LayerNorm(d_model)
                    dropout0 = nn.Dropout(dropout)
                    dropout1 = nn.Dropout(dropout)
                    dropout2 = nn.Dropout(dropout)
                    self.__setattr__('cross_attn'+str(c_layer), cross_attn)
                    self.__setattr__('linear1_'+str(c_layer), linear1)
                    self.__setattr__('linear2_'+str(c_layer), linear2)
                    self.__setattr__('norm1_'+str(c_layer), norm1)
                    self.__setattr__('norm2_'+str(c_layer), norm2)
                    self.__setattr__('dropout0_'+str(c_layer), dropout0)
                    self.__setattr__('dropout1_'+str(c_layer), dropout1)
                    self.__setattr__('dropout2_'+str(c_layer), dropout2)

                    # Gaussian smoothing
                    kernel_size = 5
                    for c_sigma in [1, 0.8, 0.5]:
                        gaussian_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
                        self.init_gaussian_filter(gaussian_filter, kernel_size, c_sigma)
                        gaussian_filter.requires_grad = False
                        self.__setattr__('gaussian_filter'+str(int(10*c_sigma)), gaussian_filter)

                    if c_layer > 0:
                        self.__setattr__('pool_net'+str(c_layer), nn.MaxPool2d(2**c_layer, stride=2**c_layer))
                        
        d_model = 64*2**self.trans_layer[0]
        if compress_flag:
            self.compressor = ScaleHyperprior(N=d_model//4, M=d_model)
        
        self.comm_thre = comm_thre
        self.compress_flag = compress_flag
        self.vis = False #True #
    
    def gen_gaussian_kernel(self, k_size=5, sigma=1):
        center = k_size // 2
        x, y = np.mgrid[0 - center : k_size - center, 0 - center : k_size - center]
        g = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
        return g
    
    def init_gaussian_filter(self, gaussian_filter, k_size=5, sigma=1):
        gaussian_kernel = self.gen_gaussian_kernel(k_size, sigma)
        gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        gaussian_filter.bias.data.zero_()

    def add_coord_map(self, input, normalized=True):
        if len(input.shape) == 5:
            b, num_agents, c, h, w = input.shape
        elif len(input.shape) == 6:
            b, num_agents, num_agents, c, h, w = input.shape

        if normalized:
            h_i = torch.arange(h).unsqueeze(-1).expand(-1,w).unsqueeze(0) / h
            w_i = torch.arange(w).unsqueeze(0).expand(h,-1).unsqueeze(0) / w - 0.5
        else:
            h_i = torch.arange(h).unsqueeze(-1).expand(-1,w).unsqueeze(0)
            w_i = torch.arange(w).unsqueeze(0).expand(h,-1).unsqueeze(0)
        
        im_i = torch.cat([h_i, w_i], dim=0).to(input.device) # (2, h, w)
        if len(input.shape) == 5:
            im_i = im_i.unsqueeze(0).expand(num_agents,-1,-1,-1).unsqueeze(0).expand(b,-1,-1,-1,-1)
        elif len(input.shape) == 6:
            im_i = im_i.unsqueeze(0).expand(num_agents,-1,-1,-1).unsqueeze(0).expand(num_agents,-1,-1,-1,-1).unsqueeze(0).expand(b,-1,-1,-1,-1,-1)
        
        input_with_index = torch.cat([input, im_i], dim=-3)  # (b, num_agents, c+2, h, w)
        return input_with_index
    
    def add_pe_map(self, x, normalized=True):
        # scale = 2 * math.pi
        temperature = 10000
        num_pos_feats = x.shape[-3] // 2

        mask = torch.zeros([x.shape[-2], x.shape[-1]], dtype=torch.bool, device=x.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(0, dtype=torch.float32)
        x_embed = not_mask.cumsum(1, dtype=torch.float32)

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)

        if len(x.shape) == 5:
            x_withpos = x + pos[None,None,:,:,:]
        elif len(x.shape) == 6:
            x_withpos = x + pos[None,None,None,:,:,:]
        return x_withpos
    
    def mul_quality_map(self, x):
        mask = torch.zeros([x.shape[-2], x.shape[-1]], dtype=torch.bool, device=x.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(0, dtype=torch.float32)
        x_embed = not_mask.cumsum(1, dtype=torch.float32)

        x_c = x.shape[-1] // 2
        y_c = x.shape[-2]

        dis = (((x_embed[:, :] - x_c)/x_c) ** 2 + ((y_embed[:, :] - y_c)/y_c) ** 2).sqrt()
        prob = (1 - dis.sigmoid()) * 2 - 0.2

        if len(x.shape) == 5:
            b, num_agents, c, _, _ = x.shape
            prob = prob[None,None,None,:,:].expand(b,num_agents,-1,-1,-1)
        elif len(x.shape) == 6:
            b, k_agents, q_agents, c, _, _ = x.shape
            prob = prob[None,None,None,None,:,:].expand(b,k_agents,q_agents,-1,-1,-1)
        return prob

    def get_colla_feats(self, x, shift_mats, trans_layer, with_pos=False, with_qual=False):
        val_feats = []
        for c_layer in trans_layer:
            feat_map = x[c_layer]

            if with_qual:
                qual_map = self.mul_quality_map(feat_map) # (b, num_agents, 1, h, w)
                # feat_map_mask = feat_map.max(dim=-3)[0]
                # ones = torch.ones_like(feat_map_mask).to(feat_map.device)
                # zeros = torch.zeros_like(feat_map_mask).to(feat_map.device)
                # feat_map_mask = torch.where(feat_map_mask>0, ones, zeros).unsqueeze(2) # (b, num_agents, 1, h, w)
                # qual_map = qual_map * feat_map_mask

            if with_pos:
                feat_map = self.add_pe_map(feat_map)
            
            b, num_agents, c, h, w = feat_map.size()
            ori_shift_mats = shift_mats[c_layer]
            noisy_shift_mats = shift_mats[-1]

            # Get the value mat (shift global feature to current agent coord) # val_feat: (b, k_agents, q_agents, c, h, w)
            shift_mats_k = ori_shift_mats.unsqueeze(1).expand(-1, num_agents, -1, -1, -1).contiguous()
            noisy_shift_mats_k = noisy_shift_mats.unsqueeze(1).expand(-1, num_agents, -1, -1, -1).contiguous()
            for agent_i in range(num_agents):
                noisy_shift_mats_k[:,agent_i, agent_i] = shift_mats_k[:,agent_i, agent_i]
            shift_mats_k = shift_mats_k.view(b*num_agents*num_agents, 3, 3).contiguous()  #  (b, k_agents, q_agents, 3, 3)
            noisy_shift_mats_k = noisy_shift_mats_k.view(b*num_agents*num_agents, 3, 3).contiguous()  #  (b, k_agents, q_agents, 3, 3)
            shift_mats_q = torch.inverse(ori_shift_mats.unsqueeze(2).expand(-1, -1, num_agents, -1, -1).contiguous()).contiguous().view(b*num_agents*num_agents, 3, 3).contiguous()   # (b, k_agents, q_agents, 3, 3)
            # cur_shift_mats = shift_mats_k @ shift_mats_q    # (b*k_agents*q_agents, 3, 3)
            cur_shift_mats = noisy_shift_mats_k @ shift_mats_q    # (b*k_agents*q_agents, 3, 3)

            global_feat = feat_map.view(b, num_agents, c, h, w).contiguous().unsqueeze(2).expand(-1, -1, num_agents, -1, -1, -1)
            global_feat = global_feat.contiguous().view(b*num_agents*num_agents, c, h, w).contiguous()
            
            val_feat = kornia.warp_perspective(global_feat, cur_shift_mats, dsize=(h, w)) # (b*num_agents*num_agents, c, h, w)
            val_feat = val_feat.view(b, num_agents, num_agents, c, h, w).contiguous() # (b, k_agents, q_agents, c, h, w)
            
            if with_qual:
                global_qual_map = qual_map.view(b, num_agents, 1, h, w).contiguous().unsqueeze(2).expand(-1, -1, num_agents, -1, -1, -1)
                global_qual_map = global_qual_map.contiguous().view(b*num_agents*num_agents, 1, h, w).contiguous()
                
                val_qual_map = kornia.warp_perspective(global_qual_map, cur_shift_mats, dsize=(h, w)) # (b*num_agents*num_agents, c, h, w)
                val_qual_map = val_qual_map.view(b, num_agents, num_agents, 1, h, w).contiguous() # (b, k_agents, q_agents, c, h, w)

                for i in range(num_agents):
                    q_qual_map = val_qual_map[:,i:i+1,i]
                    k_qual_maps = val_qual_map[:,:,i]
                    rel_qual_maps = q_qual_map + k_qual_maps
                    val_feat[:,:,i] = (k_qual_maps / (rel_qual_maps + 1e-6)) * val_feat[:,:,i]

            val_feats.append(val_feat)
        return val_feats

    def COLLA_MESSAGE(self, x, shift_mats):
        val_feats = self.get_colla_feats(x, shift_mats, self.trans_layer)

        weight_mats = []
        for i, c_layer in enumerate(self.trans_layer):
            feat_map = x[c_layer]
            val_feat = val_feats[i]

            # if self.message_mode in ['Pointwise']:
            #     feat_map = self.add_pe_map(feat_map)
            #     val_feat = self.add_pe_map(val_feat)
                # feat_map = self.add_coord_map(feat_map)
                # val_feat = self.add_coord_map(val_feat)
                # feat_map = self.add_coord_map(feat_map, normalized=False)
                # val_feat = self.add_coord_map(val_feat, normalized=False)

            b, num_agents, c, h, w = feat_map.size()
            query_feat = feat_map.unsqueeze(1).expand(-1, num_agents, -1, -1, -1, -1).contiguous()   # (b*num_agents, c, h, w) --> (b, k_agents, q_agents, c, h, w)

            if self.message_mode == 'Mean':
                # Mean
                feat_map_mask = torch.where(val_feat>0, torch.ones_like(val_feat).to(val_feat.device), torch.zeros_like(val_feat).to(val_feat.device))
                feat_fuse = val_feat.sum(dim=1) / (feat_map_mask.sum(dim=1)+1e-6)
            elif self.message_mode == 'Max':
                # Max
                feat_fuse = val_feat.max(dim=1)[0] # (b, q_agents, c, h, w)
            elif self.message_mode == 'Pointwise':
                # Pointwise: Attention Mode 1: Relu(MLP([q, k]))
                weight_mat = torch.cat([query_feat, val_feat], dim=3).view(b*num_agents*num_agents, c*2, h, w)
                weight_mat = F.relu(eval('self.weight_net'+str(c_layer))(weight_mat))    # (b*k_agents*q_agents, c, h, w)
                weight_mat = weight_mat.view(b, num_agents, num_agents, 1, h, w).softmax(dim=1)
                feat_fuse = (weight_mat * val_feat).sum(dim=1)    # (b*num_agents, c, h, w)
                weight_mats.append(weight_mat)
            elif self.message_mode == 'Concat':
                mean_feat = val_feat.sum(dim=1)# (b, num_agents, c, h, w)
                mean_feat = (mean_feat - feat_map)/4.0
                cat_feat = torch.cat([feat_map, mean_feat], dim=-3).reshape(b*num_agents, 2*c, h, w).contiguous() # (b*num_agents, 1, 2*c, h, w)
                feat_fuse = eval('self.weight_net'+str(c_layer))(cat_feat)
                feat_fuse = feat_fuse.view(b, num_agents, c, h, w)

            # 4. Return fused feat
            post_commu_feats = feat_fuse * 0.5 + feat_map * 0.5

            # if self.message_mode in ['Pointwise']:
            #     x[c_layer] = post_commu_feats[:,:,:-2,:,:]
            # else:
            x[c_layer] = post_commu_feats
        return x, weight_mats, val_feats
    
    def ATTEN_MESSAGE(self, x, shift_mats, kernel_size=3, stride=2):
        def get_padded_feat(x, kernel_size=3, stride=1):
            b, c, h, w = x.shape
            padding_w = torch.zeros([b, c, h, (kernel_size-1)*stride//2], dtype=torch.float32, device=x.device)
            padding_h = torch.zeros([b, c, (kernel_size-1)*stride//2, w+(kernel_size-1)*stride], dtype=torch.float32, device=x.device)
            x = torch.cat([padding_w, x, padding_w], dim=-1)    # (b, num_agents, c, h, w+kernel_size-1)
            x = torch.cat([padding_h, x, padding_h], dim=-2)    # (b, num_agents, c, h+kernel_size-1, w+kernel_size-1)
            return x

        def get_local_feat(x, kernel_size=3, stride=1):
            b, c, h, w = x.shape
            x = get_padded_feat(x, kernel_size, stride)
            feat_maps = []
            for i in range(kernel_size):
                for j in range(kernel_size):
                    feat_map = x[:,:,i*stride:h+i*stride,j*stride:w+j*stride]
                    feat_maps.append(feat_map.unsqueeze(1))
            
            feat_maps = torch.cat(feat_maps, dim=1) # (b, kernel**2, c, h, w)
            return feat_maps

        val_feats = self.get_colla_feats(x, shift_mats, self.trans_layer, with_pos=True) # (b, k_agents, q_agents, c, h, w)

        weight_mats = []
        for i, c_layer in enumerate(self.trans_layer):
            feat_map = x[c_layer]
            val_feat = val_feats[i]

            feat_map = self.add_pe_map(feat_map)

            b, num_agents, c, h, w = feat_map.size()
            
            query_feat = eval('self.query_net'+str(c_layer))(feat_map.view(b*num_agents, c, h, w)).view(b, num_agents, self.query_size, h, w).contiguous()
            query_feat = query_feat.unsqueeze(1).expand(-1, num_agents, -1, -1, -1, -1).contiguous()   # (b*num_agents, c, h, w) --> (b, k_agents, q_agents, c, h, w)
            query_feat = query_feat.unsqueeze(3).expand(-1, -1, -1, kernel_size**2, -1, -1, -1).contiguous()   # (b, k_agents, q_agents, c, h, w)--> (b, k_agents, q_agents, kernel_size, c, h, w)
            query_feat = query_feat.view(b*num_agents*num_agents*kernel_size**2, self.query_size, h, w).contiguous()
            
            val_feat = val_feat.view(b*num_agents*num_agents, c, h, w).contiguous()
            key_feat = eval('self.key_net'+str(c_layer))(val_feat)
            key_feat = get_local_feat(key_feat.view(b*num_agents*num_agents, self.key_size, h, w).contiguous(), kernel_size=kernel_size, stride=stride)
            key_feat = key_feat.view(b*num_agents*num_agents*kernel_size**2, self.key_size, h, w).contiguous()

            weight_mat = (query_feat * key_feat).sum(dim=-3)
            weight_mat = weight_mat.view(b, num_agents, num_agents, kernel_size**2, h, w).contiguous()
            weight_mat = weight_mat.transpose(3,2).contiguous().view(b, num_agents*kernel_size**2, num_agents, h, w).contiguous()
            weight_mat = weight_mat.softmax(dim=1)  # (b, num_agents*kernel_size**2, num_agents, h, w)
            weight_mat = weight_mat.view(b, num_agents, kernel_size**2, num_agents, h, w).unsqueeze(4)

            padded_val_feat = get_padded_feat(val_feat, kernel_size).view(b, num_agents, num_agents, c, h+kernel_size-1, w+kernel_size-1) # (b, num_agents, num_agents, c, h+2, w+2)
            count = 0
            weighted_val_feat = torch.zeros([b, num_agents, num_agents, c, h, w], dtype=torch.float32, device=val_feat.device)
            for i in range(kernel_size):
                for j in range(kernel_size):
                    weighted_val_feat += weight_mat[:,:,count,:,:,:] * padded_val_feat[:,:,:,:,i:h+i,j:w+j]
                    count += 1
            
            feat_fuse = weighted_val_feat.sum(dim=1)    # (b*num_agents, c, h, w)
            weight_mats.append(weight_mat)

            post_commu_feats = feat_fuse * 0.5 + feat_map * 0.5
            x[c_layer] = post_commu_feats
        return x, weight_mats, val_feats

    def QUALITYMAP_MESSAGE(self, x, shift_mats, round, comm_thre=0.1, sigma=0, fusion='context', compress_flag=False):
        
        def _message_packing(val_feats, quality_maps, confidence_maps, compress_flag=False):
            b, k_agents, q_agents, c, h, w = val_feats.shape

            # compression
            if compress_flag:
                # weighting feature map
                ############# TO_FIX ############
                if quality_maps is None:
                    compression_maps = quality_maps
                else:
                    compression_maps = quality_maps[0] * (1 - confidence_maps).unsqueeze(1)
                feat_comm = []
                compression_comm = []
                for q in range(q_agents):
                    feat_comm.append(torch.cat([val_feats[:,:q,q], val_feats[:,q+1:,q]], dim=1).unsqueeze(2))
                    compression_comm.append(torch.cat([compression_maps[:,:q,q], compression_maps[:,q+1:,q]], dim=1).unsqueeze(2))
                feat_comm = torch.cat(feat_comm, dim=2)
                compression_comm = torch.cat(compression_comm, dim=2)

                feat_comm = feat_comm * compression_comm # (b, k_agents-1, q_agents, c, h, w)
                feat_comm = feat_comm.reshape(b*(k_agents-1)*q_agents, c, h, w)
                
                if self.compressor.training:
                    comp_gt = feat_comm.clone()
                    comp_out = self.compressor(feat_comm)
                    recon_feat = comp_out["x_hat"].clone()
                    likelihoods = comp_out["likelihoods"]
                else:
                    self.compressor.update()
                    comp_gt = feat_comm.clone()
                    comp_out = self.compressor.compress(feat_comm)
                    strings = comp_out["strings"]
                    shape = comp_out["shape"]
                    comm_size = len(strings[0][0]) + len(strings[1][0])
                    recon_feat = self.compressor.decompress(strings, shape)["x_hat"]
                recon_feat = recon_feat.reshape(b, k_agents-1, q_agents, c, h, w)
                recon_feat = recon_feat / (compression_comm + 1e-6)
                local_com_mat_comm = []
                for q in range(q_agents):
                    local_com_mat_comm.append(torch.cat([recon_feat[:,:q,q], val_feats[:,q:q+1,q], recon_feat[:,q:,q]], dim=1).unsqueeze(2))
                local_com_mat_comm = torch.cat(local_com_mat_comm, dim=2)
            else:
                local_com_mat_comm = val_feats
                comp_gt = None
                comp_out = None
            return local_com_mat_comm, comp_gt, comp_out
        
        def _transformer_message_fusing(query_feat, val_feat, quality_map, round_id=0):
            query_feat = self.add_pe_map(query_feat)
            b, num_agents, c, h, w = query_feat.size()

            src = query_feat.permute(0,1,3,4,2).contiguous().view(b*num_agents*h*w,c).contiguous().unsqueeze(0)    # (1, b*num_agents*h*w, c)
            val = val_feat.permute(1,0,2,4,5,3).contiguous().view(num_agents, b*num_agents*h*w,c).contiguous()
            
            src2, weight_mat = eval('self.cross_attn'+str(c_layer))(src, val, value=val, attn_mask=None, key_padding_mask=None)
            src = src + eval('self.dropout1_'+str(c_layer))(src2)
            src = eval('self.norm1_'+str(c_layer))(src)
            src2 = eval('self.linear2_'+str(c_layer))(eval('self.dropout0_'+str(c_layer))(F.relu(eval('self.linear1_'+str(c_layer))(src))))
            src = src + eval('self.dropout2_'+str(c_layer))(src2)
            src = eval('self.norm2_'+str(c_layer))(src)

            feat_fuse = src.view(b, num_agents, h, w, c).contiguous().permute(0, 1, 4, 2, 3).contiguous()
            weight_mat = weight_mat.view(b, num_agents, h, w, num_agents).contiguous().permute(0, 4, 1, 2, 3).contiguous().unsqueeze(3)
            return feat_fuse, weight_mat
        
        def _weighttransformer_message_fusing(query_feat, val_feat, quality_map, round_id=0):
            query_feat = self.add_pe_map(query_feat)
            b, num_agents, c, h, w = query_feat.size()

            src = query_feat.permute(0,1,3,4,2).contiguous().view(b*num_agents*h*w,c).contiguous().unsqueeze(0)    # (1, b*num_agents*h*w, c)
            val = val_feat.permute(1,0,2,4,5,3).contiguous().view(num_agents, b*num_agents*h*w,c).contiguous()
            quality_map = quality_map.permute(0,2,3,4,5,1).contiguous().view(b*num_agents*h*w, 1, num_agents).contiguous()

            src2, weight_mat = eval('self.cross_attn'+str(c_layer))(src, val, value=val, attn_mask=None, key_padding_mask=None, quality_map=quality_map)
            src = src + eval('self.dropout1_'+str(c_layer))(src2)
            src = eval('self.norm1_'+str(c_layer))(src)
            src2 = eval('self.linear2_'+str(c_layer))(eval('self.dropout0_'+str(c_layer))(F.relu(eval('self.linear1_'+str(c_layer))(src))))
            src = src + eval('self.dropout2_'+str(c_layer))(src2)
            src = eval('self.norm2_'+str(c_layer))(src)

            feat_fuse = src.view(b, num_agents, h, w, c).contiguous().permute(0, 1, 4, 2, 3).contiguous()
            weight_mat = weight_mat.view(b, num_agents, h, w, num_agents).contiguous().permute(0, 4, 1, 2, 3).contiguous().unsqueeze(3)
            return feat_fuse, weight_mat
        
        def _wopetransformer_message_fusing(query_feat, val_feat, quality_map, round_id=0):
            query_feat = self.add_pe_map(query_feat)
            b, num_agents, c, h, w = query_feat.size()

            src = query_feat.permute(0,1,3,4,2).contiguous().view(b*num_agents*h*w,c).contiguous().unsqueeze(0)    # (1, b*num_agents*h*w, c)
            val = val_feat.permute(1,0,2,4,5,3).contiguous().view(num_agents, b*num_agents*h*w,c).contiguous()
            
            src2, weight_mat = eval('self.cross_attn'+str(c_layer))(src, val, value=val, attn_mask=None, key_padding_mask=None)
            src = src + eval('self.dropout1_'+str(c_layer))(src2)
            src = eval('self.norm1_'+str(c_layer))(src)
            src2 = eval('self.linear2_'+str(c_layer))(eval('self.dropout0_'+str(c_layer))(F.relu(eval('self.linear1_'+str(c_layer))(src))))
            src = src + eval('self.dropout2_'+str(c_layer))(src2)
            src = eval('self.norm2_'+str(c_layer))(src)

            feat_fuse = src.view(b, num_agents, h, w, c).contiguous().permute(0, 1, 4, 2, 3).contiguous()
            weight_mat = weight_mat.view(b, num_agents, h, w, num_agents).contiguous().permute(0, 4, 1, 2, 3).contiguous().unsqueeze(3)
            return feat_fuse, weight_mat
        
        def _get_local_mask(x, kernel_size=3, stride=1):
            def get_padded_feat(x, kernel_size=3, stride=1):
                b, c, h, w = x.shape
                padding_w = torch.zeros([b, c, h, (kernel_size-1)*stride//2], dtype=torch.float32, device=x.device)
                padding_h = torch.zeros([b, c, (kernel_size-1)*stride//2, w+(kernel_size-1)*stride], dtype=torch.float32, device=x.device)
                x = torch.cat([padding_w, x, padding_w], dim=-1)    # (b, num_agents, c, h, w+kernel_size-1)
                x = torch.cat([padding_h, x, padding_h], dim=-2)    # (b, num_agents, c, h+kernel_size-1, w+kernel_size-1)
                return x
            
            # mask: (b, k_agents, q_agents, 1, h, w)
            shape_len = len(x.shape)
            if shape_len == 6:
                b, k_agents, q_agents, c, h, w = x.shape
                x = x.flatten(0, 2)
            else:
                b, q_agents, c, h, w = x.shape
                x = x.flatten(0, 1)
            x = get_padded_feat(x, kernel_size, stride)
            x_local = []
            for i in range(kernel_size):
                for j in range(kernel_size):
                    x_local.append(x[:,:,i*stride:h+i*stride,j*stride:w+j*stride].unsqueeze(1))
            
            x_local = torch.cat(x_local, dim=1) # (b, kernel**2, c, h, w)
            x_local = x_local.max(dim=1)[0] # (b, c, h, w)
            if shape_len == 6:
                x_local = x_local.view(b, k_agents, q_agents, 1, h, w)
            else:
                x_local = x_local.view(b, q_agents, 1, h, w)
            return x_local

        def _get_topk(comm_maps, k=200):
            b, k_agents, q_agents, c, h, w = comm_maps.shape
            comm_maps = comm_maps.flatten(0,3)  # (b, h, w)
            comm_maps = comm_maps.flatten(1) # (b, h*w)

            topk_v, indices = torch.topk(comm_maps, k, dim=-1)  # topk_v: (b*num_agents*num_agents,max_len)
            ones_mask = torch.ones_like(comm_maps).to(comm_maps.device)
            zeros_mask = torch.zeros_like(comm_maps).to(comm_maps.device)
            topk_comm_maps = zeros_mask.scatter_(1, indices, ones_mask)
            topk_comm_maps = topk_comm_maps.view(b, k_agents, q_agents, c, h, w)
            return topk_comm_maps


        def _communication_graph_learning(val_feats, quality_maps, confidence_maps, prev_communication_mask, round_id=0, thre=0.03, sigma=0):
            # val_feats: (b, k_agents, q_agents, c, h, w) 
            # confidence_maps: (b, q_agents, 1, h, w)
            # quality_maps: (b, k_agents, q_agents, 1, h, w)
            # thre: threshold of objectiveness
            # a_ji = (1 - q_i)*q_ji
            b, k_agents, q_agents, _, h, w = quality_maps.shape
            if round_id == 0:
                communication_maps = quality_maps
            else:
                beta = 0
                communication_maps = quality_maps * (1 - confidence_maps).unsqueeze(1) - beta * confidence_maps.unsqueeze(1)
                communication_maps = F.relu(communication_maps)
            
            # thre = np.random.choice([0, 0.001, 0.01, 0.03])
            if sigma > 0:
                communication_maps = communication_maps.view(b*k_agents*q_agents, 1, h, w)
                # sigma = 1.0
                communication_maps = self.__getattr__('gaussian_filter'+str(int(10*sigma)))(communication_maps)
                communication_maps = communication_maps.view(b, k_agents, q_agents, 1, h, w)

            ones_mask = torch.ones_like(communication_maps).to(communication_maps.device)
            zeros_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
            communication_mask = torch.where((communication_maps - thre)>1e-6, ones_mask, zeros_mask)
            # communication_mask, communication_maps_topk = _bandwidthAware_comm_mask(communication_maps, B=1)

            if round_id > 0:
                # Local context
                communication_mask = _get_local_mask(communication_mask, kernel_size=11, stride=1)
                communication_maps = communication_maps * communication_mask
                communication_thres = [0.0, 0.001, 0.01, 0.03, 0.06, 0.08, 0.1, 0.13, 0.16, 0.20, 0.24, 0.28, 1.0]
                for thre_idx, comm_thre in enumerate(communication_thres):
                    if (comm_thre == thre) and (thre_idx>=2):
                        if round_id == 1:
                            thre = communication_thres[thre_idx-1]
                        else:
                            thre = communication_thres[thre_idx-2]
                        break
                # thre = min(0.001, thre)
                # thre = 0.08
                communication_mask = torch.where((communication_maps - thre)>1e-6, ones_mask, zeros_mask)
                # if round_id == 2:
                #     communication_mask = F.relu(communication_mask - prev_communication_mask)
                # communication_mask = _get_topk(communication_maps, k=400)
            # Range
            # communication_mask_0 = torch.where((communication_maps - 1e-5)>1e-6, ones_mask, zeros_mask)
            # communication_mask_1 = torch.where((0.001 - communication_maps)>1e-6, ones_mask, zeros_mask)
            # communication_mask = communication_mask_0 * communication_mask_1

            if round_id == 0:
                if sigma > 0:
                    confidence_maps = confidence_maps.view(b*q_agents, 1, h, w)
                    confidence_maps = self.__getattr__('gaussian_filter'+str(int(10*sigma)))(confidence_maps)
                    confidence_maps = confidence_maps.view(b, q_agents, 1, h, w)
                confidence_mask = torch.where((confidence_maps - thre)>1e-6, ones_mask[:,:,0], zeros_mask[:,:,0])
                # Local context
                # confidence_mask = _get_local_mask(confidence_mask, kernel_size=11, stride=1)
                # confidence_maps = confidence_maps * confidence_mask
                # confidence_mask = torch.where((confidence_maps - 0.001)>1e-6, ones_mask[:,:,0], zeros_mask[:,:,0])
                communication_rate = confidence_mask.sum()/(b*h*w*q_agents)
            else:
                communication_mask_list = []
                for i in range(num_agents):
                    communication_mask_list.append(torch.cat([communication_mask[:,:i,i],communication_mask[:,i+1:,i]], dim=1).unsqueeze(2))
                communication_mask_list = torch.cat(communication_mask_list, dim=2) # (1, 4, 5, 1, h, w)
                communication_rate = communication_mask_list.sum()/(b*h*w*q_agents*(k_agents-1))

            communication_mask_nodiag = communication_mask.clone()
            # communication_maps_topk_nodiag = communication_maps_topk.clone()
            for q in range(q_agents):
                communication_mask_nodiag[:,q,q] = ones_mask[:,q,q]
                # communication_maps_topk_nodiag[:,q,q] = ones_mask[:,q,q]
                # val_feats_aftercom[:,:q,q] = val_feats[:,:q,q] * communication_mask[:,:q,q]
                # val_feats_aftercom[:,q+1:,q] = val_feats[:,q+1:,q] * communication_mask[:,q+1:,q]
            val_feats_aftercom = val_feats * communication_mask_nodiag
            return val_feats_aftercom, communication_mask, communication_rate, None
        
        def _decoder(feat_maps, round_id):
            '''
            Input: feature map (single agent or with collaboration)
            Output: predictions
            '''
            results = {}
            ############ Infer the quality map ##############
            y = []
            for i in range(self.last_level - self.first_level):
                b, num_agents, c, h, w = feat_maps[i].shape
                y.append(feat_maps[i].view(b*num_agents, c, h, w).clone())
            self.ida_up(y, 0, len(y))
            for head in self.heads:
                results[head+'_single_r{}'.format(round_id)] = self.__getattr__(head)(y[-1]) # (b*num_agent, 2, 112, 200)
            ##################################################
            return results
        
        
        weight_mats_dict = {}
        results_dict = {}
        prev_confidence_maps = None
        prev_comm_mask = None
        for round_id in range(round):
            b, num_agents, c, h, w = x[0].shape
            results = _decoder(x, round_id)
            results_dict.update(results)
            confidence_maps = results['hm_single_r{}'.format(round_id)].clone().sigmoid() # (b, num_agents, 1, h, w)
            if self.trans_layer[0] > 0:
                confidence_maps = self.__getattr__('pool_net'+str(self.trans_layer[0]))(confidence_maps)
            confidence_maps = confidence_maps.reshape(b, num_agents, 1, confidence_maps.shape[-2], confidence_maps.shape[-1])
            
            confidence_maps_list = [0,0,0,0]
            confidence_maps_list[self.trans_layer[0]] = confidence_maps
            quality_maps = self.get_colla_feats(confidence_maps_list, shift_mats, self.trans_layer, with_pos=False)
            
            if fusion == 'wopetransformer':
                val_feats = self.get_colla_feats(x, shift_mats, self.trans_layer, with_pos=False) # (b, k_agents, q_agents, c, h, w)
            else:
                val_feats = self.get_colla_feats(x, shift_mats, self.trans_layer, with_pos=True) # (b, k_agents, q_agents, c, h, w)
            
            if prev_confidence_maps is None:
                val_feats[0], communication_mask, communication_rate, _ = _communication_graph_learning(val_feats[0], quality_maps[0], confidence_maps, prev_comm_mask, round_id, thre=comm_thre, sigma=sigma)
            else:
                val_feats[0], communication_mask, communication_rate, _ = _communication_graph_learning(val_feats[0], quality_maps[0], prev_confidence_maps, prev_comm_mask, round_id, thre=comm_thre, sigma=sigma)
            val_feats[0], comp_gt, comp_out = _message_packing(val_feats[0], quality_maps, prev_confidence_maps, compress_flag=compress_flag)
            weight_mats = []
            for i, c_layer in enumerate(self.trans_layer):
                feat_map = x[c_layer]
                b, num_agents, c, h, w = feat_map.shape
                val_feat = val_feats[i]
                quality_map = quality_maps[i]
                if fusion == 'transformer':
                    feat_fuse, weight_mat = _transformer_message_fusing(feat_map, val_feat, quality_map, round_id=0)
                elif fusion == 'weighttransformer':
                    feat_fuse, weight_mat = _weighttransformer_message_fusing(feat_map, val_feat, quality_map, round_id=0)
                elif fusion == 'wopetransformer':
                    feat_fuse, weight_mat = _wopetransformer_message_fusing(feat_map, val_feat, quality_map, round_id=0)
                weight_mats.append(weight_mat)
                x[c_layer] = feat_fuse
            weight_mats_dict[round_id] = weight_mats
            prev_confidence_maps = confidence_maps
            prev_comm_mask = communication_mask

            # x_b1 = [feat_b1,x[1],x[2],x[3]]
            # results = _decoder(x_b1, round_id=1)
            # results_dict.update(results)
        
        return x, weight_mat, results_dict, quality_map, val_feat, communication_mask, communication_rate, comp_gt, comp_out
        
    def Consistency_Mask(self, x, x_max):
        '''
        Args:
            x: (b, k_agents, q_agents, c, h, w)
        Function:
            U_q = \mean_{k=0,...,N_q; k\= q} (|| x_q - x_k || * M_k)
        '''
        C = []
        b, k_agents, q_agents, c, h, w = x.shape
        ones = torch.ones([b, k_agents-1, h, w]).to(x.device)
        zeros = torch.zeros([b, k_agents-1, h, w]).to(x.device)
        for i in range(q_agents):
            x_q = x[:,i,i]
            x_k = torch.cat([x[:,:i,i], x[:,i+1:,i]], dim=1) # (b, k_agents-1, c, h, w)
            max_k = torch.cat([x_max[:,:i,i], x_max[:,i+1:,i]], dim=1).squeeze(dim=2) # (b, k_agents-1, h, w)
            mask_ks = torch.where(max_k>0, ones, zeros) # (b, k_agents-1, h, w)
            mask_k = mask_ks.max(dim=1)[0]
            max_q = x_max[:,i,i].squeeze(1)
            mask_q = torch.where(max_q>0, ones[:,0], zeros[:,0])[0]   # (b, h, w)
            l_k = ((x_q.unsqueeze(1) - x_k).mean(dim=-3) * mask_ks).mean(dim=-3) * (k_agents-1) / (mask_ks.sum(dim=1)+1e-6) # (b, h, w)

            # Update x with the consistency mask   
            c_k = l_k.sigmoid() * mask_k * mask_q    # uncertainty for both visible area
            c_k += ones[:,0] * mask_k * (1 - mask_q)
            c_k += ones[:,0] * 0.5 * (1 - mask_k) * (1 - mask_q)
            c_k = c_k.unsqueeze(1)

            x_q = (1 - c_k) * x_q
            x_k = c_k.unsqueeze(1) * x_k
            C.append(c_k)
        C = torch.cat(C, dim=1) # (b, q_agents, h, w)
        return x, C


    def V2V_MESSAGE(self, x, shift_mats):
        weight_mats = []
        for _ in range(self.gnn_iter_num):
            val_feats = self.get_colla_feats(x, shift_mats, self.trans_layer)    # (b, k_agents, q_agents, c, h, w)
            for i, c_layer in enumerate(self.trans_layer):
                feat_map = x[c_layer]
                b, num_agents, c, h, w = feat_map.size()

                val_feat = val_feats[i]

                # feat_map = self.add_pe_map(feat_map)
                # val_feat = self.add_pe_map(val_feat)

                # feat_map = self.add_coord_map(feat_map)
                # val_feat = self.add_coord_map(val_feat)
                # feat_map = self.add_coord_map(feat_map, normalized=False)
                # val_feat = self.add_coord_map(val_feat, normalized=False)
                
                mean_feat = val_feat.sum(dim=1)# (b, num_agents, c, h, w)
                mean_feat = (mean_feat - feat_map)/4.0
                cat_feat = torch.cat([feat_map, mean_feat], dim=-3).reshape(b*num_agents, 2*c, h, w).contiguous().unsqueeze(1) # (b*num_agents, 1, 2*c, h, w)
                updated_feat, _ = eval('self.convgru'+str(c_layer))(cat_feat, None)
                feat_map = updated_feat.squeeze(1) # (b*num_agents, c, h, w)

                x[c_layer] = feat_map.view(b, num_agents, c, h, w)
        return x, weight_mats, val_feats
    

    def WHEN2COM_MESSAGE(self, x, shift_mats):
        val_feats = self.get_colla_feats(x, shift_mats, self.trans_layer)

        # When2com
        b, num_agents, c, h, w = x[-1].size()
        qk_feat = x[-1].view(b*num_agents, c, h, w).contiguous()
        query_key_maps = self.query_key_net(qk_feat) # (b*num_agents, c, h, w)
        querys = self.query_net(query_key_maps) # (b*num_agents, query_size)
        keys = self.key_net(query_key_maps) # (b*num_agents, key_size)
        query_mat = querys.view(b, num_agents, self.query_size) # (b, num_agents, query_size)
        key_mat = keys.view(b, num_agents, self.key_size) # (b, num_agents, key_size)

        weight_mats = []
        for c_layer in self.trans_layer:
            # 1. Choose the layer
            feat_map = x[c_layer]
            b, num_agents, c, h, w = feat_map.size()
            val_feat = val_feats[c_layer]
            # When2com
            feat_fuse, prob_action = self.attention_net(query_mat, key_mat, val_feat)    # (b, num_agents, c, h, w)

            # 4. Return fused feat
            post_commu_feats = feat_fuse * 0.5 + feat_map * 0.5
            x[c_layer] = post_commu_feats
        return x, weight_mats, val_feats

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
        
        if self.vis:
            cur_trans_mats = trans_mats[0].view(b*num_agents, 3, 3)
            worldgrid2worldcoord_mat = torch.Tensor(np.array([[map_scale, 0, 0], [0, map_scale, 0], [0, 0, 1]])).to(cur_trans_mats.device)
            cur_trans_mats = shift_mats[0].view(b*num_agents, 3, 3) @ torch.inverse(cur_trans_mats @ worldgrid2worldcoord_mat).contiguous()
            images_warped = kornia.warp_perspective(images, cur_trans_mats, dsize=(self.feat_H, self.feat_W))
            images_warped = images_warped.contiguous().view(b, num_agents, 3, self.feat_H, self.feat_W).contiguous()
            images_warped = self.get_colla_feats([images_warped], shift_mats, [0], with_pos=False)

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
            x_inter = self.base(images)
            x_inter = self.dla_up(x_inter)  # list [(B, 64, 112, 200), (B, 128, 56, 100), (B, 256, 28, 50), (B, 512, 14, 25)] B = b * num_agents

        if self.feat_mode in ['early', 'inter']:
            x = x_early if self.feat_mode == 'early' else x_inter
        elif self.feat_mode == 'fused':
            x = x_inter

        scale = 1/map_scale
        trans_mats = [trans_mats[0]]
        depth_weighted_feat_maps = [x]
        
        global_x_multi_depth = []
        
        for depth_layer, init_trans_mats in enumerate(trans_mats):
            cur_x = depth_weighted_feat_maps[depth_layer]
            global_x = []
            warp_images_list = []
            init_trans_mats = init_trans_mats.view(b*num_agents, 3, 3)
            # cur_trans_mats = init_trans_mats.view(b*num_agents, 3, 3)
            # fig, axes = plt.subplots(5, 4)

            for c_layer, feat_map in enumerate(cur_x):
                _, c, h, w = feat_map.size()
                init_shift_mats = shift_mats[c_layer].view(b*num_agents, 3, 3).contiguous()

                worldgrid2worldcoord_mat = torch.Tensor(np.array([[2**c_layer*scale, 0, 0], [0, 2**c_layer*scale, 0], [0, 0, 1]])).to(init_trans_mats.device)
                img_trans_mats = init_shift_mats @ torch.inverse(init_trans_mats @ worldgrid2worldcoord_mat).contiguous()
                warp_images = kornia.warp_perspective(images, img_trans_mats, dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                warp_images_list.append(warp_images)

                #########################################################################
                #                              Hard Warping                             #
                #########################################################################
                # uav_i --> global coord
                worldgrid2worldcoord_mat = torch.Tensor(np.array([[2**c_layer*map_scale, 0, 0], [0, 2**c_layer*map_scale, 0], [0, 0, 1]])).to(init_trans_mats.device)
                feat_zoom_mats = torch.Tensor(np.array(np.diag([2**(c_layer+2), 2**(c_layer+2), 1]), dtype=np.float32)).to(init_trans_mats.device)
                cur_trans_mats = init_shift_mats @ torch.inverse(init_trans_mats @ worldgrid2worldcoord_mat).contiguous() @ feat_zoom_mats
                global_feat = kornia.warp_perspective(feat_map, cur_trans_mats, dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                global_x.append(global_feat)
                
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
        if self.feat_mode == 'fused':
            global_x_fused = []
            for c_layer, feat_map in enumerate(x_early):
                # Mean
                # global_x[c_layer] = global_x[c_layer] + feat_map
                # global_x_fused.append(torch.cat([global_x[c_layer].unsqueeze(1), feat_map.unsqueeze(1)], dim=1).mean(dim=1))
                # Max
                global_x_fused.append(torch.cat([global_x[c_layer].unsqueeze(1), feat_map.unsqueeze(1)], dim=1).max(dim=1)[0])
        
        communication_rate = None
        results = None
        if self.trans_layer[-1] == -2:
            pass
        else:
            if self.feat_mode in ['early', 'inter']:
                single_x = global_x
            elif self.feat_mode == 'fused':
                single_x = global_x_fused

            for c_layer, feat_map in enumerate(single_x):
                _, c, h, w = feat_map.shape
                feat_map = feat_map.view(b, num_agents, c, h, w)
                single_x[c_layer] = feat_map

            if self.message_mode in ['Mean', 'Max', 'Pointwise', 'Concat']:
                colla_x, weight_mats, val_feats = self.COLLA_MESSAGE(single_x, shift_mats)
            elif self.message_mode in ['V2V']:
                colla_x, weight_mats, val_feats = self.V2V_MESSAGE(single_x, shift_mats)
            elif self.message_mode in ['When2com']:
                colla_x, weight_mats, val_feats = self.WHEN2COM_MESSAGE(single_x, shift_mats)
            elif self.message_mode in ['QualityMapTransformer']:
                colla_x, weight_mats, results, quality_map, val_feat, communication_mask, communication_rate, comp_gt, comp_out = self.QUALITYMAP_MESSAGE(single_x, shift_mats, self.round, self.comm_thre, self.sigma, fusion='transformer', compress_flag=self.compress_flag)
            elif self.message_mode in ['QualityMapTransformerWOPE']:
                colla_x, weight_mats, results, quality_map, val_feat, communication_mask, communication_rate, comp_gt, comp_out = self.QUALITYMAP_MESSAGE(single_x, shift_mats, self.round, self.comm_thre, self.sigma, fusion='wopetransformer', compress_flag=self.compress_flag)
            elif self.message_mode in ['QualityMapTransformerWeight']:
                colla_x, weight_mats, results, quality_map, val_feat, communication_mask, communication_rate, comp_gt, comp_out = self.QUALITYMAP_MESSAGE(single_x, shift_mats, self.round, self.comm_thre, self.sigma, fusion='weighttransformer', compress_flag=self.compress_flag)
            
            for c_layer, feat_map in enumerate(colla_x):
                b, num_agents, c, h, w = feat_map.shape
                feat_map = feat_map.view(b*num_agents, c, h, w)
                colla_x[c_layer] = feat_map

            if self.feat_mode in ['early', 'inter']:
                global_x = colla_x
            elif self.feat_mode == 'fused':
                global_x_fused = colla_x

        if self.feat_mode in ['early', 'inter']:
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

        # Visualize weight mats (b, N, h, w)
        if self.vis:
            root_dir = os.path.dirname(__file__)
            for layer_ind, c_layer in enumerate(self.trans_layer):
                # import ipdb; ipdb.set_trace()
                cur_images = images_warped[layer_ind] # (b, k_agents, q_agents, 3, h, w)
                val_feat = val_feat.max(dim=-3)[0].unsqueeze(3)
                b, k_agents, q_agents, _, _, _ = cur_images.shape
                cur_time = '{}'.format(time.time())[-4:]
                for i in range(b):
                    save_dir = os.path.join(root_dir, 'qualitymap_attn_weights_comm_show', '{}_{}'.format(cur_time, i))
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    np.save(os.path.join(save_dir, 'weight_mats'), weight_mats.detach().cpu().numpy())
                    np.save(os.path.join(save_dir, 'quality_map'), quality_map.detach().cpu().numpy())
                    np.save(os.path.join(save_dir, 'communication_mask'), communication_mask.detach().cpu().numpy())
                    np.save(os.path.join(save_dir, 'cur_images'), cur_images.detach().cpu().numpy())
                    np.save(os.path.join(save_dir, 'val_feat'), val_feat.detach().cpu().numpy())
                    for j in range(q_agents):
                        fig, axes = plt.subplots(4, 5, figsize=(20,8))
                        for k in range(k_agents):
                            if j == k:
                                axes[0,k].imshow((weight_mats[i,k,j].detach().cpu().numpy().transpose(1,2,0) * 255.).astype('uint8')) # (b, q_agents, h, w)
                            else:
                                axes[0,k].imshow(((weight_mats[i,k,j]*communication_mask[i,k,j]).detach().cpu().numpy().transpose(1,2,0) * 255.).astype('uint8')) # (b, q_agents, h, w)
                            axes[0,k].set_xticks([])
                            axes[0,k].set_yticks([])

                            axes[1,k].imshow((quality_map[i,k,j].detach().cpu().numpy().transpose(1,2,0) * 255.).astype('uint8')) # (b, q_agents, h, w)
                            axes[1,k].set_xticks([])
                            axes[1,k].set_yticks([])

                            axes[2,k].imshow((communication_mask[i,k,j].detach().cpu().numpy().transpose(1,2,0) * 255.).astype('uint8')) # (b, q_agents, h, w)
                            axes[2,k].set_xticks([])
                            axes[2,k].set_yticks([])

                            axes[3,k].imshow((cur_images[i,k,j].detach().cpu().numpy().transpose(1,2,0) * 255.).astype('uint8')) # (b, q_agents, h, w)
                            axes[3,k].set_xticks([])
                            axes[3,k].set_yticks([])
                        fig.tight_layout()
                        plt.savefig(os.path.join(save_dir, '{}.png'.format(j)))
                        plt.close()

        if results is not None:
            global_z.update(results)
        if self.compress_flag:
            global_z['comp_gt'] = comp_gt
            global_z['comp_out'] = comp_out
            global_z['comp_aux_loss'] = self.compressor.aux_loss()
        global_z['comm_rate'] = torch.zeros(b,1).to(global_x[0].device) if communication_rate is None else communication_rate
        return [global_z]
    
    def forward(self, images, trans_mats, shift_mats, map_scale=1.0):
        if self.coord == 'Global':
            return self.GlobalCoord_forward(images, trans_mats, shift_mats, map_scale)
        else:
            return self.LocalCoord_forward(images, trans_mats)
        

def get_pose_net(num_layers, heads, head_conv=256, down_ratio=4, message_mode='NO_MESSAGE_NOWARP', trans_layer=[3], coord='Local', warp_mode='HW', depth_mode='Unique', feat_mode='inter', feat_shape=[192, 352], round=1, compress_flag=False, comm_thre=0.1, sigma=0):
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
                    feat_shape=feat_shape,
                    round=round,
                    compress_flag=compress_flag,
                    comm_thre=comm_thre,
                    sigma=sigma)
    return model

