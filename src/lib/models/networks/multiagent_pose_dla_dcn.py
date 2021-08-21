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
from ipdb.__main__ import set_trace
import numpy as np
from os.path import join

import torch
from torch import nn
from torch._C import device
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import kornia

from .DCNv2.dcn_v2 import DCN
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
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=(3,3), stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
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
        self.n_feat = int(256 * (input_feat_h//4) * (input_feat_w//4))
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
        # self.first_level = int(np.log2(down_ratio))
        # self.last_level = last_level
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

    def __init__(self, query_size, key_size, warp_flag=False, attn_dropout=0.1):
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

class DLASeg(nn.Module):
    def __init__(self, base_name, heads, pretrained, down_ratio, final_kernel,
                 last_level, head_conv, out_channel=0, message_mode='NO_MESSAGE', trans_layer=[3], coord='Local'):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.base = globals()[base_name](pretrained=pretrained)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)

        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level], 
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
        if self.message_mode in ['When2com']:
            self.query_key_net = policy_net4(base_name, pretrained, down_ratio, last_level)
            self.key_net = km_generator(out_size=self.key_size, input_feat_h=448//32*4, input_feat_w=800//32*4)
            self.query_net = km_generator(out_size=self.query_size, input_feat_h=448//32*4, input_feat_w=800//32*4)
            self.attention_net = MIMOGeneralDotProductAttention(self.query_size, self.key_size)
        elif self.message_mode in ['V2V']:
            self.gnn_iter_num = 3
            for c_layer in self.trans_layer:
                if c_layer >= 0:
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
                    weight_net = nn.Conv2d(128*2**c_layer, 1, kernel_size=1, stride=1, padding=0)
                    self.__setattr__('weight_net'+str(c_layer), weight_net)

        if self.trans_layer[0] == -1:
            self.conv0 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
            self.conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
    
    def NO_MESSAGE_NOWARP(self, images):
        b, num_agents, img_c, img_h, img_w = images.size()
        images = images.view(b*num_agents, img_c, img_h, img_w)
        # Encoder
        x = self.base(images)
        x_0, x_1, x_2, x_3 = self.dla_up(x)  # list [(B, 64, 112, 200), (B, 128, 56, 100), (B, 256, 28, 50), (B, 512, 14, 25)] B = b * num_agents
        return [x_0, x_1, x_2, x_3]
    
    def NO_MESSAGE(self, images, trans_mats):
        b, num_agents, img_c, img_h, img_w = images.size()
        images = images.view(b*num_agents, img_c, img_h, img_w)

        if self.trans_layer[0] == -1:
            trans_mats = trans_mats.view(b*num_agents, 3, 3)
            # map_zoom_mats = torch.Tensor(np.array(np.diag([img_h/400, img_w/500, 1]), dtype=np.float32)).to(trans_mats.device)
            # cur_trans_mats = map_zoom_mats @ trans_mats
            # images = kornia.warp_perspective(images, cur_trans_mats, dsize=(img_h, img_w))
            # map_zoom_mats = torch.Tensor(np.array(np.diag([img_h*2/400, img_w*2/500, 1]), dtype=np.float32)).to(trans_mats.device)
            # cur_trans_mats = map_zoom_mats @ trans_mats
            worldgrid2worldcoord_mat = torch.Tensor(np.array([[500/(img_w*2), 0, -200], [0, 500/(img_h*2), -250], [0, 0, 1]])).to(trans_mats.device)
            cur_trans_mats = torch.inverse(trans_mats @ worldgrid2worldcoord_mat)
            images = kornia.warp_perspective(images, cur_trans_mats, dsize=(img_h*2, img_w*2))
        
            # Encoder
            x = self.base(images)
            x = self.dla_up(x)  # list [(B, 64, 112, 200), (B, 128, 56, 100), (B, 256, 28, 50), (B, 512, 14, 25)] B = b * num_agents
            
            for c_layer, feat_x in enumerate(x):
                # import ipdb; ipdb.set_trace()
                feat_map = eval('self.conv'+str(c_layer))(feat_x)
                _, c, h, w = feat_map.size()

                # 2. Get the value mat (trans feature to global coord)  # val_mat: (b, k_agents, q_agents, c, h, w)
                # uav_i --> global coord
                # map_zoom_mats = torch.Tensor(np.array(np.diag([h/400, w/500, 1]), dtype=np.float32)).to(trans_mats.device)
                worldgrid2worldcoord_mat = torch.Tensor(np.array([[500/w, 0, -200], [0, 500/h, -250], [0, 0, 1]])).to(trans_mats.device)
                feat_zoom_mats = torch.Tensor(np.array(np.diag([2**(c_layer+2), 2**(c_layer+2), 1]), dtype=np.float32)).to(trans_mats.device)
                cur_trans_mats = torch.inverse(trans_mats @ worldgrid2worldcoord_mat) @ feat_zoom_mats
                
                # 3. Return fused feat
                # global coord --> uav_j   
                trans_mats_inverse = torch.inverse(cur_trans_mats).view(b, num_agents, 3, 3).contiguous().view(b*num_agents, 3, 3).contiguous()
                post_commu_feats = kornia.warp_perspective(feat_map, trans_mats_inverse, dsize=(h, w)) # (b*num_agents*num_agents, c, h, w)
                x[c_layer] = post_commu_feats
            return x
        else:
            x = self.base(images)
            x = self.dla_up(x)  # list [(B, 64, 112, 200), (B, 128, 56, 100), (B, 256, 28, 50), (B, 512, 14, 25)] B = b * num_agents
            # 1. Choose the layer
            trans_mats = trans_mats.view(b*num_agents, 3, 3)
            for c_layer in self.trans_layer:
                feat_map = x[c_layer]
                _, c, h, w = feat_map.size()

                # 2. Get the value mat (trans feature to global coord)  # val_mat: (b, k_agents, q_agents, c, h, w)
                # uav_i --> global coord
                # map_zoom_mats = torch.Tensor(np.array(np.diag([h/400, w/500, 1]), dtype=np.float32)).to(trans_mats.device)
                worldgrid2worldcoord_mat = torch.Tensor(np.array([[500/w, 0, -200], [0, 500/h, -250], [0, 0, 1]])).to(trans_mats.device)
                feat_zoom_mats = torch.Tensor(np.array(np.diag([2**(c_layer+2), 2**(c_layer+2), 1]), dtype=np.float32)).to(trans_mats.device)
                cur_trans_mats = torch.inverse(trans_mats @ worldgrid2worldcoord_mat) @ feat_zoom_mats
                global_feat = kornia.warp_perspective(feat_map, cur_trans_mats, dsize=(h, w)) # (b*num_agents, c, h, w)
                
                # 3. Return fused feat
                # global coord --> uav_j   
                trans_mats_inverse = torch.inverse(cur_trans_mats).view(b, num_agents, 3, 3).contiguous().view(b*num_agents, 3, 3).contiguous()
                post_commu_feats = kornia.warp_perspective(global_feat, trans_mats_inverse, dsize=(h, w)) # (b*num_agents*num_agents, c, h, w)
                x[c_layer] = post_commu_feats
            return x

    def LOCAL_MESSAGE_NOWARP(self, images):
        b, num_agents, img_c, img_h, img_w = images.size()
        images = images.view(b*num_agents, img_c, img_h, img_w)
        # Encoder
        x = self.base(images)
        x = self.dla_up(x)  # list [(B, 64, 112, 200), (B, 128, 56, 100), (B, 256, 28, 50), (B, 512, 14, 25)] B = b * num_agents
        
        if self.trans_layer[-1] == -2:
            return x, []

        weight_mats = []
        val_mats = []
        for trans_layer in self.trans_layer:
            # 1. Choose the layer
            feat_map = x[trans_layer]
            _, c, h, w = feat_map.size()
            # 2. Get the value mat (In each uav coord)
            feat = feat_map.view(b, num_agents, c, h, w).contiguous()
            val_mat = feat.unsqueeze(2).expand(-1, -1, num_agents, -1, -1, -1).contiguous()

            # 3. Encode weights
            query_mat = feat.unsqueeze(1).expand(-1, num_agents, -1, -1, -1, -1)   # (b*num_agents, c, h, w) --> (b, k_agents, q_agents, c, h, w)
            weight_mat = torch.cat([query_mat, val_mat], dim=3).view(b*num_agents*num_agents, c*2, h, w)
            weight_mat = eval('self.weight_net'+str(trans_layer))(weight_mat)    # (b*k_agents*q_agents, c, h, w)
            weight_mat = weight_mat.view(b, num_agents, num_agents, 1, h, w).softmax(dim=1)
            feat_fuse = (weight_mat * val_mat).sum(dim=1)    # (b, num_agents, c, h, w)
            
            # 4. Return fused feat
            post_commu_feats = feat_fuse.view(b*num_agents, c, h, w).contiguous() * 0.5 + feat_map * 0.5
            x[trans_layer] = post_commu_feats
            weight_mats.append(weight_mat)
            val_mats.append(val_mat)
        return x, weight_mats, val_mats
    
    def LOCAL_MESSAGE(self, images, trans_mats):
        b, num_agents, img_c, img_h, img_w = images.size()
        images = images.view(b*num_agents, img_c, img_h, img_w)

        # Encoder
        x = self.base(images)
        x = self.dla_up(x)  # list [(B, 64, 112, 200), (B, 128, 56, 100), (B, 256, 28, 50), (B, 512, 14, 25)] B = b * num_agents

        if self.trans_layer[-1] == -2:
            return x, []

        weight_mats = []
        val_mats = []
        for trans_layer in self.trans_layer:
            # 1. Choose the layer
            feat_map = x[trans_layer]
            _, c, h, w = feat_map.size()

            # 2. Get the value mat (trans feature to current agent coord) # val_mat: (b, k_agents, q_agents, c, h, w)
            # uav_i --> global coord 
            trans_mats = trans_mats.view(b*num_agents, 3, 3)
            # map_zoom_mats = torch.Tensor(np.array(np.diag([h*2/400, w*2/500, 1]), dtype=np.float32)).to(trans_mats.device)
            worldgrid2worldcoord_mat = torch.Tensor(np.array([[500/(w*2), 0, -200], [0, 500/(h*2), -250], [0, 0, 1]])).to(trans_mats.device)
            feat_zoom_mats = torch.Tensor(np.array(np.diag([2**(trans_layer+2), 2**(trans_layer+2), 1]), dtype=np.float32)).to(trans_mats.device)
            cur_trans_mats = torch.inverse(trans_mats @ worldgrid2worldcoord_mat) @ feat_zoom_mats
            global_feat = kornia.warp_perspective(feat_map, cur_trans_mats, dsize=(2*h, 2*w)) # (b*num_agents, c, h, w)
            global_feat = global_feat.view(b, num_agents, c, 2*h, 2*w).contiguous().unsqueeze(2).expand(-1, -1, num_agents, -1, -1, -1)
            global_feat = global_feat.contiguous().view(b*num_agents*num_agents, c, 2*h, 2*w).contiguous()
            # global coord --> uav_j   
            trans_mats_inverse = torch.inverse(cur_trans_mats).view(b, num_agents, 3, 3).contiguous().unsqueeze(1).expand(-1, num_agents, -1, -1, -1).contiguous().view(b*num_agents*num_agents, 3, 3).contiguous()
            val_mat = kornia.warp_perspective(global_feat, trans_mats_inverse, dsize=(h, w)) # (b*num_agents*num_agents, c, h, w)
            val_mat = val_mat.view(b, num_agents, num_agents, c, h, w).contiguous()

            # 3. Encode weights
            val_mat = val_mat.view(b, num_agents, num_agents, c, h, w).contiguous() # (b, k_agents, q_agents, c, h, w)
            query_mat = feat_map.view(b, num_agents, c, h, w).unsqueeze(1).expand(-1, num_agents, -1, -1, -1, -1)   # (b*num_agents, c, h, w) --> (b, k_agents, q_agents, c, h, w)

            weight_mat = torch.cat([query_mat, val_mat], dim=3).view(b*num_agents*num_agents, c*2, h, w)
            weight_mat = eval('self.weight_net'+str(trans_layer))(weight_mat)    # (b*k_agents*q_agents, c, h, w)
            weight_mat = weight_mat.view(b, num_agents, num_agents, 1, h, w).softmax(dim=1)
            feat_fuse = (weight_mat * val_mat).sum(dim=1)    # (b, num_agents, c, h, w)

            # 4. Return fused feat
            post_commu_feats = feat_fuse.view(b*num_agents, c, h, w).contiguous() * 0.5 + feat_map * 0.5
            x[trans_layer] = post_commu_feats
            weight_mats.append(weight_mat)
            val_mats.append(val_mat)
        return x, weight_mats, val_mats
    
    def GLOBAL_MESSAGE(self, images, trans_mats):
        b, num_agents, img_c, img_h, img_w = images.size()
        images = images.view(b*num_agents, img_c, img_h, img_w)

        # Encoder
        x = self.base(images)
        x = self.dla_up(x)  # list [(B, 64, 112, 200), (B, 128, 56, 100), (B, 256, 28, 50), (B, 512, 14, 25)] B = b * num_agents

        if self.trans_layer[-1] == -2:
            return x, []

        weight_mats = []
        val_mats = []
        for trans_layer in self.trans_layer:
            # 1. Choose the layer
            feat_map = x[trans_layer]
            _, c, h, w = feat_map.size()

            # 2. Get the value mat (trans feature to global coord)  # val_mat: (b, k_agents, q_agents, c, h, w)
            # uav_i --> global coord
            trans_mats = trans_mats.view(b*num_agents, 3, 3)
            worldgrid2worldcoord_mat = torch.Tensor(np.array([[500/w, 0, -200], [0, 500/h, -250], [0, 0, 1]])).to(trans_mats.device)
            feat_zoom_mats = torch.Tensor(np.array(np.diag([2**(trans_layer+2), 2**(trans_layer+2), 1]), dtype=np.float32)).to(trans_mats.device)
            cur_trans_mats = torch.inverse(trans_mats @ worldgrid2worldcoord_mat) @ feat_zoom_mats
            global_feat = kornia.warp_perspective(feat_map, cur_trans_mats, dsize=(h, w)).view(b, num_agents, c, h, w).contiguous() # (b*num_agents, c, h, w)
            val_mat = global_feat.unsqueeze(2).expand(-1, -1, num_agents, -1, -1, -1).contiguous()

            # 3. Encode weights
            query_mat = global_feat.unsqueeze(1).expand(-1, num_agents, -1, -1, -1, -1)   # (b*num_agents, c, h, w) --> (b, k_agents, q_agents, c, h, w)
            weight_mat = torch.cat([query_mat, val_mat], dim=3).view(b*num_agents*num_agents, c*2, h, w)
            weight_mat = eval('self.weight_net'+str(trans_layer))(weight_mat)    # (b*k_agents*q_agents, c, h, w)
            weight_mat = weight_mat.view(b, num_agents, num_agents, 1, h, w).softmax(dim=1)
            feat_fuse = (weight_mat * val_mat).sum(dim=1)    # (b, num_agents, c, h, w)
            
            # print(prob_action)
            # 4. Return fused feat
            # global coord --> uav_j   
            trans_mats_inverse = torch.inverse(cur_trans_mats).view(b, num_agents, 3, 3).contiguous().view(b*num_agents, 3, 3).contiguous()
            feat_fuse = kornia.warp_perspective(feat_fuse.view(b*num_agents, c, h, w).contiguous(), trans_mats_inverse, dsize=(h, w)) # (b*num_agents*num_agents, c, h, w)
            post_commu_feats = feat_fuse * 0.5 + feat_map * 0.5
            x[trans_layer] = post_commu_feats
            weight_mats.append(weight_mat)
            val_mats.append(val_mat)
        return x, weight_mats, val_mats
    
    def SINGLE_GLOBAL_MESSAGE(self, images, trans_mats):
        b, num_agents, img_c, img_h, img_w = images.size()
        images = images.view(b*num_agents, img_c, img_h, img_w)

        # Encoder
        x = self.base(images)
        x = self.dla_up(x)  # list [(B, 64, 112, 200), (B, 128, 56, 100), (B, 256, 28, 50), (B, 512, 14, 25)] B = b * num_agents

        if self.trans_layer[-1] == -2:
            return x

        for trans_layer in self.trans_layer:
            # 1. Choose the layer
            feat_map = x[trans_layer]
            _, c, h, w = feat_map.size()

            # 2. Get the value mat (trans feature to global coord)  # val_mat: (b, k_agents, q_agents, c, h, w)
            # uav_i --> global coord
            trans_mats = trans_mats.view(b*num_agents, 3, 3)
            worldgrid2worldcoord_mat = torch.Tensor(np.array([[500/w, 0, -200], [0, 500/h, -250], [0, 0, 1]])).to(trans_mats.device)
            feat_zoom_mats = torch.Tensor(np.array(np.diag([2**(trans_layer+2), 2**(trans_layer+2), 1]), dtype=np.float32)).to(trans_mats.device)
            cur_trans_mats = torch.inverse(trans_mats @ worldgrid2worldcoord_mat) @ feat_zoom_mats
            global_feat = kornia.warp_perspective(feat_map, cur_trans_mats, dsize=(h, w)) # (b*num_agents, c, h, w)
            val_mat = global_feat.view(b, num_agents, c, h, w).contiguous()
            # 3. Encode message (in global coord)
            feat_fuse = val_mat.max(dim=1)[0].unsqueeze(1)
            feat_fuse = feat_fuse.expand(-1, num_agents, -1, -1, -1).contiguous().view(b*num_agents, c, h, w).contiguous()
            # 4. Return fused feat
            # global coord --> uav_j 
            trans_mats_inverse = torch.inverse(cur_trans_mats).view(b, num_agents, 3, 3).contiguous().view(b*num_agents, 3, 3).contiguous()
            feat_fuse = kornia.warp_perspective(feat_fuse.view(b*num_agents, c, h, w).contiguous(), trans_mats_inverse, dsize=(h, w)) # (b*num_agents, c, h, w)
            post_commu_feats = feat_map * 0.5 + feat_fuse * 0.5
            x[trans_layer] = post_commu_feats
        return x

    def LocalCoord_forward(self, images, trans_mats):
        weight_mats = None
        val_mats = None
        if self.message_mode == 'NO_MESSAGE_NOWARP':
            x = self.NO_MESSAGE_NOWARP(images)
        elif self.message_mode == 'NO_MESSAGE':
            x = self.NO_MESSAGE(images, trans_mats)
        elif self.message_mode == 'LOCAL_MESSAGE_NOWARP':
            x, weight_mats, val_mats = self.LOCAL_MESSAGE_NOWARP(images)
        elif self.message_mode == 'LOCAL_MESSAGE':
            x, weight_mats, val_mats = self.LOCAL_MESSAGE(images, trans_mats)
        elif self.message_mode == 'GLOBAL_MESSAGE':
            x, weight_mats, val_mats = self.GLOBAL_MESSAGE(images, trans_mats)
        elif self.message_mode == 'SINGLE_GLOBAL_MESSAGE':
            x = self.SINGLE_GLOBAL_MESSAGE(images, trans_mats)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(y[-1]) # (b*num_agent, 2, 112, 200)
        if weight_mats is not None:
            z['weight_mats'] = weight_mats
        if val_mats is not None:
            z['val_mats'] = val_mats
        return [z]
    
    def MEAN_MESSAGE(self, x, trans_layer=[0], with_residual=True):
        '''
        Mode: average the message
        Input: x shape [b, num_agents, c, h, w]
        '''
        for c_layer in trans_layer:
            # 1. Choose the layer
            feat_map = x[c_layer]
            # 2. Encode message (in global coord)
            feat_map_mask = torch.where(feat_map>0, torch.ones_like(feat_map).to(feat_map.device), torch.zeros_like(feat_map).to(feat_map.device))
            feat_fuse = feat_map.sum(dim=1) / (feat_map_mask.sum(dim=1)+1e-6)
            if with_residual:
                feat_fuse = feat_fuse.unsqueeze(1)
                x[c_layer] = x[c_layer] * 0.5 + feat_fuse * 0.5
            else:
                x[c_layer] = feat_fuse
        return x
    
    def MAX_MESSAGE(self, x, trans_layer=[0], with_residual=True):
        '''
        Mode: average the message
        Input: x shape [b, num_agents, c, h, w]
        '''
        for c_layer in trans_layer:
            # 1. Choose the layer
            feat_map = x[c_layer]
            # 2. Encode message (in global coord)
            feat_fuse = feat_map.max(dim=1)[0]
            if with_residual:
                feat_fuse = feat_fuse.unsqueeze(1)
                x[c_layer] = x[c_layer] * 0.5 + feat_fuse * 0.5
            else:
                x[c_layer] = feat_fuse
        return x
    
    def WHEN2COM_MESSAGE(self, x, trans_layer=[0], with_residual=True):
        b, num_agents, c, h, w = x[-1].shape
        qk_feat = x[-1].view(b*num_agents, c, h, w).contiguous()
        query_key_maps = self.query_key_net(qk_feat) # (b*num_agents, c, h, w)
        querys = self.query_net(query_key_maps) # (b*num_agents, query_size)
        keys = self.key_net(query_key_maps) # (b*num_agents, key_size)
        query_mat = querys.view(b, num_agents, self.query_size) # (b, num_agents, query_size)
        key_mat = keys.view(b, num_agents, self.key_size) # (b, num_agents, key_size)
        for c_layer in trans_layer:
            val_mat = x[c_layer]
            feat_fuse, prob_action = self.attention_net(query_mat, key_mat, val_mat)    # (b, num_agents, c, h, w)
            if with_residual:
                x[c_layer] = feat_fuse * 0.5 + x[c_layer] * 0.5
            else:
                x[c_layer] = feat_fuse
        # print(prob_action)
        return x
    
    def V2V_MESSAGE(self, x, trans_layer=[0], with_residual=True):
        for c_layer in trans_layer:
            b, num_agents, c, h, w = x[c_layer].shape
            for _ in range(self.gnn_iter_num):
                mean_feat = x[c_layer].sum(dim=1).unsqueeze(1).expand(-1, num_agents, -1, -1, -1) # (b, num_agents, c, h, w)
                mean_feat = (mean_feat - x[c_layer])/4.0
                cat_feat = torch.cat([x[c_layer], mean_feat], dim=2) # (b, num_agents, 2*c, h, w)
                cat_feat = cat_feat.view(b*num_agents, 1, 2*c, h, w).contiguous()
                updated_feat, _ = eval('self.convgru'+str(c_layer))(cat_feat, None)
                updated_feat = updated_feat.view(b, num_agents, c, h, w).contiguous()  # (b, num_agents, c, h, w)
                x[c_layer] = updated_feat
        return x
    
    def POINTWISE_MESSAGE(self, x, trans_layer=[0], with_residual=True):
        for c_layer in trans_layer:
            b, num_agents, c, h, w = x[c_layer].shape
            val_mat = x[c_layer].unsqueeze(2).expand(-1, -1, num_agents, -1, -1, -1).contiguous()
            query_mat = x[c_layer].unsqueeze(1).expand(-1, num_agents, -1, -1, -1, -1)   # (b*num_agents, c, h, w) --> (b, k_agents, q_agents, c, h, w)
            weight_mat = torch.cat([query_mat, val_mat], dim=3).view(b*num_agents*num_agents, c*2, h, w)
            weight_mat = eval('self.weight_net'+str(c_layer))(weight_mat)    # (b*k_agents*q_agents, c, h, w)
            weight_mat = F.relu(weight_mat).view(b, num_agents, num_agents, 1, h, w).softmax(dim=1)
            feat_fuse = (weight_mat * val_mat).sum(dim=1)    # (b, num_agents, c, h, w)
            if with_residual:
                x[c_layer] = feat_fuse * 0.5 + x[c_layer] * 0.5
            else:
                x[c_layer] = feat_fuse
        return x

    def GlobalCoord_forward(self, images, trans_mats):
        b, num_agents, img_c, img_h, img_w = images.size()
        images = images.view(b*num_agents, img_c, img_h, img_w)

        warp_image = True if len(self.trans_layer) == 1 and self.trans_layer[-1] == -1 else False
        
        if warp_image:
            print('Warp image')
            trans_mats = trans_mats.view(b*num_agents, 3, 3)
            # worldgrid2worldcoord_mat = torch.Tensor(np.array([[500/img_w, 0, -200], [0, 500/img_h, -250], [0, 0, 1]])).to(trans_mats.device)
            worldgrid2worldcoord_mat = torch.Tensor(np.array([[1/img_w, 0, 0], [0, 1/img_h, 0], [0, 0, 1]])).to(trans_mats.device)
            cur_trans_mats = torch.inverse(trans_mats @ worldgrid2worldcoord_mat)
            images = kornia.warp_perspective(images, cur_trans_mats, dsize=(img_h, img_w))
        
        # Encoder
        x = self.base(images)
        x = self.dla_up(x)  # list [(B, 64, 112, 200), (B, 128, 56, 100), (B, 256, 28, 50), (B, 512, 14, 25)] B = b * num_agents

        scale = 4
        if warp_image:
            global_x = x
        else:
            global_x = []
            trans_mats_list = []
            trans_mats = trans_mats.view(b*num_agents, 3, 3)
            for c_layer, feat_map in enumerate(x):
                _, c, h, w = feat_map.size()

                # 2. Get the value mat (trans feature to global coord)  # val_mat: (b, k_agents, q_agents, c, h, w)
                # uav_i --> global coord
                # worldgrid2worldcoord_mat = torch.Tensor(np.array([[500/(w*scale), 0, -200], [0, 500/(h*scale), -250], [0, 0, 1]])).to(trans_mats.device)
                worldgrid2worldcoord_mat = torch.Tensor(np.array([[1/(w*scale), 0, 0], [0, 1/(h*scale), 0], [0, 0, 1]])).to(trans_mats.device)
                feat_zoom_mats = torch.Tensor(np.array(np.diag([2**(c_layer+2), 2**(c_layer+2), 1]), dtype=np.float32)).to(trans_mats.device)
                cur_trans_mats = torch.inverse(trans_mats @ worldgrid2worldcoord_mat).contiguous() @ feat_zoom_mats
                global_feat = kornia.warp_perspective(feat_map, cur_trans_mats, dsize=(h*scale, w*scale)) # (b*num_agents, c, h, w)
                # global_feat = kornia.resize(global_feat, size=(h*4, w*4))
                global_x.append(global_feat)
                trans_mats_list.append(cur_trans_mats)
        
        #########################################################################
        #        Merge the feature of multi-agents (with bandwidth cost)        #
        #########################################################################
        if self.trans_layer[-1] == -2:
            pass
        else:
            for c_layer, feat_map in enumerate(global_x):
                _, c, h, w = feat_map.shape
                feat_map = feat_map.view(b, num_agents, c, h, w)
                global_x[c_layer] = feat_map
            if self.message_mode == 'Mean':
                global_x = self.MEAN_MESSAGE(global_x, self.trans_layer)
            elif self.message_mode == 'Max':
                global_x = self.MAX_MESSAGE(global_x, self.trans_layer)
            elif self.message_mode == 'When2com':
                global_x = self.WHEN2COM_MESSAGE(global_x, self.trans_layer)
            elif self.message_mode == 'V2V':
                global_x = self.V2V_MESSAGE(global_x, self.trans_layer)
            elif self.message_mode == 'Pointwise':
                global_x = self.POINTWISE_MESSAGE(global_x, self.trans_layer)
            
            for c_layer, feat_map in enumerate(global_x):
                b, num_agents, c, h, w = feat_map.shape
                feat_map = feat_map.view(b*num_agents, c, h, w)
                global_x[c_layer] = feat_map

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(global_x[i].clone())
        self.ida_up(y, 0, len(y))

        _, _, h, w = y[-1].shape
        # trans_mats_inverse = torch.inverse(trans_mats_list[0]).view(b, num_agents, 3, 3).contiguous().view(b*num_agents, 3, 3).contiguous()
        global_z = {}
        z = {}
        for head in self.heads:
            global_z[head] = self.__getattr__(head)(y[-1]) # (b*num_agent, 2, 112, 200)
            # z[head] = kornia.warp_perspective(global_z[head], trans_mats_inverse, dsize=(h, w)) # (b*num_agents*num_agents, c, h, w)
        # return [z]
        return [global_z]
    
    def forward(self, images, trans_mats):
        if self.coord == 'Global':
            return self.GlobalCoord_forward(images, trans_mats)
        else:
            return self.LocalCoord_forward(images, trans_mats)
        

def get_pose_net(num_layers, heads, head_conv=256, down_ratio=4, message_mode='NO_MESSAGE_NOWARP', trans_layer=[3], coord='Local'):
    model = DLASeg('dla{}'.format(num_layers), heads,
                    pretrained=True,
                    down_ratio=down_ratio,
                    final_kernel=1,
                    last_level=5,
                    head_conv=head_conv,
                    message_mode=message_mode,
                    trans_layer=trans_layer,
                    coord=coord)
    return model

