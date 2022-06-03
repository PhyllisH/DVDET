#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import math
from kornia.geometry.transform.imgwarp import warp_affine

import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

import _ext as _backend


class _DCNv2(Function):
    @staticmethod
    def forward(ctx, input, offset, mask, weight, bias, stride, padding, dilation, deformable_groups):
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.kernel_size = _pair(weight.shape[2:4])
        ctx.deformable_groups = deformable_groups
        output = _backend.dcn_v2_forward(
            input,
            weight,
            bias,
            offset,
            mask,
            ctx.kernel_size[0],
            ctx.kernel_size[1],
            ctx.stride[0],
            ctx.stride[1],
            ctx.padding[0],
            ctx.padding[1],
            ctx.dilation[0],
            ctx.dilation[1],
            ctx.deformable_groups,
        )
        ctx.save_for_backward(input, offset, mask, weight, bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input, grad_offset, grad_mask, grad_weight, grad_bias = _backend.dcn_v2_backward(
            input,
            weight,
            bias,
            offset,
            mask,
            grad_output,
            ctx.kernel_size[0],
            ctx.kernel_size[1],
            ctx.stride[0],
            ctx.stride[1],
            ctx.padding[0],
            ctx.padding[1],
            ctx.dilation[0],
            ctx.dilation[1],
            ctx.deformable_groups,
        )

        return (
            grad_input,
            grad_offset,
            grad_mask,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
        )


dcn_v2_conv = _DCNv2.apply


class DCNv2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        deformable_groups=1,
    ):
        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def forward(self, input, offset, mask):
        assert 2 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] == offset.shape[1]
        assert self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] == mask.shape[1]
        return dcn_v2_conv(
            input,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.deformable_groups,
        )


class DCN(DCNv2):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        deformable_groups=1,
    ):
        super(DCN, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, deformable_groups)

        channels_ = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            channels_,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return dcn_v2_conv(
            input,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.deformable_groups,
        )

class CF_DCN(DCNv2):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        deformable_groups=1,
    ):
        super(CF_DCN, self).__init__(in_channels+2, out_channels, kernel_size, stride, padding, dilation, deformable_groups)
        channels_ = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            channels_,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        b, c, h, w = input.shape
        h_i = torch.arange(h).unsqueeze(-1).expand(-1,w).unsqueeze(0) / h
        w_i = torch.arange(w).unsqueeze(0).expand(h,-1).unsqueeze(0) / w - 0.5
        # h_i = torch.arange(h).unsqueeze(-1).expand(-1,w).unsqueeze(0)
        # w_i = torch.arange(w).unsqueeze(0).expand(h,-1).unsqueeze(0)
        im_i = torch.cat([h_i, w_i], dim=0).to(input.device) # (2, h, w)
        input_with_index = torch.cat([input, im_i.unsqueeze(0).expand(b,-1,-1,-1)], dim=1)  # (b, c+2, h, w)
        # import ipdb; ipdb.set_trace()
        out = self.conv_offset_mask(input_with_index)  # (b, c, h, w)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        b, N, h, w = offset.shape
        n = N // 2
        k = int(math.sqrt(n))
        kh_i = torch.arange(-int(k//2), int(k//2)+1).unsqueeze(-1).expand(-1,k).reshape(1,n)
        kw_i = torch.arange(-int(k//2), int(k//2)+1).unsqueeze(0).expand(k,-1).reshape(1,n)
        k_i = torch.cat([kh_i, kw_i], dim=0)    # (2, 25)
        k_i = k_i.unsqueeze(-1).expand(-1, -1, h).unsqueeze(-1).expand(-1, -1, -1, w)  # (2, 25, h, w)
        im_i = im_i.unsqueeze(1).expand(-1, n, -1, -1)  # (2, 25, h, w)
        index_i = im_i + k_i.to(offset.device)
        index_i = index_i.reshape(N, h, w).unsqueeze(0).expand(b, -1, -1, -1) # (b, 50, h, w) or (b, 18, h, w)
        # import ipdb; ipdb.set_trace()
        index_i = index_i + offset
        return dcn_v2_conv(
            input_with_index,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.deformable_groups,
        ), index_i, mask

class EGOAWARE_DCN(DCNv2):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        deformable_groups=1,
    ):
        super(EGOAWARE_DCN, self).__init__(in_channels+2, out_channels, kernel_size, stride, padding, dilation, deformable_groups)
        channels_ = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels*2,
            channels_,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input, ego_input):
        b, c, h, w = input.shape
        # h_i = torch.arange(h).unsqueeze(-1).expand(-1,w).unsqueeze(0) / h
        # w_i = torch.arange(w).unsqueeze(0).expand(h,-1).unsqueeze(0) / w - 0.5
        # # h_i = torch.arange(h).unsqueeze(-1).expand(-1,w).unsqueeze(0)
        # # w_i = torch.arange(w).unsqueeze(0).expand(h,-1).unsqueeze(0)
        # im_i = torch.cat([h_i, w_i], dim=0).to(input.device) # (2, h, w)
        # input_with_index = torch.cat([input, im_i.unsqueeze(0).expand(b,-1,-1,-1)], dim=1)  # (b, c+2, h, w)
        input_with_index = torch.cat([input, ego_input], dim=1)  # (b, c*2, h, w)
        # import ipdb; ipdb.set_trace()
        out = self.conv_offset_mask(input_with_index)  # (b, c, h, w)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        # b, N, h, w = offset.shape
        # n = N // 2
        # k = int(math.sqrt(n))
        # kh_i = torch.arange(-int(k//2), int(k//2)+1).unsqueeze(-1).expand(-1,k).reshape(1,n)
        # kw_i = torch.arange(-int(k//2), int(k//2)+1).unsqueeze(0).expand(k,-1).reshape(1,n)
        # k_i = torch.cat([kh_i, kw_i], dim=0)    # (2, 25)
        # k_i = k_i.unsqueeze(-1).expand(-1, -1, h).unsqueeze(-1).expand(-1, -1, -1, w)  # (2, 25, h, w)
        # im_i = im_i.unsqueeze(1).expand(-1, n, -1, -1)  # (2, 25, h, w)
        # index_i = im_i + k_i.to(offset.device)
        # index_i = index_i.reshape(N, h, w).unsqueeze(0).expand(b, -1, -1, -1) # (b, 50, h, w) or (b, 18, h, w)
        # index_i = index_i + offset
        return dcn_v2_conv(
            input,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.deformable_groups,
        ), None, mask
class _DCNv2Pooling(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        rois,
        offset,
        spatial_scale,
        pooled_size,
        output_dim,
        no_trans,
        group_size=1,
        part_size=None,
        sample_per_part=4,
        trans_std=0.0,
    ):
        ctx.spatial_scale = spatial_scale
        ctx.no_trans = int(no_trans)
        ctx.output_dim = output_dim
        ctx.group_size = group_size
        ctx.pooled_size = pooled_size
        ctx.part_size = pooled_size if part_size is None else part_size
        ctx.sample_per_part = sample_per_part
        ctx.trans_std = trans_std

        output, output_count = _backend.dcn_v2_psroi_pooling_forward(
            input,
            rois,
            offset,
            ctx.no_trans,
            ctx.spatial_scale,
            ctx.output_dim,
            ctx.group_size,
            ctx.pooled_size,
            ctx.part_size,
            ctx.sample_per_part,
            ctx.trans_std,
        )
        ctx.save_for_backward(input, rois, offset, output_count)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, rois, offset, output_count = ctx.saved_tensors
        grad_input, grad_offset = _backend.dcn_v2_psroi_pooling_backward(
            grad_output,
            input,
            rois,
            offset,
            output_count,
            ctx.no_trans,
            ctx.spatial_scale,
            ctx.output_dim,
            ctx.group_size,
            ctx.pooled_size,
            ctx.part_size,
            ctx.sample_per_part,
            ctx.trans_std,
        )

        return grad_input, None, grad_offset, None, None, None, None, None, None, None, None


dcn_v2_pooling = _DCNv2Pooling.apply


class DCNv2Pooling(nn.Module):
    def __init__(
        self,
        spatial_scale,
        pooled_size,
        output_dim,
        no_trans,
        group_size=1,
        part_size=None,
        sample_per_part=4,
        trans_std=0.0,
    ):
        super(DCNv2Pooling, self).__init__()
        self.spatial_scale = spatial_scale
        self.pooled_size = pooled_size
        self.output_dim = output_dim
        self.no_trans = no_trans
        self.group_size = group_size
        self.part_size = pooled_size if part_size is None else part_size
        self.sample_per_part = sample_per_part
        self.trans_std = trans_std

    def forward(self, input, rois, offset):
        assert input.shape[1] == self.output_dim
        if self.no_trans:
            offset = input.new()
        return dcn_v2_pooling(
            input,
            rois,
            offset,
            self.spatial_scale,
            self.pooled_size,
            self.output_dim,
            self.no_trans,
            self.group_size,
            self.part_size,
            self.sample_per_part,
            self.trans_std,
        )


class DCNPooling(DCNv2Pooling):
    def __init__(
        self,
        spatial_scale,
        pooled_size,
        output_dim,
        no_trans,
        group_size=1,
        part_size=None,
        sample_per_part=4,
        trans_std=0.0,
        deform_fc_dim=1024,
    ):
        super(DCNPooling, self).__init__(
            spatial_scale,
            pooled_size,
            output_dim,
            no_trans,
            group_size,
            part_size,
            sample_per_part,
            trans_std,
        )

        self.deform_fc_dim = deform_fc_dim

        if not no_trans:
            self.offset_mask_fc = nn.Sequential(
                nn.Linear(self.pooled_size * self.pooled_size * self.output_dim, self.deform_fc_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.deform_fc_dim, self.deform_fc_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.deform_fc_dim, self.pooled_size * self.pooled_size * 3),
            )
            self.offset_mask_fc[4].weight.data.zero_()
            self.offset_mask_fc[4].bias.data.zero_()

    def forward(self, input, rois):
        offset = input.new()

        if not self.no_trans:

            # do roi_align first
            n = rois.shape[0]
            roi = dcn_v2_pooling(
                input,
                rois,
                offset,
                self.spatial_scale,
                self.pooled_size,
                self.output_dim,
                True,  # no trans
                self.group_size,
                self.part_size,
                self.sample_per_part,
                self.trans_std,
            )

            # build mask and offset
            offset_mask = self.offset_mask_fc(roi.view(n, -1))
            offset_mask = offset_mask.view(n, 3, self.pooled_size, self.pooled_size)
            o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
            offset = torch.cat((o1, o2), dim=1)
            mask = torch.sigmoid(mask)

            # do pooling with offset and mask
            return (
                dcn_v2_pooling(
                    input,
                    rois,
                    offset,
                    self.spatial_scale,
                    self.pooled_size,
                    self.output_dim,
                    self.no_trans,
                    self.group_size,
                    self.part_size,
                    self.sample_per_part,
                    self.trans_std,
                )
                * mask
            )
        # only roi_align
        return dcn_v2_pooling(
            input,
            rois,
            offset,
            self.spatial_scale,
            self.pooled_size,
            self.output_dim,
            self.no_trans,
            self.group_size,
            self.part_size,
            self.sample_per_part,
            self.trans_std,
        )
