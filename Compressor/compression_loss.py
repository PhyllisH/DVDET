import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
from loguru import logger
from utils.v2xsim_utils.detection_util import (bev_box_decode_torch,
                                               center_to_corner_box2d_torch)

from loss.base_loss import (SoftmaxFocalClassificationLoss,
                            WeightedSmoothL1LocalizationLoss)


class CompressionLoss(nn.Module):
    def __init__(self, args):
        super(CompressionLoss, self).__init__()
        self.cls_criteria = SoftmaxFocalClassificationLoss()
        # self.loc_criteria = WeightedSmoothL1LocalizationLoss()
        self.weight_cls = args['weight_cls']
        self.weight_loc = args['weight_loc']
        self.weight_rate = args['weight_rate']
        self.loss_dict = {}

    def forward(self, output_dict, target_dict):
        labels = target_dict['labels']  # [20, 256, 256, 6, 2] binary, gt for cls
        labels = labels.view(output_dict['cls'].shape[0],-1,output_dict['cls'].shape[-1])
        N = labels.shape[0]

        reg_targets = target_dict['reg_targets']  # [20, 256, 256, 6, 1, 6]
        reg_loss_mask = target_dict['reg_loss_mask']
        anchors = target_dict['anchors']

        cls_loss = torch.sum(self.cls_criteria(output_dict['cls'], labels)) / N
        loc_loss = self.corner_loss(anchors,reg_loss_mask,reg_targets,output_dict['loc'])

        _, _, H, W = output_dict['size']
        num_pixels = N * H * W
        rate_loss = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output_dict["likelihoods"].values()
        )

        total_loss = self.weight_cls * cls_loss + self.weight_loc * loc_loss + self.weight_rate * rate_loss
        self.loss_dict.update({'total_loss': total_loss,
                               'reg_loss': loc_loss,
                               'cls_loss': cls_loss,
                               'comp_loss': rate_loss})

        return total_loss

    def logging(self, epoch, batch_id, batch_len, writer):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict['total_loss'].item()
        reg_loss = self.loss_dict['reg_loss'].item()
        cls_loss = self.loss_dict['cls_loss'].item()

        # print("[epoch %d][%d/%d], || Loss: %.4f || cls Loss: %.4f"
        #       " || reg Loss: %.4f" % (
        #           epoch, batch_id + 1, batch_len,
        #           total_loss, cls_loss, reg_loss))

        writer.add_scalar('Regression_loss', reg_loss,
                          epoch * batch_len + batch_id)
        writer.add_scalar('Confidence_loss', cls_loss,
                          epoch * batch_len + batch_id)
        if batch_id % 10 == 0:
            logger.info(f"[epoch {epoch}][{batch_id + 1}/{batch_len}] Total Loss: {total_loss:.4f} \t Cls Loss:{cls_loss:.4f} \t Reg Loss:{reg_loss:.4f}")


    def corner_loss(self,anchors,reg_loss_mask,reg_targets,pred_result):
        # anchors [20, 256, 256, 6, 6]
        # reg_loss_mask [20, 256, 256, 6, 1]
        # reg_targets [20, 256, 256, 6, 1, 6]
        # pred_result [20, 256, 256, 6, 1, 6]

        N = pred_result.shape[0]
        anchors = anchors.unsqueeze(-2).expand(anchors.shape[0],anchors.shape[1],anchors.shape[2],anchors.shape[3],reg_loss_mask.shape[-1],anchors.shape[-1])
        # print("anchors:",anchors.shape)
        # anchors [20, 256, 256, 6, 1, 6]
        assigned_anchor = anchors[reg_loss_mask]
        assigned_target = reg_targets[reg_loss_mask]
        assigned_pred = pred_result[reg_loss_mask]
        # print(assigned_anchor.shape,assigned_pred.shape,assigned_target.shape) # [79872, 6] [79872, 6] [79872, 6] , 79872 is not a constant, is the sum of a batch of reg_loss_mask num
        pred_decode = bev_box_decode_torch(assigned_pred,assigned_anchor)
        target_decode = bev_box_decode_torch(assigned_target,assigned_anchor)
        pred_corners = center_to_corner_box2d_torch(pred_decode[...,:2],pred_decode[...,2:4],pred_decode[...,4:])
        target_corners = center_to_corner_box2d_torch(target_decode[...,:2],target_decode[...,2:4],target_decode[...,4:])
        loss_loc = torch.sum(torch.norm(pred_corners-target_corners,dim=-1)) / N

        return loss_loc
