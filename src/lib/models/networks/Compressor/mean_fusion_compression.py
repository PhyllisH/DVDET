from .sub_modules.detection_head import ClassificationHead, SingleRegressionHead
from .sub_modules.lidar_module import lidar_encoder, lidar_decoder
from .sub_modules.compressor import ScaleHyperprior
from utils.v2xsim_utils.pose_util import pose2tfv, vec2matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from icecream import ic


class MeanFusionCompression(nn.Module):
    def __init__(self, args):
        super(MeanFusionCompression, self).__init__()
        self.out_seq_len = 1
        self.box_code_size = 6
        self.category_num = 2
        self.anchor_num_per_loc = 6

        self.classification = ClassificationHead(args)
        self.regression = SingleRegressionHead(args)
        self.encoder = lidar_encoder(height_feat_size=13)
        self.decoder = lidar_decoder(height_feat_size=13)
        self.compressor = ScaleHyperprior(N=128, M=192)

        self.train_flag = True
        self.layer = args['fusion_layer']
        self.agent_num = args["num_agent"]
        self.kd_flag = args["kd_flag"]
        self.compress_flag = args["compress_flag"]

    def agents2batch(self, feats):
        num_agent = feats.shape[1]
        feat_list = []
        for i in range(num_agent):
            feat_list.append(feats[:, i, :, :, :])
        feat_mat = torch.cat(tuple(feat_list), 0)
        return feat_mat

    def forward(self, data_dict):
        # we need is bevs, pose and num_agent_tensor
        pose = data_dict["pose"]
        pose3d = data_dict["pose3d"]
        bevs = data_dict["bev_seq"]
        num_agent_real = data_dict["num_agent_real"]
        trans_mats = data_dict["trans_matrices"]
        batch_size = pose.shape[0]

        trans_vecs = pose2tfv(pose)

        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)
        x_0, x_1, x_2, x_3, x_4 = self.encoder(bevs)
        device = bevs.device

        if self.layer == 4:
            feat_maps = x_4
            size = (1, 512, 16, 16)
        elif self.layer == 3:
            feat_maps = x_3
            size = (1, 256, 32, 32)
        elif self.layer == 2:
            feat_maps = x_2
            size = (1, 128, 64, 64)
        elif self.layer == 1:
            feat_maps = x_1
            size = (1, 64, 128, 128)
        elif self.layer == 0:
            feat_maps = x_0
            size = (1, 32, 256, 256)

        # print(feat_maps.shape, x_3.shape, x_2.shape, x_1.shape)

        # get feat maps for each agent [10 512 16 16] -> [2 5 512 16 16]
        feat_map = {}
        feat_list = []

        for i in range(self.agent_num):
            feat_map[i] = torch.unsqueeze(feat_maps[batch_size * i:batch_size * (i + 1)], 1)
            feat_list.append(feat_map[i])

        local_com_mat = torch.cat(tuple(feat_list), 1)  # [2 5 512 16 16] [batch, agent, channel, height, width]
        local_com_mat_update = torch.cat(tuple(feat_list), 1)  # to avoid the inplace operation
        feat_comm = feat_maps

        # compression
        if self.compress_flag:
            if self.train_flag:
                comp_out = self.compressor(feat_comm)
                recon_feat = comp_out["x_hat"]
                likelihoods = comp_out["likelihoods"]
                recon_map = {}
                recon_list = []
                for i in range(self.agent_num):
                    recon_map[i] = torch.unsqueeze(recon_feat[batch_size * i:batch_size * (i + 1)], 1)
                    recon_list.append(feat_map[i])
                local_com_mat_comm = torch.cat(tuple(recon_list), 1)
            else:
                comp_out = self.compressor.compress(feat_comm)
                strings = comp_out["strings"]
                shape = comp_out["shape"]
                comm_size = len(strings[0][0]) + len(strings[1][0])
                recon_feat = self.compressor.decompress(strings, shape)["x_hat"]
                recon_map = {}
                recon_list = []
                for i in range(self.agent_num):
                    recon_map[i] = torch.unsqueeze(recon_feat[batch_size * i:batch_size * (i + 1)], 1)
                    recon_list.append(feat_map[i])
                local_com_mat_comm = torch.cat(tuple(recon_list), 1)
        else:
            local_com_mat_comm = copy.deepcopy(local_com_mat)

        for b in range(batch_size):
            num_agent = num_agent_real[b, 0]
            for i in range(num_agent):
                tg_agent = local_com_mat[b, i]
                all_warp = trans_vecs[b, i]  # transformation [2 5 5 4 4]
                all_warp = vec2matrix(all_warp)

                neighbor_feat_list = list()
                neighbor_feat_list.append(tg_agent)
                for j in range(num_agent):
                    if j != i:
                        nb_agent = torch.unsqueeze(local_com_mat_comm[b, j], 0)  # [1 512 16 16]
                        nb_warp = all_warp[j]  # [4 4]
                        # normalize the translation vector
                        x_trans = (4 * nb_warp[0, 2]) / 128
                        y_trans = -(4 * nb_warp[1, 2]) / 128

                        theta_rot = torch.tensor(
                            [[nb_warp[0, 0], nb_warp[0, 1], 0.0], [nb_warp[1, 0], nb_warp[1, 1], 0.0]]).type(
                            dtype=torch.float).to(device)
                        theta_rot = torch.unsqueeze(theta_rot, 0)
                        grid_rot = F.affine_grid(theta_rot, size=torch.Size(size))  # 得到grid 用于grid sample

                        theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans]]).type(
                            dtype=torch.float).to(device)
                        theta_trans = torch.unsqueeze(theta_trans, 0)
                        grid_trans = F.affine_grid(theta_trans, size=torch.Size(size))  # 得到grid 用于grid sample

                        # first rotate the feature map, then translate it
                        warp_feat_rot = F.grid_sample(nb_agent, grid_rot, mode='bilinear')
                        warp_feat_trans = F.grid_sample(warp_feat_rot, grid_trans, mode='bilinear')
                        warp_feat = torch.squeeze(warp_feat_trans)
                        neighbor_feat_list.append(warp_feat)

                # mean fusion
                mean_feat = torch.mean(torch.stack(neighbor_feat_list), dim=0)  # [c, h, w]
                # feature update
                local_com_mat_update[b, i] = mean_feat

        # weighted feature maps is passed to decoder
        feat_fuse_mat = self.agents2batch(local_com_mat_update)

        if self.kd_flag == 1:
            if self.layer == 4:
                x_8, x_7, x_6, x_5 = self.decoder(x_0, x_1, x_2, x_3, feat_fuse_mat, batch_size, kd_flag=self.kd_flag)
            elif self.layer == 3:
                x_8, x_7, x_6, x_5 = self.decoder(x_0, x_1, x_2, feat_fuse_mat, x_4, batch_size, kd_flag=self.kd_flag)
            elif self.layer == 2:
                x_8, x_7, x_6, x_5 = self.decoder(x_0, x_1, feat_fuse_mat, x_3, x_4, batch_size, kd_flag=self.kd_flag)
            elif self.layer == 1:
                x_8, x_7, x_6, x_5 = self.decoder(x_0, feat_fuse_mat, x_2, x_3, x_4, batch_size, kd_flag=self.kd_flag)
            elif self.layer == 0:
                x_8, x_7, x_6, x_5 = self.decoder(feat_fuse_mat, x_1, x_2, x_3, x_4, batch_size, kd_flag=self.kd_flag)
            x = x_8
        else:
            if self.layer == 4:
                x = self.decoder(x_0, x_1, x_2, x_3, feat_fuse_mat, batch_size, kd_flag=self.kd_flag)
            elif self.layer == 3:
                x = self.decoder(x_0, x_1, x_2, feat_fuse_mat, x_4, batch_size, kd_flag=self.kd_flag)
            elif self.layer == 2:
                x = self.decoder(x_0, x_1, feat_fuse_mat, x_3, x_4, batch_size, kd_flag=self.kd_flag)
            elif self.layer == 1:
                x = self.decoder(x_0, feat_fuse_mat, x_2, x_3, x_4, batch_size, kd_flag=self.kd_flag)
            elif self.layer == 0:
                x = self.decoder(feat_fuse_mat, x_1, x_2, x_3, x_4, batch_size, kd_flag=self.kd_flag)

        # vis = vis.permute(0, 3, 1, 2)
        # if not maps is None:
        #     x = torch.cat([x,maps],axis=-1)
        # if not vis is None:
        #     x = torch.cat([x,vis],axis=1)

        # Cell Classification head
        cls_preds = self.classification(x)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.view(cls_preds.shape[0], -1, self.category_num)

        # Detection head
        loc_preds = self.regression(x)
        loc_preds = loc_preds.permute(0, 2, 3, 1).contiguous()
        loc_preds = loc_preds.view(-1, loc_preds.size(1), loc_preds.size(2), self.anchor_num_per_loc, self.out_seq_len,
                                   self.box_code_size)
        # loc_pred (N * T * W * H * loc)

        # no_kd here
        if self.compress_flag:
            if self.train_flag:
                result = {'loc': loc_preds,
                          'cls': cls_preds,
                          'likelihoods': likelihoods,
                          'size': size}
            else:
                result = {'loc': loc_preds,
                          'cls': cls_preds,
                          'comm_size': comm_size}
        else:
            result = {'loc': loc_preds,
                      'cls': cls_preds}

        return result

