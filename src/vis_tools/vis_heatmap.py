import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import seaborn as sns
from torch import ones

# root_dir = '/GPFS/data/yhu/code/BEV_Det/CoDet/src/lib/models/networks/qualitymap_attn_weights_comm_show/5254_0'
# root_dir = '/GPFS/data/yhu/code/BEV_Det/CoDet/src/lib/models/networks/qualitymap_attn_weights_comm_show/3991_0'
# root_dir = '/GPFS/data/yhu/code/BEV_Det/CoDet/src/lib/models/networks/qualitymap_attn_weights_comm_show/6695_0'

# 0.1
# root_dir = '/GPFS/data/yhu/code/BEV_Det/CoDet/src/lib/models/networks/qualitymap_attn_weights_comm_show/0814_0'
# 0.3
# root_dir = '/GPFS/data/yhu/code/BEV_Det/CoDet/src/lib/models/networks/qualitymap_attn_weights_comm_show/1054_0'
root_dir = '/GPFS/data/yhu/code/BEV_Det/CoDet/src/lib/models/networks/qualitymap_attn_weights_comm_show/0134_0'

weight_mats = np.load(os.path.join(root_dir, 'weight_mats.npy'))
quality_map = np.load(os.path.join(root_dir, 'quality_map.npy'))
communication_mask = np.load(os.path.join(root_dir, 'communication_mask.npy'))
cur_images = np.load(os.path.join(root_dir, 'cur_images.npy'))
val_feats = np.load(os.path.join(root_dir, 'val_feat.npy'))

sns.set(font_scale = 2)
b, k_agents, q_agents, _, _, _ = cur_images.shape
for i in range(b):
    j = 4
    for k in [1,2,4]:
        # flag = True if k==4 else False
        flag = False
        # flag = True
        fig = plt.figure()
        if j == k:
            ax = sns.heatmap(weight_mats[i,k,j][0], cbar=flag)
        else:
            ax = sns.heatmap((weight_mats[i,k,j]*communication_mask[i,k,j])[0],cbar=flag) # (b, q_agents, h, w)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        plt.savefig(os.path.join('4', 'weight_mats_{}.png'.format(k)))
        plt.close()
        
        fig = plt.figure()
        ax = sns.heatmap(quality_map[i,k,j][0],cbar=flag)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        plt.savefig(os.path.join('4', 'qualitymap_{}.png'.format(k)))
        plt.close()

        fig = plt.figure()
        ax = sns.heatmap(1-quality_map[i,k,j][0],cbar=flag)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        plt.savefig(os.path.join('4', 'requestmap_{}.png'.format(k)))
        plt.close()

        # fig = plt.figure()
        # ax = sns.heatmap(communication_mask[i,k,j][0],cbar=flag)
        # ax.set_xticks([])
        # ax.set_yticks([])
        # fig.tight_layout()
        # plt.savefig(os.path.join('4', 'communication_mask_{}.png'.format(k)))
        # plt.close()

        fig = plt.figure()
        # ax = sns.heatmap(communication_mask[i,k,j][0]*(1-communication_mask[i,1,j][0]),cbar=flag)
        ax = sns.heatmap(communication_mask[i,k,j][0],cbar=flag)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        plt.savefig(os.path.join('4', 'communication_mask_{}.png'.format(k)))
        plt.close()

        fig = plt.figure()
        ax = sns.heatmap(communication_mask[i,k,j][0]*val_feats[i,k,j][0],cbar=flag)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        plt.savefig(os.path.join('4', 'val_feat_{}.png'.format(k)))
        plt.close()

        fig, ax = plt.subplots(1, 1)
        # image_mask = cur_images[i,k,j].max(axis=0)  # (h,w)
        # ones_mask = (image_mask > 0) * 1
        # ones_mask = ones_mask[None,].repeat(3,axis=0)
        # cur_images[i,k,j] = np.where(ones_mask, cur_images[i,k,j], 1-ones_mask)
        ax.imshow((cur_images[i,k,j].transpose(1,2,0) * 255.).astype('uint8')) # (b, q_agents, h, w)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        plt.savefig(os.path.join('4', 'image_{}.png'.format(k)))
        plt.close()


# b, k_agents, q_agents, _, _, _ = cur_images.shape
# for i in range(b):
#     for j in range(q_agents):
#         if j != 4:
#             pass
#         fig, axes = plt.subplots(4, 5, figsize=(20,8))
#         for k in range(k_agents):
#             if j == k:
#                 axes[0,k].imshow((weight_mats[i,k,j].transpose(1,2,0) * 255.).astype('uint8')) # (b, q_agents, h, w)
#             else:
#                 axes[0,k].imshow(((weight_mats[i,k,j]*communication_mask[i,k,j]).transpose(1,2,0) * 255.).astype('uint8')) # (b, q_agents, h, w)
#             axes[0,k].set_xticks([])
#             axes[0,k].set_yticks([])

#             axes[1,k].imshow((quality_map[i,k,j].transpose(1,2,0) * 255.).astype('uint8')) # (b, q_agents, h, w)
#             axes[1,k].set_xticks([])
#             axes[1,k].set_yticks([])

#             axes[2,k].imshow((communication_mask[i,k,j].transpose(1,2,0) * 255.).astype('uint8')) # (b, q_agents, h, w)
#             axes[2,k].set_xticks([])
#             axes[2,k].set_yticks([])

#             axes[3,k].imshow((cur_images[i,k,j].transpose(1,2,0) * 255.).astype('uint8')) # (b, q_agents, h, w)
#             axes[3,k].set_xticks([])
#             axes[3,k].set_yticks([])
#         fig.tight_layout()
#         plt.savefig('{}.png'.format(j))
#         plt.close()
