'''
Author: yhu
Contact: phyllis1sjtu@outlook.com
LastEditTime: 2021-07-26 20:03:16
Description: 
'''
from __future__ import absolute_import, with_statement
from __future__ import division
from __future__ import print_function
import imp

import pickle as pkl
import json
from kornia.utils import image
import numpy as np
import math
import cv2
import os
import random
import matplotlib.pyplot as plt
from numpy.lib.ufunclike import isposinf
from pyquaternion import Quaternion
import pycocotools.coco as coco

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, transform_matrix

import sys
sys.path.append('./')
from transformation import get_imgcoord2worldgrid_matrices, get_imgcoord_matrices, get_worldcoord_from_imagecoord, get_2d_polygon, get_crop_shift_mat
import kornia
import torch

####################### Default Settings #########################
# dataset_dir = '/DATA7_DB7/data/shfang/airsim_camera_seg_15'
dataset_dir = '/GPFS/data/yhu/Dataset/airsim_camera/airsim_camera_seg_15'
# dataset_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data/airsim_camera_10scene')
vis_score_thre = 0.3
camera_intrinsic = [[400.0, 0.0, 400.0],
                        [0.0, 400.0, 225.0],
                        [0.0, 0.0, 1.0]]
camera_intrinsic = np.array(camera_intrinsic)
# worldgrid2worldcoord_mat = np.array([[1, 0, -100], [0, 1, -100], [0, 0, 1]])
# scale_h = 450/500 * 4
# scale_w = 800/500 * 4
scale_h = 500/500 * 4
scale_w = 500/500 * 4
map_scale_h = 1 / scale_h
map_scale_w = 1 / scale_w
world_X_left = 200
world_Y_left = 250
with_rotat = True
worldgrid2worldcoord_mat = np.array([[1/scale_w, 0, -world_X_left], [0, 1/scale_h, -world_Y_left], [0, 0, 1]])
# default_worldgrid2worldcoord_mat = np.array([[500/800, 0, -200], [0, 500/450, -250], [0, 0, 1]])
default_worldgrid2worldcoord_mat = np.array([[1, 0, -world_X_left], [0, 1, -world_Y_left], [0, 0, 1]])
# image_size = (int(500*scale_h), int(500*scale_w))
image_size = (int(192/map_scale_h), int(352/map_scale_w))
# image_size = (int(500*scale_h), int(300*scale_w))
# image_size = (225, 400)
##################################################################
def BoxCoordTrans(coord, translation, rotation, mode='L2G', with_rotat=False, sensor_type="BOTTOM"):
    project_mat = get_imgcoord_matrices(translation.copy(), rotation.copy(), camera_intrinsic)
    project_mat = project_mat @ default_worldgrid2worldcoord_mat
    # project_mat = project_mat @ worldgrid2worldcoord_mat
    if with_rotat:
        rotat_mat = get_crop_shift_mat(translation.copy(), rotation.copy(), sensor_type, 1, 1, world_X_left, world_Y_left)

    if mode == 'L2G':
        project_mat = rotat_mat @ np.linalg.inv(project_mat) if with_rotat else np.linalg.inv(project_mat)
    else:
        project_mat = project_mat @ np.linalg.inv(rotat_mat) if with_rotat else project_mat

    coord = np.concatenate([coord[:2], np.ones([1, coord.shape[1]])], axis=0)
    coord_warp = project_mat @ coord
    coord_warp = coord_warp / coord_warp[2, :]
    return coord_warp

def WorldCoord2WorldGrid(coord):
    x = (coord[0:1] + 200) * scale_w
    y = (coord[1:2] + 250) * scale_h
    return np.concatenate([x, y], axis=0)


def vis_img(sensor, coco_, catIds, res_annos_all, dataset_dir, mode='Local'):
    """
    Visualize and save image (img_id) 
    :param img_id: int, image id in class coco_
    :param save_path: str, file path to save the visualized image
    :param coco_: class
    :param catIds: list,
    :param res_annos_all: the prediction result in coco format
    :param dataset_dir: image root path
    :return: image <h, w, 3> or None
    """
    img_id = sensor['image_id']
    img = coco_.loadImgs(img_id)[0]
    annIds = coco_.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    annos = coco_.loadAnns(annIds)
    res_annos = [anno for anno in res_annos_all if anno['image_id'] == img_id]
    if len(res_annos) < 1:
        return

    if mode == 'Local':
        image_u = cv2.imread(os.path.join(dataset_dir, img['file_name']))
        image_g = CoordTrans(image_u.copy(), sensor['translation'].copy(), sensor['rotation'].copy(), mode='L2G')
        image_up = vis_cam(image_u.copy(), annos)
        image_up = vis_cam(image_up, res_annos, color=(0, 0, 255), vis_thre=vis_score_thre)
        image_gp = vis_cam_g(image_g, annos, sensor['translation'].copy(), sensor['rotation'].copy())
        image_gp = vis_cam_g(image_gp, res_annos, sensor['translation'].copy(), sensor['rotation'].copy(), color=(0, 0, 255), vis_thre=vis_score_thre)
    else:
        # import ipdb; ipdb.set_trace()
        image_u = cv2.imread(os.path.join(dataset_dir, img['file_name']))
        sensor_type = sensor['image'].split('/')[1].split('_')[1]
        image_g = CoordTrans(image_u.copy(), sensor['translation'].copy(), sensor['rotation'].copy(), mode='L2G', with_rotat=with_rotat, sensor_type=sensor_type)
        image_up = vis_cam_g(image_u, annos, sensor['translation'].copy(), sensor['rotation'].copy(), mode='G2L', with_rotat=with_rotat, sensor_type=sensor_type)
        image_up = vis_cam_g(image_up, res_annos, sensor['translation'].copy(), sensor['rotation'].copy(), color=(0, 0, 255), vis_thre=vis_score_thre, mode='G2L',  with_rotat=with_rotat, sensor_type=sensor_type)
        image_gp = vis_cam(image_g.copy(), annos, scale=scale_h)
        image_gp = vis_cam(image_gp, res_annos, color=(0, 0, 255), vis_thre=vis_score_thre, scale=scale_h)

        image_global = CoordTrans(image_u.copy(), sensor['translation'].copy(), sensor['rotation'].copy(), mode='L2G', with_rotat=False, sensor_type=sensor_type)

    return image_up, image_gp, image_global

def CoordTrans(image, translation, rotation, mode='L2G', with_rotat=False, sensor_type='BOTTOM'):
    """
    Coordnate transform:
        mode: 'L2G' from local (uav) to global
        mode: 'G2L' from global to local (uav)
    :param image: <h, w, c> BRG image
    :translation and rotation: local coord
    :return: img_warp <h, w, c> 
    """
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if with_rotat:
        world_mat = worldgrid2worldcoord_mat
    else:
        world_mat = default_worldgrid2worldcoord_mat
    project_mat = get_imgcoord2worldgrid_matrices(translation.copy(),
                                                      rotation.copy(),
                                                      camera_intrinsic,
                                                      world_mat)
    
    # project_mat2 = get_imgcoord_matrices(translation.copy(),
    #                                                   rotation.copy(),
    #                                                   camera_intrinsic)
    # project_mat2 = np.linalg.inv(project_mat2 @ worldgrid2worldcoord_mat)
    
    if mode == 'L2G':
        trans_mat = project_mat
    else:
        trans_mat = np.linalg.inv(project_mat)
    
    if with_rotat:
        rotat_mat = get_crop_shift_mat(translation.copy(), rotation.copy(), sensor_type, map_scale_w, map_scale_h, world_X_left, world_Y_left)
        trans_mat = rotat_mat@trans_mat
    data = kornia.image_to_tensor(image, keepdim=False)

    if with_rotat:
        dsize = image_size
    else:
        dsize = (500, 500)
    data_warp = kornia.warp_perspective(data.float(),
                                        torch.tensor(trans_mat).repeat([1, 1, 1]).float(),
                                        dsize=dsize)

    # convert back to numpy
    img_warp = kornia.tensor_to_image(data_warp.byte())
    return img_warp

def vis_cam(image, annos, color=(127, 255, 0), vis_thre=-1, scale=1):
    # if vis_thre == -1:
    #     image = np.zeros_like(image) * 255
    # image = np.array(image * 0.85, dtype=np.int32)
    # image = np.array(image * 0.7, dtype=np.int32)
    # alpha = np.ones([image.shape[0], image.shape[1], 1]) * 100
    # image = np.concatenate([image, alpha], axis=-1)
    # color = (255, 255, 255)
    # color = (240, 32, 160)
    # thickness = 2
    # for anno in annos:
    #     # if (anno.get('score', 0) > vis_thre) and (not anno.get('ignore', 0)):
    #     if (anno.get('score', 0) > vis_thre):
    #         if 'corners' in anno:
    #             polygon = np.array(anno['corners'][:8]).reshape([4,2])
    #             polygon = polygon * scale
    #             # color = (240, 32, 160) if polygon[:,0].max() < 700 else (127, 255, 0)
    #             image = cv2.polylines(image, pts=np.int32([polygon.reshape(-1, 1, 2)]), isClosed=True, color=color, thickness=thickness)
    #         else:
    #             bbox = anno['bbox']
    #             if len(bbox) == 4:
    #                 bbox = [x*scale for x in bbox]
    #                 x, y, w, h = bbox
    #                 image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness)
    #             else:
    #                 polygon = np.array(get_2d_polygon(np.array(bbox[:8]).reshape([4,2]).T)).reshape([4,2])
    #                 polygon = polygon * scale
    #                 image = cv2.polylines(image, pts=np.int32([polygon.reshape(-1, 1, 2)]), isClosed=True, color=color, thickness=thickness)
    return image

def xywh2polygon(bbox):
    x, y, w, h = bbox
    left_up = [x, y]
    left_bottom = [x, y+h]
    right_up = [x+w, y]
    right_bottom = [x+w, y+h]
    return np.array([left_up, left_bottom, right_bottom, right_up]).reshape([4,2]).T

def vis_cam_g(image, annos, tranlation, rotation, color=(127, 255, 0), vis_thre=-1, mode='L2G', with_rotat=with_rotat, sensor_type='BOTTOM'):
    # ori_image = image.copy().mean(axis=-1)
    for anno in annos:
        # if (anno.get('score', 0) > vis_thre) and (not anno.get('ignore', 0)):
        if (anno.get('score', 0) > vis_thre):
            if 'bbox' in anno:
                bbox = anno['bbox']
                if len(bbox) == 4:
                    polygon = xywh2polygon(bbox)    # [2, 4]
                else:
                    polygon = np.array(bbox[:8]).reshape([4,2]).T
            else:
                polygon = np.array(anno['corners'][:8]).reshape([4,2]).T
            bbox_g = BoxCoordTrans(polygon.copy(), tranlation, rotation, mode, with_rotat, sensor_type)
            bbox_g = np.array(get_2d_polygon(bbox_g[:2])).reshape([4,2])
            # bbox_g_back = BoxCoordTrans(bbox_g.copy(), tranlation, rotation, mode='G2L')
            # print('Diff: ', np.abs(bbox_g_back[:2]-polygon).sum())
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # bbox_g = WorldCoord2WorldGrid(bbox_g[:2])
            # print('bbox_g: ', bbox_g)
            # print('x: {}-{}, y: {}-{}'.format(ori_image.nonzero()[1].min(), ori_image.nonzero()[1].max(), ori_image.nonzero()[0].min(), ori_image.nonzero()[0].max()))
            image = cv2.polylines(image, pts=np.int32([bbox_g.reshape(-1, 1, 2)]), isClosed=True, color=color, thickness=1)
    # cv2.imwrite('test.png', image)
    return image

def vis_uav(images):
    h, w, c = images['FRONT'].shape
    image = np.zeros([h*3, w*3, c])
    image[:h, w:w*2, :] = images['FRONT']
    image[h:h*2] = np.concatenate([images['LEFT'], images['BOTTOM'], images['RIGHT']], axis=1)
    image[h*2:, w:w*2, :] = images['BACK']
    return image

def vis_sample(images):
    uavs = list(images.keys())
    h, w, c = images[uavs[0]].shape
    image = np.zeros([h*3, w*3, c])
    image[:h, w:w*2, :] = images[uavs[0]]
    image[h:h*2] = np.concatenate([images[uavs[1]], images[uavs[2]], images[uavs[3]]], axis=1)
    image[h*2:, w:w*2, :] = images[uavs[-1]]
    return image

def visualize_result():
    # resFile
    # result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/ctdet/coco_dla/results.json')
    # result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/multiagent_det/dla_multiagent/results.json')
    # result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/multiagent_det/dla_multiagent_NO_MESSAGE/results.json')
    # result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/multiagent_det/dla_multiagent_NO_MESSAGE/results_LateFused.json')
    # result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/multiagent_det/dla_multiagent_NO_MESSAGE/results_BEV.json')
    # result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/multiagent_det/dla_multiagent_nowarp_NO_MESSAGE/results.json')
    # result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/multiagent_det/dla_multiagent_nowarp_NO_MESSAGE/results_LateFused.json')
    # result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/multiagent_det/dla_multiagent_SINGLE_GLOBAL_MESSAGE/results.json')
    # result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/multiagent_det/dla_multiagent_withwarp_NO_MESSAGE_FeatTransImage2/results.json')
    # result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/multiagent_det/dla_multiagent_withwarp_NO_MESSAGE_FeatTransAll/results.json')
    # result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/multiagent_det/dla_multiagent_withwarp_NO_MESSAGE_FeatTransAll_GlobalCoord/trainval_results.json')
    # result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/multiagent_det/dla_multiagent_withwarp_GlobalCoord_Polygon_debug/results.json')
    # result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/multiagent_det/dla_multiagent_withwarp_GlobalCoord_Polygon_UAVGT/results.json')
    # result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/multiagent_det/dla_multiagent_withwarp_GlobalCoord_Polygon_FeatMap_800_450/results.json')
    # result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/multiagent_det/dla_multiagent_withwarp_GlobalCoord_GTFeatMap_352_192_Down4_BEVGT_40m_Town5_Baseline_MapScale1/results.json')
    # result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/multiagent_det/dla_multiagent_withwarp_GlobalCoord_GTFeatMap_352_192_Down4_BEVGT_40m_Town5_Baseline_MapScale2/results.json')
    # result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/multiagent_det/dla_multiagent_withwarp_GlobalCoord_GTFeatMap_800_450_Down4_BEVGT_40m_Baseline_Town5/results.json')
    # result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/multiagent_det/dla_multiagent_withwarp_GlobalCoord_GTFeatMap_352_192_Down4_BEVGT_40m_Town5_V2V_MapScale2/results.json')
    # result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/multiagent_det/dla_multiagent_withwarp_GlobalCoord_GTFeatMap_352_192_BEVGT_40m_Town5_Baseline_NOMESSAGE_MapScale1/results_Global.json')
    # result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/multiagent_det/GlobalCoord_Inter_DADW_WeightedDepth/results_Global.json')
    # result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/multiagent_det/LocalCoord_repeat/results_Local.json')
    # result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/multiagent_det/LocalCoord_repeat/results_BEV_Polygon.json')
    # result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/multiagent_det/JointCoord_Inter_DADW_WeightedDepth/results_Global_Town5.json')
    # result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/multiagent_det/JointCoord_Inter_DADW_WeightedDepth/results_Local_Town5.json')
    # result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/multiagent_det/JointCoord_Inter_DADW_WeightedDepth/results_Local.json')

    # COLLABORATION
    # result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/multiagent_det/NO_MESSAGE/results_Global.json')
    # result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/multiagent_det/V2V/results_Global.json')
    result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/multiagent_det/HW_QualityMapMessage_Translayer0_CommMask_Transformer_Weight_Updated/results_Global.json')
    # result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/multiagent_det/HW_V2V_Updated/results_Global.json')
    # result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/multiagent_det/HW_When2com_Updated/results_Global.json')
    # result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/multiagent_det/HW_NO_MESSAGE/results_Global.json')
    # result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/multiagent_det/HW_Pointwise/results_Global.json')
    # result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/multiagent_det/HW_QualityMapMessage_Translayer0_CommMask_Transformer_WOPE_Repeat_Updated/results_Global.json')

    coord_mode = 'Global' if ('Global' in result_path) or ('BEV' in result_path) else 'Local'
    print('Coord_mode: ', coord_mode)
    res_annos_all = json.load(open(result_path))

    # coco_ = coco.COCO(os.path.join(os.path.dirname(__file__), '..', '..', 'data/airsim_camera_10scene/annotations/train_instances.json'))
    # coco_ = coco.COCO(os.path.join(os.path.dirname(__file__), '..', '..', 'data/airsim_camera_10scene/annotations/val_instances.json'))
    if coord_mode == 'Global':
        # coco_ = coco.COCO(os.path.join(dataset_dir, 'multiagent_annotations/{}_val_instances_global_crop.json'.format(40)))
        coco_ = coco.COCO(os.path.join(dataset_dir, 'multiagent_annotations/Collaboration/{}_val_instances_global_crop.json'.format(40)))
        # coco_ = coco.COCO( os.path.join(os.path.dirname(__file__), '..', '..', 'exp/multiagent_det/LocalCoord_repeat/gts_Global.json'))
        # coco_ = coco.COCO(os.path.join(dataset_dir, 'multiagent_annotations/{}_val_instances_global.json'.format(40)))
        # coco_ = coco.COCO(os.path.join(dataset_dir, 'multiagent_annotations/val_instances_global.json'))
        # coco_ = coco.COCO(os.path.join(os.path.dirname(__file__), '..', '..', 'data/airsim_camera_10scene/multiagent_annotations/train_instances_global.json'))
    else:    
        # coco_ = coco.COCO(os.path.join(dataset_dir, 'multiagent_annotations/{}_val_instances.json'.format(40)))
        coco_ = coco.COCO(os.path.join(dataset_dir, 'multiagent_annotations/Collaboration/{}_val_instances.json'.format(40)))
        # coco_ = coco.COCO(os.path.join(dataset_dir, 'multiagent_annotations/val_instances.json'))
    
    catIds = coco_.getCatIds()
    imgIds = coco_.getImgIds()

    # save path
    save_path = os.path.join(os.path.dirname(result_path), 'vis')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Vis Sample
    samples = pkl.load(open(os.path.join(dataset_dir, 'multiagent_annotations', 'Collaboration', '{}_val_instances_sample.pkl'.format(40)), 'rb'))['samples']
    # samples = pkl.load(open(os.path.join(dataset_dir, 'multiagent_annotations', 'val_instances_sample.pkl'), 'rb'))['samples']
    # samples = pkl.load(open(os.path.join(dataset_dir, 'multiagent_annotations', 'train_instances_sample.pkl'), 'rb'))['samples']
    for sample_id, sample in enumerate(samples):
        if sample_id in [54]:
            pass
        else:
            continue
        images_up = {}
        images_gp = {}
        images_global = {}
        for k, sensor in sample.items():
            if k.startswith('vehicles'):
                continue
            else:
                if sensor['image_id'] in imgIds:
                    sample_save_path = os.path.join(save_path, '{}'.format(sample_id))
                    if not os.path.exists(sample_save_path):
                        os.makedirs(sample_save_path)
                    image_up, image_gp, image_global = vis_img(sensor, coco_, catIds, res_annos_all, dataset_dir, coord_mode)
                    if image_up is not None:
                        images_up[k] = image_up
                        images_gp[k] = image_gp
                        images_global[k] = image_global
        uavs = set([x.split('_')[-1] for x in images_up])
        cams = set([x.split('_')[0] for x in images_up])
        uav_images = {}
        uav_images_g = {}
        for uav in uavs:
            cur_uav = {}
            cur_uav_g = {}
            for cam in cams:
                cam_name = '{}_{}'.format(cam, uav)
                cur_uav[cam] = images_up[cam_name]
                cv2.imwrite('{}/{}_pred.png'.format(sample_save_path, cam_name), cur_uav[cam])
                # cur_uav_g[cam] = CoordTrans(ori_images[cam_name].copy(), translations[cam_name].copy(), rotations[cam_name].copy())
                # cur_uav_g[cam] = vis_cam_g(cur_uav_g[cam], box_annos[cam_name], translations[cam_name].copy(), rotations[cam_name].copy())
                # cur_uav_g[cam] = vis_cam_g(cur_uav_g[cam], res_box_annos[cam_name], translations[cam_name].copy(), rotations[cam_name].copy(), color=(0, 0, 255), vis_thre=vis_score_thre)
                cur_uav_g[cam] = images_global[cam_name]
                cv2.imwrite('{}/{}_pred_g.png'.format(sample_save_path, cam_name), images_gp[cam_name])
                # # vis original
                # ori_img = images[cam_name]
                # cv2.imwrite('ori.png', ori_img)
                # # vis warp
                # img_warp = cur_uav_g[cam]
                # cv2.imwrite('img_warp.png', img_warp)
                # # vis warp back
                # img_warp_back = CoordTrans(img_warp.copy(), translations[cam_name], rotations[cam_name], 'G2L')
                # cv2.imwrite('img_warp_back.png', img_warp_back)
            uav_images[uav] = vis_uav(cur_uav)
            cv2.imwrite('{}/{}_pred.png'.format(sample_save_path, uav), uav_images[uav])
            uav_images_g[uav] = np.concatenate([x[None,:,:,:] for _, x in cur_uav_g.items()], axis=0).max(axis=0)
            cv2.imwrite('{}/{}_pred_g.png'.format(sample_save_path, uav), uav_images_g[uav])
        sample_image = vis_sample(uav_images)
        cv2.imwrite('{}/pred.png'.format(sample_save_path), sample_image)
        sample_image_g = np.concatenate([x[None,:,:,:] for _, x in uav_images_g.items()], axis=0).max(axis=0)
        cv2.imwrite('{}/pred_g.png'.format(sample_save_path, uav), sample_image_g)
        # import ipdb; ipdb.set_trace()

        for cam in cams:
            cur_cam_g = {}
            for uav in uavs:
                cam_name = '{}_{}'.format(cam, uav)
                cur_cam_g[uav] = images_global[cam_name]
            cam_g = np.concatenate([x[None,:,:,:] for _, x in cur_cam_g.items()], axis=0).max(axis=0)
            cv2.imwrite('{}/{}_pred_g.png'.format(sample_save_path, cam), cam_g)
        
        # cam_images = {}
        # cam_images_g = {}
        # padding = np.ones([450,20,3], dtype=np.uint8) * 255
        # padding_g = np.ones([768,20,3], dtype=np.uint8) * 255
        # for cam in cams:
        #     cur_cam = [np.concatenate([x,padding], axis=1) for cam_name, x in images_up.items() if cam in cam_name]
        #     cur_cam_g = [np.concatenate([x,padding_g], axis=1) for cam_name, x in images_gp.items() if cam in cam_name]
        #     # cur_cam_g = [x for cam_name, x in images_gp.items() if cam in cam_name]
        #     cam_images[cam] = np.concatenate(cur_cam, axis=1)
        #     cv2.imwrite('{}/{}_pred.png'.format(sample_save_path, cam), cam_images[cam])
        #     # cam_images_g[cam] = np.concatenate([x[None,:,:,:] for x in cur_cam_g], axis=0).max(axis=0)
        #     cam_images_g[cam] = np.concatenate(cur_cam_g, axis=1)
        #     cv2.imwrite('{}/{}_pred_g.png'.format(sample_save_path, cam), cam_images_g[cam])

    # Vis Image
    # for img_id in imgIds[:100]:
    #     vis_img(img_id, save_path, coco_, catIds, res_annos_all, dataset_dir)

if __name__ == '__main__':
    visualize_result()
