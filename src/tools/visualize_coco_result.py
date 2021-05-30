'''
Author: yhu
Contact: phyllis1sjtu@outlook.com
LastEditTime: 2021-05-30 10:13:33
Description: 
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle as pkl
import json
import numpy as np
import math
import cv2
import os
import random
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
import pycocotools.coco as coco

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, transform_matrix

import sys
sys.path.append('./')
from transformation import get_imgcoord2worldgrid_matrices
import kornia
import torch

####################### Default Settings #########################
vis_score_thre = 0.5
camera_intrinsic = [[400.0, 0.0, 400.0],
                        [0.0, 400.0, 225.0],
                        [0.0, 0.0, 1.0]]
camera_intrinsic = np.array(camera_intrinsic)
# worldgrid2worldcoord_mat = np.array([[1, 0, -100], [0, 1, -100], [0, 0, 1]])
worldgrid2worldcoord_mat = np.array([[1, 0, -250], [0, 1, -250], [0, 0, 1]])
image_size = (450, 800)
# image_size = (225, 400)
###################################################################


def vis_img(img_id, save_path, coco_, catIds, res_annos_all, dataset_dir):
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
    img = coco_.loadImgs(img_id)[0]
    annIds = coco_.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    annos = coco_.loadAnns(annIds)
    res_annos = [anno for anno in res_annos_all if anno['image_id'] == img_id]
    if len(res_annos) < 1:
        return

    image = cv2.imread(os.path.join(dataset_dir, img['file_name']))

    image = vis_cam(image, annos)
    image = vis_cam(image, res_annos, color=(0, 0, 255), vis_thre=vis_score_thre)
    cv2.imwrite('{}/{}_pred.png'.format(save_path, img_id), image)
    return image

def CoordTrans(image, translation, rotation, mode='L2G'):
    """
    Coordnate transform:
        mode: 'L2G' from local (uav) to global
        mode: 'G2L' from global to local (uav)
    :param image: <h, w, c> BRG image
    :translation and rotation: local coord
    :return: img_warp <h, w, c> 
    """
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    project_mat = get_imgcoord2worldgrid_matrices(translation.copy(),
                                                      rotation.copy(),
                                                      camera_intrinsic,
                                                      worldgrid2worldcoord_mat)
    if mode == 'L2G':
        trans_mat = project_mat
    else:
        trans_mat = np.linalg.inv(project_mat)
        
    data = kornia.image_to_tensor(image, keepdim=False)
    data_warp = kornia.warp_perspective(data.float(),
                                        torch.tensor(trans_mat).repeat([1, 1, 1]).float(),
                                        dsize=image_size)

    # convert back to numpy
    img_warp = kornia.tensor_to_image(data_warp.byte())
    return img_warp

def vis_cam(image, annos, color=(127, 255, 0), vis_thre=-1):
    for anno in annos:
        if anno.get('score', 0) > vis_thre:
            bbox = anno['bbox']
            x, y, w, h = bbox
            image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 1)
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
    # coco_ = coco.COCO(os.path.join(os.path.dirname(__file__), '..', '..', 'data/airsim_camera_10scene/annotations/train_instances.json'))
    # coco_ = coco.COCO(os.path.join(os.path.dirname(__file__), '..', '..', 'data/airsim_camera_10scene/annotations/val_instances.json'))
    coco_ = coco.COCO(os.path.join(os.path.dirname(__file__), '..', '..', 'data/airsim_camera_10scene/multiagent_annotations/val_instances.json'))
    dataset_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data/airsim_camera_10scene')
    catIds = coco_.getCatIds()
    imgIds = coco_.getImgIds()

    # resFile
    # result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/ctdet/coco_dla/results.json')
    result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/multiagent_det/dla_multiagent/results.json')
    res_annos_all = json.load(open(result_path))

    # save path
    save_path = os.path.join(os.path.dirname(result_path), 'vis')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Vis Sample
    samples = pkl.load(open(os.path.join(dataset_dir, 'multiagent_annotations', 'val_instances_sample.pkl'), 'rb'))['samples']
    for sample_id, sample in enumerate(samples):
        images = {}
        translations = {}
        rotations = {}
        for k, sensor in sample.items():
            if k.startswith('vehicles'):
                continue
            else:
                if sensor['image_id'] in imgIds:
                    sample_save_path = os.path.join(save_path, '{}'.format(sample_id))
                    if not os.path.exists(sample_save_path):
                        os.makedirs(sample_save_path)
                    image = vis_img(sensor['image_id'], sample_save_path, coco_, catIds, res_annos_all, dataset_dir)
                    if image is not None:
                        images[k] = image
                        translations[k] = sensor['translation']
                        rotations[k] = sensor['rotation']
        uavs = set([x.split('_')[-1] for x in images])
        cams = set([x.split('_')[0] for x in images])
        uav_images = {}
        uav_images_g = {}
        for uav in uavs:
            cur_uav = {}
            cur_uav_g = {}
            for cam in cams:
                cam_name = '{}_{}'.format(cam, uav)
                cur_uav[cam] = images[cam_name]
                cur_uav_g[cam] = CoordTrans(images[cam_name].copy(), translations[cam_name], rotations[cam_name])
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


    # Vis Image
    # for img_id in imgIds[:100]:
    #     vis_img(img_id, save_path, coco_, catIds, res_annos_all, dataset_dir)

if __name__ == '__main__':
    visualize_result()
