'''
Author: yhu
Contact: phyllis1sjtu@outlook.com
LastEditTime: 2021-05-25 16:24:32
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


vis_score_thre = 0.5

def visualize_image_with_bbox():
    # coco_ = coco.COCO(os.path.join(os.path.dirname(__file__), '..', '..', 'data/airsim_camera_10scene/annotations/train_instances.json'))
    coco_ = coco.COCO(os.path.join(os.path.dirname(__file__), '..', '..', 'data/airsim_camera_10scene/annotations/val_instances.json'))
    dataset_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data/airsim_camera_10scene')
    # dataset_dir = "/DB/rhome/shaohengfang/datasets/airsim/airsim_camera_demo"
    catIds = coco_.getCatIds()
    imgIds = coco_.getImgIds()
    for i in range(1):
        img = coco_.loadImgs(imgIds[i])[0]
        annIds = coco_.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        annos = coco_.loadAnns(annIds)

        image = cv2.imread(os.path.join(dataset_dir, img['file_name']))

        for anno in annos:
            bbox = anno['bbox']
            x, y, w, h = bbox
            anno_image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 2)
            cv2.imshow('demo', anno_image)
        cv2.waitKey()

def vis_img(img_id, save_path, coco_, catIds, res_annos_all, dataset_dir):
    img = coco_.loadImgs(img_id)[0]
    annIds = coco_.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    annos = coco_.loadAnns(annIds)
    res_annos = [anno for anno in res_annos_all if anno['image_id'] == img_id]
    if len(res_annos) < 1:
        return

    image = cv2.imread(os.path.join(dataset_dir, img['file_name']))

    for anno in annos:
        bbox = anno['bbox']
        x, y, w, h = bbox
        image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (127, 255, 0), 1)
        # cv2.imshow('demo', anno_image)
        # cv2.imwrite('{}/{}_gt.png'.format(save_path, img_id), anno_image)
    for anno in res_annos:
        # print(anno['score'])
        if anno['score'] > vis_score_thre:
            bbox = anno['bbox']
            x, y, w, h = bbox
            image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 1)
            # cv2.imshow('demo', anno_image)
    cv2.imwrite('{}/{}_pred.png'.format(save_path, img_id), image)
    return image

def vis_uav(images, uav, save_path):
    h, w, c = images['FRONT'].shape
    image = np.zeros([h*3, w*3, c])
    image[:h, w:w*2, :] = images['FRONT']
    image[h:h*2] = np.concatenate([images['LEFT'], images['BOTTOM'], images['RIGHT']], axis=1)
    image[h*2:, w:w*2, :] = images['BACK']
    cv2.imwrite('{}/{}_pred.png'.format(save_path, uav), image)
    return image

def vis_sample(images, save_path):
    uavs = list(images.keys())
    h, w, c = images[uavs[0]].shape
    image = np.zeros([h*3, w*3, c])
    image[:h, w:w*2, :] = images[uavs[0]]
    image[h:h*2] = np.concatenate([images[uavs[1]], images[uavs[2]], images[uavs[3]]], axis=1)
    image[h*2:, w:w*2, :] = images[uavs[-1]]
    cv2.imwrite('{}/pred.png'.format(save_path), image)
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
        uavs = set([x.split('_')[-1] for x in images])
        cams = set([x.split('_')[0] for x in images])
        uav_images = {}
        for uav in uavs:
            cur_uav = {}
            for cam in cams:
                cur_uav[cam] = images['{}_{}'.format(cam, uav)]
            uav_images[uav] = vis_uav(cur_uav, uav, sample_save_path)
        vis_sample(uav_images, sample_save_path)

    # import ipdb; ipdb.set_trace()
    # Vis image
    # for img_id in range(1, 100):
    for img_id in imgIds[:100]:
        img = coco_.loadImgs(img_id)[0]
        annIds = coco_.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        annos = coco_.loadAnns(annIds)
        res_annos = [anno for anno in res_annos_all if anno['image_id'] == img_id]
        if len(res_annos) < 1:
            continue

        image = cv2.imread(os.path.join(dataset_dir, img['file_name']))

        for anno in annos:
            bbox = anno['bbox']
            x, y, w, h = bbox
            anno_image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (127, 255, 0), 1)
            # cv2.imshow('demo', anno_image)
            # cv2.imwrite('{}/{}_gt.png'.format(save_path, img_id), anno_image)
        for anno in res_annos:
            # print(anno['score'])
            if anno['score'] > vis_score_thre:
                bbox = anno['bbox']
                x, y, w, h = bbox
                anno_image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 1)
                # cv2.imshow('demo', anno_image)
                cv2.imwrite('{}/{}_pred.png'.format(save_path, img_id), anno_image)

        # cv2.waitKey()


if __name__ == '__main__':
    visualize_result()
