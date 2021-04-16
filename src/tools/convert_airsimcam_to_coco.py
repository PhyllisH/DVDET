from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import json
import numpy as np
import math
import cv2
import os
import random
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
import pycocotools.coco as coco

# DATA_PATH = '../../data/kitti/'

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, transform_matrix

train_split = ['scene_0', 'scene_1', 'scene_2', 'scene_3', 'scene_4', 'scene_5',
               'scene_6', 'scene_7', 'scene_8', 'scene_9', 'scene_11', 'scene_12',
               'scene_13', 'scene_14', 'scene_16', 'scene_17', 'scene_18', 'scene_19',
               'scene_20', 'scene_21', 'scene_22', 'scene_23', 'scene_24', 'scene_26',
               'scene_28', 'scene_29', 'scene_30', 'scene_31', 'scene_32', 'scene_33',
               'scene_34', 'scene_35', 'scene_36', 'scene_37', 'scene_38', 'scene_39',
               'scene_40', 'scene_42', 'scene_44', 'scene_45', 'scene_46', 'scene_47',
               'scene_48', 'scene_49', 'scene_50', 'scene_51', 'scene_52', 'scene_53',
               'scene_55', 'scene_56', 'scene_57', 'scene_61', 'scene_62', 'scene_63',
               'scene_65', 'scene_66', 'scene_67', 'scene_68', 'scene_69', 'scene_70',
               'scene_71', 'scene_72', 'scene_73', 'scene_75', 'scene_76', 'scene_77',
               'scene_78', 'scene_79', 'scene_80', 'scene_81', 'scene_82', 'scene_83',
               'scene_84', 'scene_87', 'scene_88', 'scene_90', 'scene_92', 'scene_94',
               'scene_95', 'scene_97', 'scene_98', 'scene_99', 'scene_100', 'scene_101',
               'scene_102', 'scene_103', 'scene_104', 'scene_105', 'scene_106', 'scene_107',
               'scene_108', 'scene_109', 'scene_110', 'scene_111', 'scene_112', 'scene_113',
               'scene_114', 'scene_116', 'scene_118', 'scene_119']

val_split = ['scene_10', 'scene_15', 'scene_25', 'scene_27', 'scene_41', 'scene_43',
             'scene_54', 'scene_58', 'scene_59', 'scene_60', 'scene_64', 'scene_74',
             'scene_85', 'scene_86', 'scene_89', 'scene_91', 'scene_93', 'scene_96',
             'scene_115', 'scene_117']


def quaternion2euler(rotation):
    w, x, y, z = rotation[0], rotation[1], rotation[2], rotation[3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z


def _get_rotation_matrix(translation, rotation):
    roll, pitch, yaw = quaternion2euler(rotation)
    c_y = np.cos(yaw)
    s_y = np.sin(yaw)
    c_r = np.cos(roll)
    s_r = np.sin(roll)
    c_p = np.cos(pitch)
    s_p = np.sin(pitch)
    matrix = np.matrix(np.identity(4))
    matrix[0, 3] = translation[0]
    matrix[1, 3] = translation[1]
    matrix[2, 3] = translation[2]
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix


def _get_vehicle_coord(anno_data):
    translation = anno_data["translation"]
    size = anno_data["size"]
    a = size[0]
    size[0] = size[1]
    size[1] = a
    rotation = anno_data["rotation"]

    # cords the bounding box of a vehicle
    cords = np.zeros((8, 4))
    cords[0, :] = np.array([size[0] / 2, size[1] / 2, -size[2] / 2, 1])
    cords[1, :] = np.array([-size[0] / 2, size[1] / 2, -size[2] / 2, 1])
    cords[2, :] = np.array([-size[0] / 2, -size[1] / 2, -size[2] / 2, 1])
    cords[3, :] = np.array([size[0] / 2, -size[1] / 2, -size[2] / 2, 1])
    cords[4, :] = np.array([size[0] / 2, size[1] / 2, size[2] / 2, 1])
    cords[5, :] = np.array([-size[0] / 2, size[1] / 2, size[2] / 2, 1])
    cords[6, :] = np.array([-size[0] / 2, -size[1] / 2, size[2] / 2, 1])
    cords[7, :] = np.array([size[0] / 2, -size[1] / 2, size[2] / 2, 1])

    vehicle_world_matrix = _get_rotation_matrix(translation, rotation)

    world_cords = np.dot(vehicle_world_matrix, np.transpose(cords))
    return np.array(world_cords)


def get_2d_bounding_box(cords):
    x_min = cords[0, 0]
    x_max = cords[0, 0]
    y_min = cords[1, 0]
    y_max = cords[1, 0]
    for i in range(1, 8):
        if cords[0, i] < x_min:
            x_min = cords[0, i]
        if cords[0, i] > x_max:
            x_max = cords[0, i]
        if cords[1, i] < y_min:
            y_min = cords[1, i]
        if cords[1, i] > y_max:
            y_max = cords[1, i]
    return x_min, y_min, x_max - x_min, y_max - y_min


def convert_coco():
    # data_dir = 'C:/Users/35387/Desktop/airsim_camera_demo'
    data_dir = '/DB/rhome/shaohengfang/datasets/airsim/airsim_camera_demo'
    DEBUG = False
    nusc = NuScenes(version='v1.0-mini', dataroot=data_dir, verbose=True)

    cats = ['car', 'car_overlook']
    splits = ['train', 'val']
    scene_split = {'train': train_split, 'val': val_split}
    cat_ids = {cat: i + 1 for i, cat in enumerate(cats)}

    F = 400  # focal
    H = 450  # height
    W = 800  # width
    camera_intrinsic = [[400.0, 0.0, 400.0],
                        [0.0, 400.0, 225.0],
                        [0.0, 0.0, 1.0]]

    cat_info = []
    for i, cat in enumerate(cats):
        cat_info.append({'supercategory': 'vehicle', 'name': cat, 'id': i + 1})
    image_id = 0
    bbox_id = 0
    for split in splits:
        ret = {'images': [], "type": "instances", 'annotations': [], 'categories': cat_info}
        for scene in nusc.scene:
            if not scene["name"] in scene_split[split]:
                continue
            scene_token = scene['token']
            cur_sample_token = scene['first_sample_token']
            while cur_sample_token != "":
                print(cur_sample_token)
                cur_sample = nusc.get("sample", cur_sample_token)
                # =======================
                # execute the current sample data
                anno_tokens = cur_sample["anns"]
                # get the vehicle coords in global frame
                vehicle_cords = []
                for anno_token in anno_tokens:
                    anno_data = nusc.get("sample_annotation", anno_token)
                    vehicle_cords.append(_get_vehicle_coord(anno_data))

                sample_data = cur_sample["data"]
                sensors = list(sample_data.keys())
                for sensor in sensors:
                    # image info
                    sensor_record = nusc.get("sample_data", sample_data[sensor])
                    image_id += 1
                    image_info = {'file_name': sensor_record['filename'],
                                  'id': image_id,
                                  'height': 450,
                                  'width': 900}
                    ret['images'].append(image_info)
                    # anno info
                    calibrated_record = nusc.get("calibrated_sensor", sensor_record["calibrated_sensor_token"])
                    im_position = calibrated_record["translation"]
                    im_position[2] = -im_position[2]
                    im_rotation = calibrated_record["rotation"]
                    im_rotation[3] = -im_rotation[3]
                    im_rotation = Quaternion(im_rotation)

                    cat_id = 1
                    if sensor[:10] == "CAM_BOTTOM":
                        cat_id = 2
                    for vehicle_cord in vehicle_cords:
                        flag = True
                        # get bbox from vehicle_cord
                        vehicle_cord_ = np.array(vehicle_cord)
                        vehicle_cord_ = vehicle_cord_[:3, :]
                        for j in range(3):
                            vehicle_cord_[j, :] = vehicle_cord_[j, :] - im_position[j]
                        vehicle_cord_[:3, :] = np.dot(im_rotation.rotation_matrix, vehicle_cord_[:3, :])
                        vehicle_cord_[:3, :] = np.dot(Quaternion([0.5, -0.5, 0.5, -0.5]).rotation_matrix.T,
                                                      vehicle_cord_[:3, :])
                        depths = vehicle_cord_[2, :]
                        for j in range(8):
                            if depths[j] < 0:
                                flag = False
                        if not flag:
                            continue
                        vehicle_points = view_points(vehicle_cord_[:3, :], np.array(camera_intrinsic), normalize=True)
                        x, y, w, h = get_2d_bounding_box(vehicle_points)
                        if x < 0 or y < 0 or (x + w) > 800 or (y + h) > 450:
                            flag = False
                        if not flag:
                            continue
                        bbox_id += 1
                        ann = {'area': w * h,
                               'iscrowd': 0,
                               'image_id': image_id,
                               'bbox': [800 - x - w, y, w, h],
                               'category_id': cat_id,
                               'id': bbox_id,
                               'ignore': 0,
                               'segmentation': []}
                        ret['annotations'].append(ann)
                # =======================
                cur_sample_token = cur_sample['next']
        print("# images: ", len(ret['images']))
        print("# annotations: ", len(ret['annotations']))
        # out_path = 'C:/Users/35387/Desktop/airsim_camera_demo/airsim_instances_{}.json'.format(split)
        out_path = '/DB/rhome/shaohengfang/model/CenterNet/data/airsim_camera/annotations/{}_instances.json'.format(split)
        json.dump(ret, open(out_path, 'w'))


if __name__ == '__main__':
    convert_coco()
