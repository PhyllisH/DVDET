'''
Author: yhu
Contact: phyllis1sjtu@outlook.com
LastEditTime: 2021-07-26 00:47:54
Description: Convert the single-view dataformat to multi-view dataformat
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import json
from typing_extensions import OrderedDict
from ipdb.__main__ import set_trace
from matplotlib.image import imread
import numpy as np
import math
import cv2
import os
import random
import pickle as pkl
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
import pycocotools.coco as coco
from tqdm import tqdm
import copy

# DATA_PATH = '../../data/kitti/'

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, transform_matrix
import sys
sys.path.append(os.path.dirname(__file__))
from realdata_transformation import *
from transformation import get_2d_bounding_box, get_angle_polygon, get_shift_coord, WorldCoord2WorldGrid, get_2d_polygon
from visualize_coco_result import CoordTrans
from copy import deepcopy

'''
GT:
'samples': [{
    'SENSOR': 
        {
            'trans_mat': (3, 4) # the UAV coordinate transform matrix
            'image':  # the RGB image
            'image_id': # the image id
            'translation':
            'rotation':
            'vehicles_i': (n, 4), np.array # the box in current coordinate (2D)
            'vehicles_g': (n, 4), np.array # the box in global coordinate (2D)
            'category_id': (n,), # the box category
        },
    'vehicles_s': (n, 4), np.array # all the boxes in the global coordinate (2D)
}],
'images': []
'annos': []

And COCO Annotation Format: single-view in image or global coordinates


Dataloader Output:
Single view (local coords):
    Item:
        Training:
            image: [1, C, H, W] # UAV image
            trans_mats: [1, 3, 4] # UAV coordinate
            boxes: [1, MAX_BOXES, 4] # box in current coordinate
        Eval:
            image: [1, C, H, W] # UAV image
            transform_matrix: [1, 2, 3] # UAV coordinate
            boxes: [1, BOXES, 4] # box in current coordinate
            all_boxes: [1, ALL_BOXES, 4] # all the boxes in the scene in global coordinate

Single / Multi view (global coords):
    Item: 
        Training:
            g_image: [1, MAX_UAV, C, H, W] # UAV image (global)
            g_boxes: [1, MAX_UAV, MAX_BOXES, 4] # box in global coordinate
            all_boxes: [1, MAX_UAV, MAX_BOXES, 4] # all the boxes in the scene in global coordinate
        Eval:
            g_image: [1, MAX_UAV, C, H, W] # UAV image (global)
            g_boxes: [1, MAX_UAV, MAX_BOXES, 4] # box in global coordinate
            all_boxes: [1, MAX_UAV, MAX_BOXES, 4] # all the boxes in the scene in global coordinate

Multi view:
    Item: 
        Training:
            image: [1, MAX_UAV, C, H, W] # UAV image
            transform_matrix: [1, MAX_UAV, 2, 3] # UAV coordinate
            boxes: [1, MAX_UAV, MAX_BOXES, 4] # box in each coordinate
            g_image: [1, MAX_UAV, C, H, W] # UAV image (global)
            g_boxes: [1, MAX_UAV, MAX_BOXES, 4] # box in global coordinate
            all_boxes: [1, MAX_UAV, MAX_BOXES, 4] # all the boxes in the scene in global coordinate
        Eval:
            image: [1, MAX_UAV, C, H, W] # UAV image
            transform_matrix: [1, MAX_UAV, 2, 3] # UAV coordinate
            boxes: [1, MAX_UAV, MAX_BOXES, 4] # box in each coordinate
            g_boxes: [1, MAX_UAV, MAX_BOXES, 4] # box in global coordinate
            all_boxes: [1, MAX_UAV, MAX_BOXES, 4] # all the boxes in the scene in global coordinate
            all_boxes: [1, ALL_BOXES, 4] # all the boxes in the scene in global coordinate

def TRtoTMat(translation, rotation):
    # translation: camera parameter
    # rotation: camera parameter
    return T

def UAVtoGlobal(UAV_I, T):
    # UAV_I: (C, H, W) Image in UAV Coordinates
    # T: Transform Matrix from UAV Coord to Global Coord
    return Global_I

def UAVtoUAV(UAV_I1, T1, T2):
    # UAV_I1: (C, H, W) Image in UAV1 Coordinates
    # T1: Transform Matrix from UAV1 Coord to Global Coord
    # T2: Transform Matrix from UAV2 Coord to Global Coord
    return UAV_I2
'''

# offsets
scene_offset = {
    5: (0, 0, 0, -0.8, 3.6, 0),
    9: (0, 0, 0, -0.4, 2.6, 0),
    3: (0, 0, 0, -0.0, 1.2, 0),
    12: (0, 0, 0, -0.0, 4, 0),
    13: (0, 0, 0, -0.1, 1.5, 0),
    15: (0, 0, 0, 0.4, 2.1, 0),
    14: (0, 0, 0, -0.2, 1.7, 0),
    1: (0, 0, 0, -0.4, 2.9, 0),
    2: (0, 0, 0, -0.4, 2.9, 0),
    4: (0, 0, 0, -0.25, 1.1, 0)
}
# ground_height = -30.0  # car center height of the scene

train_split = [1, 2, 3, 4, 12, 13, 14, 15]
val_split = [5, 9]

town_config = OrderedDict()
town_config[0] = {
    'world_X': 400,
    'world_Y': 400,
    'world_X_left': 200,
    'world_Y_left': 200
}


def convert_multiview_coco(town_id=1, height=40):
    ##################### Get category info ###############
    cats = ['car']
    cat_info = []
    for i, cat in enumerate(cats):
        cat_info.append({'supercategory': 'vehicle', 'name': cat, 'id': i + 1})
    #######################################################

    ########## Camera intrinsic and image settings ########
    F = 400  # focal
    H = 480  # height
    W = 720  # width
    camera_intrinsic = np.array([[486.023, 0, 359.066],
                                 [0, 486.105, 240.959],
                                 [0, 0, 1]])
    world_X = town_config[town_id]['world_X']
    world_Y = town_config[town_id]['world_Y']
    image_H = 480
    image_W = 720
    world_X_left = town_config[town_id]['world_X_left']
    world_Y_left = town_config[town_id]['world_Y_left']
    worldgrid2worldcoord_mat = np.array([[1, 0, -world_X_left], [0, 1, -world_Y_left], [0, 0, 1]])
    #######################################################

    data_dir = '/DB/public/uav_dataset'
    if not os.path.exists(os.path.join(data_dir, 'multiagent_annotations')):
        os.makedirs(os.path.join(data_dir, 'multiagent_annotations'))

    save_dir = data_dir
    splits = ['train', 'val']
    scene_split = {'train': train_split, 'val': val_split}
    image_id = 0
    bbox_id = 0

    for split in splits:
        ret_i = {'images': [], "type": "instances", 'annotations': [], 'categories': cat_info}
        ret_g = {'images': [], "type": "instances", 'annotations': [], 'categories': cat_info}
        ret_s = {"samples": [], "type": "sample", "categories": cat_info}
        ret_g_crop = {'images': [], "type": "instances", 'annotations': [], 'categories': cat_info}
        for scene_no in tqdm(scene_split[split]):
            offset = scene_offset[scene_no]
            scene_dir = os.path.join(data_dir, 'scene_{}_lidar'.format(scene_no))
            img_path = os.path.join(data_dir, 'image', '{}'.format(scene_no))
            xml_file = os.path.join(scene_dir, 'metadata.xml')  # a xml file includes the origin point coordinate in ENU
            img_info_file = os.path.join(data_dir, 'image', '{}.csv'.format(scene_no))  # a csv file
            scene_origin = get_origin_enu(xml_file)  # scene5 (121.42091098612988, 31.02890249679692, 45.19054702894702)

            # with open(os.path.join(data_dir, 'new_annotation', 'new_{}_annotation.json'.format(scene_no)), 'r') as f:
            with open(os.path.join(data_dir, 'cleaned_annotation', 'new_{}_annotation.json'.format(scene_no)), 'r') as f:
                samples = json.load(f)['samples']

            for cur_sample in samples:
                cur_sample_info = OrderedDict()
                img_dir = cur_sample["img_dir"]
                img_name = cur_sample["img_dir"].split('/')[-1]

                vehicle_cords = []
                vehicle_2ds = []
                heights = []
                for anno in cur_sample["annotation"]:
                    anno_id = anno['anno_id']
                    vehicle_cord = get_box_corners(anno['3d_object'])
                    vehicle_cords.append(vehicle_cord)
                    vehicle_2ds.append(get_2d_bounding_box(vehicle_cord.copy()))
                    heights.append(vehicle_cord[-1].mean())

                ground_height = np.array(heights).mean()
                cur_sample_info['vehicles_s'] = np.array(vehicle_2ds)

                image_id += 1
                cur_UAV_sample = OrderedDict()
                sensor = 'CAM_{}_id_{}'.format('FRONT', 0)

                ori_camera_position, camera_rotation = get_dji_pic_info(img_name, img_info_file, offset, scene_origin)
                camera_position = ori_camera_position.copy()
                camera_position[2] = float(- ground_height)

                cur_UAV_sample['translation'] = camera_position.copy()
                cur_UAV_sample['rotation'] = Quaternion(matrix=camera_rotation.copy()).q
                world2image_trans_mat = get_imgcoord_matrices_dji(camera_position.copy(),
                                                        camera_rotation.copy(),
                                                        camera_intrinsic,
                                                        z0=0)
                cur_UAV_sample['trans_mat'] = world2image_trans_mat @ worldgrid2worldcoord_mat
                
                for z0 in [-1.0, -0.5, 0.5, 0.75, 1.0, 1.5, 2.0, 8.0]:
                    if z0 < 0:
                        z_key = 'trans_mat_n{:03d}'.format(int((-1)*z0*10))
                    else:
                        z_key = 'trans_mat_p{:03d}'.format(int(z0*10))
                    cur_UAV_sample[z_key] = get_imgcoord_matrices_dji(camera_position.copy(),
                                                        camera_rotation.copy(),
                                                        camera_intrinsic,
                                                        z0=z0) @ worldgrid2worldcoord_mat

                # local image
                image_info = {'file_name': img_dir,
                                'id': image_id,
                                'height': H,
                                'width': W}
                ret_i['images'].append(image_info)
                ret_g['images'].append(image_info)
                ret_g_crop['images'].append(image_info)

                cur_UAV_sample['image'] = img_dir
                cur_UAV_sample['image_id'] = image_id

                shift_mats = OrderedDict()
                for scale in [1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32, 64]:
                    cur_shift_mat, _ = get_crop_shift_mat(tranlation=camera_position.copy(), \
                                                        rotation=camera_rotation.copy(), \
                                                        map_scale_w=scale, map_scale_h=scale, \
                                                        world_X_left=world_X_left, world_Y_left=world_Y_left) 
                    shift_mats[scale] = cur_shift_mat
                cur_UAV_sample['shift_mats'] = shift_mats

                cat_id = 1
                vehicles_i = []
                vehicles_g = []
                vehicles_g_corners = []
                vehicles_z = []
                category_id = []

                cur_anno = []
                cur_g_crop_anno = []
                for vehicle_cord in vehicle_cords:
                    # get bbox from vehicle_cord
                    vehicle_cord_img = world2img(vehicle_cord.copy(), camera_rotation.copy(), camera_intrinsic.copy(), ori_camera_position.copy(), copy.deepcopy(offset))
                    
                    if vehicle_cord_img.shape[-1] == 0:
                        continue

                    x, y, w, h = get_2d_bounding_box(deepcopy(vehicle_cord_img))
                    # Check box
                    if x < 0 or y < 0 or (x + w) > W or (y + h) > H:
                        continue
                    
                    bbox_id += 1
                    ignore = 0
                    ann = { 'area': w * h,
                            'iscrowd': 0,
                            'image_id': image_id,
                            'bbox': [x, y, w, h],
                            'category_id': cat_id,
                            'id': bbox_id,
                            'ignore': ignore,
                            'segmentation': []}
                    ret_i['annotations'].append(ann)
                    vehicles_i.append([x, y, w, h])
                    category_id.append(cat_id)

                    cur_anno.append(ann)

                    vehicle_grid = WorldCoord2WorldGrid(vehicle_cord[:3, :].copy(), scale_w=1, scale_h=1, world_X_left=world_X_left, world_Y_left=world_Y_left)
                    vehicle_z = [vehicle_cord[2].min()-ground_height, vehicle_cord[2].max()-ground_height]

                    x, y, w, h = get_2d_bounding_box(vehicle_grid[:3])
                    polygon_xywhcs, corners = get_angle_polygon(deepcopy(vehicle_grid[:2,:4]))

                    ann = { 'area': w * h,
                            'iscrowd': 0,
                            'image_id': image_id,
                            'bbox': [x, y, w, h],
                            'polygon': polygon_xywhcs,
                            'corners': corners,
                            'category_id': cat_id,
                            'id': bbox_id,
                            'ignore': ignore,
                            'segmentation': []}
                    ret_g['annotations'].append(ann)

                    vehicles_g.append([x, y, w, h])
                    vehicles_g_corners.append(corners)
                    vehicles_z.append(vehicle_z)

                    vehicle_grid_crop = get_shift_coord(vehicle_grid, shift_mats[1])
                    # vehicle_grid_crop = shift_mats[1] @ np.concatenate([vehicle_grid.copy(), vehicle_cord[2:3, :]], axis=0)
                    # vehicle_grid_crop = shift_mats[1] @ np.concatenate([vehicle_grid.copy(), np.ones([1, vehicle_cord.shape[-1]])], axis=0)
                    # vehicle_grid_crop = shift_mats[1] @ np.concatenate([vehicle_grid.copy(), -np.ones([1, vehicle_cord.shape[-1]])*30], axis=0)
                    # vehicle_grid_crop = vehicle_grid_crop[:3] / vehicle_grid_crop[2, :]

                    x_crop, y_crop, w_crop, h_crop = get_2d_bounding_box(vehicle_grid_crop[:3])
                    polygon_xywhcs_crop, corners_crop = get_angle_polygon(deepcopy(vehicle_grid_crop[:2,:4]))
                    ann_crop = { 'area': w_crop * h_crop,
                            'iscrowd': 0,
                            'image_id': image_id,
                            'bbox': [x_crop, y_crop, w_crop, h_crop],
                            'polygon': polygon_xywhcs_crop,
                            'corners': corners_crop,
                            'category_id': cat_id,
                            'id': bbox_id,
                            'ignore': 0,
                            'segmentation': []}
                    ret_g_crop['annotations'].append(ann_crop)

                    cur_g_crop_anno.append(ann_crop)
                # if scene_no == 5:
                # ori_image = cv2.imread(os.path.join(data_dir,img_dir))
                # ori_image = cv2.resize(ori_image, (720, 480))
                # image = vis_cam(ori_image.copy(), cur_anno)
                # cv2.imwrite('realdata_test_i.png', image)

                # img_warp_center = kornia.image_to_tensor(ori_image.copy(), keepdim=False)
                # img_warp_center = kornia.warp_perspective(img_warp_center.float(),
                #                                     torch.tensor(shift_mats[1]@np.linalg.inv(cur_UAV_sample['trans_mat'])).repeat([1, 1, 1]).float(),
                #                                     dsize=(int(96/(1/1)), int(128/(1/1))))
                # img_warp_center = kornia.tensor_to_image(img_warp_center.byte())
                # img_warp_center = vis_cam(img_warp_center.copy(), cur_g_crop_anno)
                # cv2.imwrite('realdata_test_g.png', img_warp_center)


                cur_UAV_sample['vehicles_i'] = np.array(vehicles_i)
                cur_UAV_sample['vehicles_g'] = np.array(vehicles_g)
                cur_UAV_sample['vehicles_g_corners'] = np.array(vehicles_g_corners)
                cur_UAV_sample['vehicles_z'] = np.array(vehicles_z)
                cur_UAV_sample['category_id'] = np.array(category_id)
                cur_sample_info['{}_{}'.format('FRONT', 0)] = cur_UAV_sample
                
                ret_s['samples'].append(cur_sample_info)

        
        print("# samples: ", len(ret_s['samples']))
        print("# images: ", len(ret_i['images']))
        print("# annotations: ", len(ret_i['annotations']))
        # out_path = 'C:/Users/35387/Desktop/airsim_camera_demo/airsim_instances_{}.json'.format(split)
        out_path = os.path.join(data_dir, 'multiagent_annotations/{}_{}_instances_woignoredbox.json'.format(height, split))
        out_global_path = os.path.join(data_dir, 'multiagent_annotations/{}_{}_instances_global_woignoredbox.json'.format(height, split))
        out_sample_path = os.path.join(data_dir, 'multiagent_annotations/{}_{}_instances_sample.pkl'.format(height, split))
        out_global_crop_path = os.path.join(data_dir, 'multiagent_annotations/{}_{}_instances_global_crop_woignoredbox.json'.format(height, split))
        json.dump(ret_i, open(out_path, 'w'))
        json.dump(ret_g, open(out_global_path, 'w'))
        json.dump(ret_g_crop, open(out_global_crop_path, 'w'))
        pkl.dump(ret_s, open(out_sample_path, 'wb'))

def pop_ignored_box(data_dir='/DATA7_DB7/data/shfang/airsim_camera_seg_15/', height=40):
    splits = ['train', 'val']
    for split in splits:
        out_path = os.path.join(data_dir, 'multiagent_annotations/{}_{}_instances.json'.format(height, split))
        out_global_path = os.path.join(data_dir, 'multiagent_annotations/{}_{}_instances_global.json'.format(height, split))
        out_global_crop_path = os.path.join(data_dir, 'multiagent_annotations/{}_{}_instances_global_crop.json'.format(height, split))
        for anno_path in [out_path, out_global_path, out_global_crop_path]:
            with open(anno_path, 'r') as f:
                annos = json.load(f)
            annos_noignore = []
            for anno in tqdm(annos['annotations']):
                if anno['ignore'] == 0:
                    annos_noignore.append(anno)
            annos['annotations'] = annos_noignore
            save_path = os.path.join(os.path.dirname(anno_path), os.path.basename(anno_path).split('.')[0]+'_woignoredbox.json')
            print(save_path)
            with open(save_path, 'w') as f:
                json.dump(annos, f)

def vis_cam(image, annos, color=(127, 255, 0), vis_thre=-1):
    for anno in annos:
        if (anno.get('score', 0) > vis_thre) and (not anno.get('ignore', 0)):
            if 'corners' in anno:
                polygon = np.array(anno['corners'][:8]).reshape([4,2])
                image = cv2.polylines(image, pts=np.int32([polygon.reshape(-1, 1, 2)]), isClosed=True, color=color, thickness=1)
            else:
                bbox = anno['bbox']
                if len(bbox) == 4:
                    # bbox = [x*4 for x in bbox]
                    x, y, w, h = bbox
                    image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 1)
                else:
                    polygon = np.array(get_2d_polygon(np.array(bbox[:8]).reshape([4,2]).T)).reshape([4,2])
                    polygon = polygon * 4
                    image = cv2.polylines(image, pts=np.int32([polygon.reshape(-1, 1, 2)]), isClosed=True, color=color, thickness=1)
                
    return image

if __name__ == '__main__':
    # convert_multiview_coco(town_id=0, height=40)
    # convert_multiview_coco(town_id=1, height=40)
    # convert_multiview_coco(town_id=1, height=60)
    # convert_multiview_coco(town_id=1, height=80)
    # convert_multiview_coco(town_id=1, height=100)
    # convert_multiview_coco(town_id=2, height=40)
    # convert_multiview_coco(town_id=2, height=60)
    # convert_multiview_coco(town_id=2, height=80)
    convert_multiview_coco(town_id=0, height=40)
    # town_ids = [0]
    # for town_id in town_ids:
    #     if town_id == 0:
    #         data_dir = '/DB/public/uav_dataset'
    #     pop_ignored_box(data_dir)