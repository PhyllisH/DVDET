'''
Author: yhu
Contact: phyllis1sjtu@outlook.com
LastEditTime: 2021-06-28 15:00:47
Description: 
'''
'''
Input: results.json multi-agent in uav coord
Output1: results.json sample in global coord
Output2: results.json agents in uav coord with the merged result
'''

'''
Functions:
    - coord trans
    - nms
    - 
'''

from io import TextIOWrapper
import re
from unicodedata import category
from ipdb.__main__ import set_trace
import numpy as np
import os
import json
import pickle as pkl
import math
import cv2
import random
import kornia
from numpy.core.fromnumeric import resize
import torch
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import sys
from torch.utils import data
from tqdm import tqdm
import copy
from collections import OrderedDict

sys.path.append(__file__)
sys.path.append(os.path.join(__file__, '../'))
sys.path.append('/GPFS/data/yhu/code/CoDet/src/lib/external/')
try:
    from nms import nms, soft_nms
except:
    print('NMS not imported! If you need it,'
          ' do \n cd $CenterNet_ROOT/src/lib/external \n make')

from transformation import get_imgcoord2worldgrid_matrices, get_imgcoord_matrices, get_worldcoord_from_imagecoord, get_crop_shift_mat, get_2d_polygon


####################### Default Settings #########################
MAX_PER_IMAGE = 100
# dataset_dir = '/DATA7_DB7/data/shfang/airsim_camera_seg_15'
dataset_dir = '/GPFS/data/yhu/Dataset/airsim_camera/airsim_camera_seg_15'
# dataset_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data/airsim_camera_10scene')
camera_intrinsic = [[400.0, 0.0, 400.0],
                        [0.0, 400.0, 225.0],
                        [0.0, 0.0, 1.0]]
camera_intrinsic = np.array(camera_intrinsic)
# worldgrid2worldcoord_mat = np.array([[1, 0, -100], [0, 1, -100], [0, 0, 1]])
# scale_h = 450/500 * 1
# scale_w = 800/500 * 1
# scale_h = 500/500 
# scale_w = 500/500
scale_h = 1
scale_w = 1
map_scale_h = 1 /4 * scale_h
map_scale_w = 1 /4 * scale_w
# world_X_left = 200
# world_Y_left = 250
with_rotat = True
# worldgrid2worldcoord_mat = np.array([[1/scale_w, 0, -world_X_left], [0, 1/scale_h, -world_Y_left], [0, 0, 1]])
# default_worldgrid2worldcoord_mat = np.array([[500/800, 0, -200], [0, 500/450, -250], [0, 0, 1]])
# default_worldgrid2worldcoord_mat = np.array([[1, 0, -world_X_left], [0, 1, -world_Y_left], [0, 0, 1]])
# image_size = (int(500*scale_h), int(500*scale_w))
image_size = (int(192/map_scale_h), int(352/map_scale_w))
# image_size = (int(500*scale_h), int(300*scale_w))
# image_size = (225, 400)
img_h = 450
img_w = 800

town_config = OrderedDict()
town_config[5] = {
    'world_X': 500,
    'world_Y': 500,
    'world_X_left': 200,
    'world_Y_left': 250,
    'dataset_dir': '/GPFS/data/yhu/Dataset/airsim_camera/airsim_camera_seg_15'
}
town_config[6] = {
    'world_X': 1200,
    'world_Y': 600,
    'world_X_left': 400,
    'world_Y_left': 200,
    'dataset_dir': '/GPFS/data/shfang/dataset/airsim_camera/airsim_camera_seg_town6_v2'
}
town_config[4] = {
    'world_X': 900,
    'world_Y': 900,
    'world_X_left': 400,
    'world_Y_left': 500,
    'dataset_dir': '/GPFS/data/shfang/dataset/airsim_camera/airsim_camera_seg_town4_v2_40m'
}

worldgrid2worldcoord_mats = OrderedDict()
default_worldgrid2worldcoord_mats = OrderedDict()
world_X_lefts = OrderedDict()
world_Y_lefts = OrderedDict()
for town_id in [5, 6, 4]:
    world_X = town_config[town_id]['world_X']
    world_Y = town_config[town_id]['world_Y']
    image_H = 450
    image_W = 800
    world_X_lefts[town_id] = town_config[town_id]['world_X_left']
    world_Y_lefts[town_id] = town_config[town_id]['world_Y_left']
    # scale_h = image_H/world_Y
    # scale_w = image_W/world_X
    # worldgrid2worldcoord_mats[town_id] = np.array([[world_X/image_W, 0, -world_X_left], [0, world_Y/image_H, -world_Y_left], [0, 0, 1]])
    worldgrid2worldcoord_mats[town_id] = np.array([[map_scale_w, 0, -town_config[town_id]['world_X_left']], [0, map_scale_h, -town_config[town_id]['world_Y_left']], [0, 0, 1]])
    default_worldgrid2worldcoord_mats[town_id] = np.array([[1, 0, -town_config[town_id]['world_Y_left']], [0, 1, -town_config[town_id]['world_Y_left']], [0, 0, 1]])
    data_dir = town_config[town_id]['dataset_dir']

##################################################################

def BoxCoordTrans(coord, translation, rotation, mode='L2G', with_rotat=False, sensor_type="BOTTOM", town_id=5):
    project_mat = get_imgcoord_matrices(translation.copy(), rotation.copy(), camera_intrinsic)
    # project_mat = project_mat @ default_worldgrid2worldcoord_mats[town_id]
    # project_mat = project_mat @ default_worldgrid2worldcoord_mat
    # project_mat = project_mat @ worldgrid2worldcoord_mat
    project_mat = project_mat @ worldgrid2worldcoord_mats[town_id]
    
    if with_rotat:
        rotat_mat = get_crop_shift_mat(translation.copy(), rotation.copy(), sensor_type, map_scale_w, map_scale_h, world_X_lefts[town_id], world_Y_lefts[town_id])

    if mode == 'L2G':
        project_mat = rotat_mat @ np.linalg.inv(project_mat) if with_rotat else np.linalg.inv(project_mat)
    else:
        project_mat = project_mat @ np.linalg.inv(rotat_mat) if with_rotat else project_mat

    coord = np.concatenate([coord[:2], np.ones([1, coord.shape[1]])], axis=0)
    coord_warp = project_mat @ coord
    coord_warp = coord_warp / coord_warp[2, :]
    return coord_warp

def ImgCoordTrans(image, translation, rotation, mode='L2G', with_rotat=False, sensor_type='BOTTOM', town_id=5):
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
                                                      worldgrid2worldcoord_mat=worldgrid2worldcoord_mats[town_id])
    
    if mode == 'L2G':
        trans_mat = project_mat
    else:
        trans_mat = np.linalg.inv(project_mat)
    
    if with_rotat:
        rotat_mat = get_crop_shift_mat(translation.copy(), rotation.copy(), sensor_type, map_scale_w, map_scale_h, world_X_lefts[town_id], world_Y_lefts[town_id])
        trans_mat = rotat_mat@trans_mat
    data = kornia.image_to_tensor(image, keepdim=False)
    data_warp = kornia.warp_perspective(data.float(),
                                        torch.tensor(trans_mat).repeat([1, 1, 1]).float(),
                                        dsize=image_size)

    # convert back to numpy
    img_warp = kornia.tensor_to_image(data_warp.byte())
    return img_warp

def vis_cam(image, annos, color=(127, 255, 0), vis_thre=-1, scale=1):
    for anno in annos:
        if anno.get('score', 0) > vis_thre:
            bbox = anno['bbox']
            bbox = [x*scale for x in bbox]
            if len(bbox) == 4:
                x, y, w, h = bbox
                image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 1)
            else:
                polygon = np.array(get_2d_polygon(np.array(bbox[:8]).reshape([4,2]).T)).reshape([4,2])
                image = cv2.polylines(image, pts=np.int32([polygon.reshape(-1, 1, 2)]), isClosed=True, color=color, thickness=1)
    return image

def test_coord(sensor, coco_, annos, trans_annos, gt_annos=None, gt_trans_annos=None, vis_score_thre=0.3, town_id=5):
    img_id = sensor['image_id']
    sensor_type = sensor['image'].split('/')[1].split('_')[1]
    img = coco_.loadImgs(img_id)[0]
    image_u = cv2.imread(os.path.join(town_config[town_id]['dataset_dir'], img['file_name']))
    image_g = ImgCoordTrans(image_u.copy(), sensor['translation'].copy(), sensor['rotation'].copy(), mode='L2G', with_rotat=with_rotat, sensor_type=sensor_type, town_id=town_id)
    image_up = vis_cam(image_u.copy(), annos, color=(0, 0, 255), vis_thre=vis_score_thre)
    if gt_annos is not None:
        image_up = vis_cam(image_up.copy(), gt_annos, color=(0, 255, 0), vis_thre=-1)
    image_gp = vis_cam(image_g.copy(), trans_annos, color=(0, 0, 255), vis_thre=vis_score_thre, scale=1)
    if gt_trans_annos is not None:
        image_gp = vis_cam(image_gp.copy(), gt_trans_annos, color=(0, 255, 0), vis_thre=-1, scale=1)
    cv2.imwrite('test.png', image_up)
    cv2.imwrite('test_g.png', image_gp)
    
def xywh2polygon(bbox):
    x, y, w, h = bbox
    left_up = [x, y]
    left_bottom = [x, y+h]
    right_up = [x+w, y]
    right_bottom = [x+w, y+h]
    return np.array([left_up, left_bottom, right_bottom, right_up]).reshape([4,2]).T

def xywh2xyxy(bbox):
    x, y, w, h = bbox
    return [x, y, x+w, y+h]

def xyxy2xywh(bbox):
    x0, y0, x1, y1 = bbox
    return [x0, y0, x1-x0, y1-y0]

def polygon2xywh(bbox):
    x_min = min(bbox[0])
    x_max = max(bbox[0])
    y_min = min(bbox[1])
    y_max = max(bbox[1])
    return [x_min, y_min, x_max-x_min, y_max-y_min]

def ImgAnnoCoordTrans(annos, tranlation, rotation, mode='L2G', with_rotat=False, sensor_type='BOTTOM', town_id=5):
    trans_annos_bbox = copy.deepcopy(annos)
    trans_annos_polygon = copy.deepcopy(annos)
    for i in range(len(annos)):
        bbox = annos[i]['bbox']
        if not isinstance(bbox, np.ndarray):
            polygon = xywh2polygon(bbox)    # [2, 4]
        else:
            polygon = bbox
        trans_polygon = BoxCoordTrans(polygon.copy(), tranlation, rotation, mode, with_rotat, sensor_type, town_id)
        trans_annos_polygon[i]['bbox'] = list(trans_polygon[:2].T.reshape([-1,])) + [annos[i]['score']]
        trans_annos_bbox[i]['bbox'] = polygon2xywh(trans_polygon)
    return trans_annos_bbox, trans_annos_polygon


def UAV2BEV(result_path, gt_path, gt_g_path, sample_path, data_dir, towns2id, save_town_id=5):
    res_annos_all = json.load(open(result_path))
    gt_annos_all = json.load(open(gt_path))['annotations']
    gt_g_annos_all = json.load(open(gt_g_path))['annotations']
    coco_ = coco.COCO(gt_path)
    catIds = coco_.getCatIds()
    imgIds = coco_.getImgIds()

    # samples = pkl.load(open(sample_path, 'rb'))['samples']
    if data_dir.endswith('pkl'):
        samples = pkl.load(open(sample_path, 'rb'))
        img_dir = pkl.load(open(data_dir, 'rb'))
    else:
        samples = pkl.load(open(sample_path, 'rb'))['samples']
        img_dir = data_dir

    res_annos_all_BEV = []
    res_annos_all_BEV_polygon = []
    for sample_id, sample in tqdm(enumerate(samples)):
        for k, sensor in sample.items():
            if k.startswith('vehicles'):
                continue
            else:
                if sensor['image_id'] in imgIds:
                    if isinstance(img_dir, list):
                        town_id = towns2id[img_dir[sample_id]]
                    else:
                        town_id = towns2id[img_dir]
                    res_annos = [anno.copy() for anno in res_annos_all if anno['image_id'] == sensor['image_id']]
                    gt_annos = [anno.copy() for anno in gt_annos_all if anno['image_id'] == sensor['image_id']]
                    gt_trans_annos = [anno.copy() for anno in gt_g_annos_all if anno['image_id'] == sensor['image_id']]
                    res_annos_BEV_bbox, res_annos_BEV_polygon = ImgAnnoCoordTrans(res_annos.copy(), sensor['translation'].copy(), sensor['rotation'].copy(), mode='L2G', with_rotat=with_rotat, sensor_type=sensor['image'].split('/')[1].split('_')[1],town_id=town_id)
                    # if town_id !=5:
                    # test_coord(sensor, coco_, res_annos, res_annos_BEV_polygon, gt_annos, gt_trans_annos, vis_score_thre=0.3, town_id=town_id)
                    # import ipdb; ipdb.set_trace()
                    # test_coord(sensor, coco_, res_annos, gt_annos, vis_score_thre=-1)
                    res_annos_all_BEV.extend(res_annos_BEV_bbox)
                    res_annos_all_BEV_polygon.extend(res_annos_BEV_polygon)
    with open(os.path.join(os.path.dirname(result_path), 'results_BEV_{}.json'.format(save_town_id)), 'w') as f:
        json.dump(res_annos_all_BEV, f)
    with open(os.path.join(os.path.dirname(result_path), 'results_BEV_Polygon_{}.json'.format(save_town_id)), 'w') as f:
        json.dump(res_annos_all_BEV_polygon, f)
    return res_annos_all_BEV

def LateFuse(result_path, gt_path, sample_path):
    res_annos_all = json.load(open(result_path))
    coco_ = coco.COCO(gt_path)
    catIds = coco_.getCatIds()
    imgIds = coco_.getImgIds()
    samples = pkl.load(open(sample_path, 'rb'))['samples']

    res_annos_all_BEV = []
    res_annos_all_UAV_LateFused = []
    for sample_id, sample in tqdm(enumerate(samples)):
        BEV_polygon = []
        for k, sensor in sample.items():
            if k.startswith('vehicles'):
                continue
            else:
                if sensor['image_id'] in imgIds:
                    res_annos = [anno.copy() for anno in res_annos_all if anno['image_id'] == sensor['image_id']]
                    res_annos_BEV_bbox, res_annos_BEV_polygon = ImgAnnoCoordTrans(res_annos.copy(), sensor['translation'].copy(), sensor['rotation'].copy(), mode='L2G', with_rotat=with_rotat, sensor_type=sensor['image'].split('/')[1].split('_')[1])
                    # test_coord(sensor, coco_, res_annos, res_annos_BEV_bbox, vis_score_thre=0.3)
                    res_annos_all_BEV.extend(res_annos_BEV_bbox)
                    BEV_polygon.extend(res_annos_BEV_polygon)
        # BEV Polygons mapped to UAV coord
        for k, sensor in sample.items():
            if k.startswith('vehicles'):
                continue
            else:
                if sensor['image_id'] in imgIds:
                    res_annos = [anno.copy() for anno in res_annos_all if anno['image_id'] == sensor['image_id']]
                    UAV_bbox, _ = ImgAnnoCoordTrans(BEV_polygon.copy(), sensor['translation'].copy(), sensor['rotation'].copy(), mode='G2L', with_rotat=with_rotat, sensor_type=sensor['image'].split('/')[1].split('_')[1])
                    UAV_bbox = [x for x in UAV_bbox if x['bbox'][0] >= 0 and x['bbox'][1] >= 0 and (x['bbox'][0]+x['bbox'][2] < img_w) and (x['bbox'][1]+x['bbox'][3]) < img_h]
                    res_annos = [x for x in res_annos if x['bbox'][0] >= 0 and x['bbox'][1] >= 0 and (x['bbox'][0]+x['bbox'][2] < img_w) and (x['bbox'][1]+x['bbox'][3]) < img_h]
                    res = res_annos + UAV_bbox
                    print('res: ',  len(res))
                    if len(res) <= 1:
                        print(sensor['image_id'])
                        res_annos_all_UAV_LateFused.extend(res)
                        continue
                    res_xyxy = {}
                    for detection in res:
                        if detection['category_id'] in res_xyxy:
                            res_xyxy[detection['category_id']].append(np.array([xywh2xyxy(detection['bbox'])+[detection['score']]]))
                        else:
                            res_xyxy[detection['category_id']] = [np.array([xywh2xyxy(detection['bbox'])+[detection['score']]])]
                    UAV_LateFused = []
                    for category_id in res_xyxy:
                        res_xyxy[category_id] = np.concatenate(res_xyxy[category_id], axis=0).astype(np.float32)
                        soft_nms(res_xyxy[category_id], Nt=0.5, method=2)
                        res_xyxy[category_id] = res_xyxy[category_id][res_xyxy[category_id][:,-1].argsort()[::-1]]
                        for i in range(len(res_xyxy[category_id])):
                            bbox_after_nms = {}
                            if res_xyxy[category_id][i][0] >= 0 and res_xyxy[category_id][i][1] >= 0 and res_xyxy[category_id][i][2] < img_w and res_xyxy[category_id][i][3] < img_h:
                                bbox_after_nms['image_id'] = sensor['image_id']
                                bbox_after_nms['category_id'] = category_id
                                bbox_after_nms['bbox'] = xyxy2xywh([float(x) for x in res_xyxy[category_id][i][:4]])
                                bbox_after_nms['score'] = float(res_xyxy[category_id][i][-1])
                                UAV_LateFused.append(bbox_after_nms)
                    # print('nms: ', len(UAV_LateFused), 'score: {:.02f}'.format(UAV_LateFused[0]['score']))
                    # import ipdb; ipdb.set_trace()
                    res_annos_all_UAV_LateFused.extend(UAV_LateFused[:max(len(UAV_LateFused), MAX_PER_IMAGE)])

    with open(os.path.join(os.path.dirname(result_path), 'results_BEV.json'), 'w') as f:
        json.dump(res_annos_all_BEV, f)

    with open(os.path.join(os.path.dirname(result_path), 'results_LateFused.json'), 'w') as f:
        json.dump(res_annos_all_UAV_LateFused, f)
    return res_annos_all_BEV, res_annos_all_UAV_LateFused

def diff_Fuse(gt_g_path, result_path):
    with open(os.path.join(os.path.dirname(result_path), 'results_BEV.json'), 'r') as f:
        single_results = json.load(f)
    # with open(os.path.join(os.path.dirname(result_path), 'results_LateFused.json'), 'r') as f:
    #     multi_results = json.load(f)
    with open(gt_g_path, 'r') as f:
        multi_results = json.load(f)['annotations']

    single_img_idx = list(set([x['image_id'] for x in single_results]))
    multi_img_idx = list(set([x['image_id'] for x in multi_results]))

    import ipdb; ipdb.set_trace()
    assert single_img_idx == multi_img_idx

    for i in range(len(single_results)):
        single = single_results[i]
        multi = multi_results[i]
        # import ipdb; ipdb.set_trace()
        if np.abs(np.array(single['bbox']) - np.array(multi['bbox'])).sum() > 1e-5:
            print(i, single['bbox'], multi['bbox'])
        if single['image_id'] != multi['image_id']:
            print(i, single['image_id'], multi['image_id'])
        if single['category_id'] != multi['category_id']:
            print(i, single['category_id'], multi['category_id'])
        if abs(single['score'] - multi['score']) > 1e-5:
            print(i, single['score'], multi['score'])


    import ipdb; ipdb.set_trace()
    

def run_eval(result_path, gt_path, gt_g_path):
    # Single-Agent Det in UAV Coord
    print('############### Single-Agent Det in UAV Coord ############')
    print('single-agent path: ', result_path)
    coco_ = coco.COCO(gt_path)
    coco_dets = coco_.loadRes(result_path)
    coco_eval = COCOeval(coco_, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print('##########################################################')

    # Single-Agent Det in BEV Coord
    print('###### Single-Agent Det in BEV Coord from UAV Coord ######')
    coco_g_ = coco.COCO(gt_g_path)
    coco_dets = coco_g_.loadRes(os.path.join(os.path.dirname(result_path), 'results_BEV.json'))
    coco_eval = COCOeval(coco_g_, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print('##########################################################')

    # Single-Agent Det in BEV Coord
    print('############### Single-Agent Det in BEV Coord ############')
    coco_g_ = coco.COCO(gt_g_path)
    coco_dets = coco_g_.loadRes(os.path.join(os.path.dirname(result_path), 'results_Global.json'))
    coco_eval = COCOeval(coco_g_, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print('##########################################################')

    # Single-Agent Det in BEV Coord
    # print('############### Multi-Agent Det in UAV Coord ############')
    # # coco_ = coco.COCO(gt_path)
    # print('multi-agent path: ', os.path.join(os.path.dirname(result_path), 'results_LateFused.json'))
    # coco_dets = coco_.loadRes(os.path.join(os.path.dirname(result_path), 'results_LateFused.json'))
    # coco_eval = COCOeval(coco_, coco_dets, "bbox")
    # coco_eval.evaluate()
    # coco_eval.accumulate()
    # coco_eval.summarize()
    # print('##########################################################')


if __name__ == '__main__':
    towns = {
        5: '/GPFS/data/yhu/Dataset/airsim_camera/airsim_camera_seg_15',
        6: '/GPFS/data/shfang/dataset/airsim_camera/airsim_camera_seg_town6_v2',
        4: '/GPFS/data/shfang/dataset/airsim_camera/airsim_camera_seg_town4_v2_40m/'
    }
    towns2id = {y:x for x,y in towns.items()}

    # data_dir = '/DATA5_DB8/data/public/airsim_camera/airsim_camera_10scene'
    # data_dir = '/DATA7_DB7/data/shfang/airsim_camera_seg_15/'
    # data_dir = '/GPFS/data/yhu/Dataset/airsim_camera/airsim_camera_seg_15'

    # gt_path = os.path.join(data_dir, 'multiagent_annotations/{}_val_instances.json'.format(40))
    # gt_g_path = os.path.join(data_dir, 'multiagent_annotations/{}_val_instances_global_crop_woignoredbox.json'.format(40))
    # sample_path = os.path.join(data_dir, 'multiagent_annotations/{}_val_instances_sample.pkl'.format(40))

    # gt_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/multiagent_det/LocalCoord_repeat/gts_Local.json')
    
    # result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/multiagent_det/dla_multiagent_withwarp_GlobalCoord_GTFeatMap_352_192_JointUAVBEVGT_40m_Town5_Baseline_MapScale1_B23/results_Local.json')
    # result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/multiagent_det/LocalCoord_repeat/results_Local.json')
    # result_LateFused_path = os.path.join(os.path.dirname(result_path), 'results_LateFused.json')

    # sample_path = os.path.join(os.path.dirname(__file__), '..', 'gts.pkl')
    # data_path = os.path.join(os.path.dirname(__file__), '..', 'imgs_dir.pkl')

    # _, _ = LateFuse(result_path, gt_path, sample_path)
    # UAV2BEV(result_path, gt_path, gt_g_path, sample_path, data_path, towns2id)
    # run_eval(result_path, gt_path, gt_g_path)
    # diff_Fuse(gt_g_path, result_path)

    for town_id in [4,5,6]:
        data_dir = towns[town_id]
        gt_path = os.path.join(data_dir, 'multiagent_annotations/{}_val_instances_woignoredbox.json'.format(40))
        gt_g_path = os.path.join(data_dir, 'multiagent_annotations/{}_val_instances_global_crop_woignoredbox.json'.format(40))
        sample_path = os.path.join(data_dir, 'multiagent_annotations/{}_val_instances_sample.pkl'.format(40))
        result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exp/multiagent_det/LocalCoord_repeat/results_Local_Town{}.json'.format(town_id))
        UAV2BEV(result_path, gt_path, gt_g_path, sample_path, data_dir, towns2id, save_town_id=town_id)