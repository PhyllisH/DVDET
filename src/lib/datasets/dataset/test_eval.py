'''
Author: yhu
Contact: phyllis1sjtu@outlook.com
LastEditTime: 2021-07-25 20:15:43
Description: 
'''

from collections import OrderedDict
import json
from logging import disable
import os
import re
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
from utils.eval_utils import eval_map
from copy import copy, deepcopy
import cv2
import pickle as pkl
from tqdm import tqdm


def todict(annos):
    annotations = {}
    for polygon in annos:
        image_id = polygon['image_id']
        category_id = polygon['category_id']
        if image_id not in annotations:
            annotations[image_id] = {}
        
        cur_polygon = polygon['corners'] if 'corners' in polygon else polygon['bbox']
        if category_id in annotations[image_id]:
            annotations[image_id][category_id].append(cur_polygon)
        else:
            annotations[image_id][category_id] = [cur_polygon]
    return annotations

def filter_polygon(polygons):
    x_min = polygons[:,::2].min(axis=1)
    y_min = polygons[:,1::2].min(axis=1)
    x_max = polygons[:,::2].max(axis=1)
    y_max = polygons[:,1::2].max(axis=1)

    keep = np.where((x_min >= 0) * (y_min > 0) * (x_max <= 800) * (y_max <= 450))[0]

    if len(keep) == 0:
        return []
    elif len(keep) == 1:
        return polygons[keep][None,]
    else:
        return polygons[keep]

def keep_single_polygon(polygon, score_thre=0.3):
    if len(polygon) == 9:
        score = polygon[-1]
    else:
        score = 1.0
    x_min = min(polygon[:8][::2])
    x_max = max(polygon[:8][::2])
    y_min = min(polygon[:8][1::2])
    y_max = max(polygon[:8][1::2])
    if (x_min >= 0) and (y_min > 0) and (x_max <= 800) and (y_max <= 450) and score >= score_thre:
        return True
    else:
        return False


def toevalformat(annos):
    image_idx = list(annos.keys())
    image_idx.sort()
    category_set = [list(x.keys()) for _, x in annos.items()]
    counts = np.array([len(x) for x in category_set])
    category_idx = category_set[np.argmax(counts)]
    annotations = []
    # print(image_idx)
    # print(category_idx)
    for i in range(len(image_idx)):
        image_id = image_idx[i]
        if image_id in annos:
            annotations.append([])
            for j in range(len(category_idx)):
                category_id = category_idx[j]
                if category_id in annos[image_id]:
                    if len(annos[image_id][category_id]) > 1:
                        annos[image_id][category_id] = np.concatenate([np.array(x).reshape([1, len(x)]) for x in annos[image_id][category_id]], axis=0)
                    else:
                        annos[image_id][category_id] = np.array(annos[image_id][category_id]).reshape([1, len(annos[image_id][category_id][0])])
                    annotations[i].append(filter_polygon(annos[image_id][category_id]))
                else:
                    annotations[i].append([])
    return annotations

def run_polygon_eval(anno_path_cocoformat, det_path_cocoformat):
    if isinstance(det_path_cocoformat, str):
        with open(det_path_cocoformat, 'r') as f:
            det_results = json.load(f)
            # for anno in det_results:
            #     bbox = anno['bbox'][:8]
            #     anno['bbox'][:8] = [x/4.0 for x in bbox]
    else:
        det_results = det_path_cocoformat

    if isinstance(anno_path_cocoformat, str):
        with open(anno_path_cocoformat, 'r') as f:
            gt_annos = json.load(f)
    else:
        gt_annos = anno_path_cocoformat

    
    det_evalformat = todict(det_results)
    gt_evalformat = todict(gt_annos['annotations'])

    for image_id in det_evalformat:
        if image_id not in gt_evalformat:
            gt_evalformat[image_id] = {}

    det_evalformat = toevalformat(det_evalformat)
    gt_evalformat = toevalformat(gt_evalformat)

    # import ipdb; ipdb.set_trace()

    iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
    
    print('####################### IOU AVG ####################### ')
    mean_aps = []
    for iouthr in iouThrs:
        mean_ap, _ = eval_map(deepcopy(det_evalformat), deepcopy(gt_evalformat), iou_thr=iouthr, mode='area', nproc=8, print_flag=False)
        print('AP@{:.02f}: {:.04f}'.format(iouthr, mean_ap[0]))
        mean_aps.append(mean_ap[0])

    print('AP@0.5:0.95: {:.04f}'.format(np.mean(np.array(mean_aps))))

    # print('####################### IOU 0.75 ####################### ')
    # eval_map(det_evalformat, gt_evalformat, iou_thr=0.75, mode='area', nproc=8)

    # print('####################### IOU 0.5 ####################### ')
    # eval_map(det_evalformat, gt_evalformat, iou_thr=0.5, mode='area', nproc=8)
    # eval_map(det_evalformat, gt_evalformat, iou_thr=0.5, mode='ap', nproc=8)

    # print('####################### IOU 0.25 ####################### ')
    # eval_map(det_evalformat, gt_evalformat, iou_thr=0.25, mode='area', nproc=8)
    # eval_map(det_evalformat, gt_evalformat, iou_thr=0.25, mode='ap', nproc=8)

def split_towns(res_annos_all, gt_annos_all):
    towns = {
        5: '/GPFS/data/yhu/Dataset/airsim_camera/airsim_camera_seg_15',
        6: '/GPFS/data/shfang/dataset/airsim_camera/airsim_camera_seg_town6_v2',
        4: '/GPFS/data/shfang/dataset/airsim_camera/airsim_camera_seg_town4_v2_40m/'
    }
    towns2id = {y:x for x,y in towns.items()}

    town_res_annos = OrderedDict()
    town_gt_annos = OrderedDict()
    town_gt_images = OrderedDict()
    for town_id in towns:
        town_res_annos[town_id] = []
        town_gt_annos[town_id] = []
        town_gt_images[town_id] = []

    sample_path = os.path.join('/GPFS/data/yhu/code/BEV_Det/CoDet/src/gts.pkl')
    data_dir = os.path.join('/GPFS/data/yhu/code/BEV_Det/CoDet/src/imgs_dir.pkl')
    samples = pkl.load(open(sample_path, 'rb'))
    img_dir = pkl.load(open(data_dir, 'rb'))
    for sample_id, sample in tqdm(enumerate(samples)):
        for k, sensor in sample.items():
            if k.startswith('vehicles'):
                continue
            else:
                town_id = towns2id[img_dir[sample_id]]
                res_annos = [anno.copy() for anno in res_annos_all if anno['image_id'] == sensor['image_id']]
                gt_annos = [anno.copy() for anno in gt_annos_all['annotations'] if anno['image_id'] == sensor['image_id']]
                gt_images = [anno.copy() for anno in gt_annos_all['images'] if anno['id'] == sensor['image_id']]
                town_res_annos[town_id].extend(res_annos)
                town_gt_annos[town_id].extend(gt_annos)
                town_gt_images[town_id].extend(gt_images)
    return town_res_annos, town_gt_annos, town_gt_images

def get_splited_town(anno_path_cocoformat, det_path_cocoformat):
    if isinstance(det_path_cocoformat, str):
        with open(det_path_cocoformat, 'r') as f:
            det_results = json.load(f)
            for anno in det_results:
                bbox = anno['bbox'][:8]
                anno['bbox'][:8] = [x/4.0 for x in bbox]
    else:
        det_results = det_path_cocoformat

    if isinstance(anno_path_cocoformat, str):
        with open(anno_path_cocoformat, 'r') as f:
            gt_annos = json.load(f)
    else:
        gt_annos = anno_path_cocoformat
    
    town_res_annos, town_gt_annos, town_gt_images = split_towns(det_results, gt_annos)
    for town_id in town_res_annos:
        cur_gt_annos = gt_annos.copy()
        cur_gt_annos['annotations'] = town_gt_annos[town_id]
        cur_gt_annos['images'] = town_gt_images[town_id]

        with open(os.path.join(os.path.dirname(det_path_cocoformat), 'results_BEV_{}.json'.format(town_id)), 'w') as f:
            json.dump(town_res_annos[town_id], f)
        with open(os.path.join(os.path.dirname(det_path_cocoformat), 'gts_BEV_{}.json'.format(town_id)), 'w') as f:
            json.dump(cur_gt_annos, f)
    
    return


def split_accord_dis(anno_path_cocoformat, det_path_cocoformat):
    # distances = list(range(0, 220, 40))
    distances = [0, 50, 100, 200]
    print(distances)

    with open(anno_path_cocoformat, 'r') as f:
        gt_annos = json.load(f)
    
    with open(det_path_cocoformat, 'r') as f:
        det_results = json.load(f)
    
    gt_annos_all = gt_annos['annotations']
    gt_images_all = gt_annos['images']

    for start, end in zip(distances[:-1], distances[1:]):
        cur_gt_annos = [anno.copy() for anno in gt_annos_all if start<=anno['polygon'][1]<=end]
        cur_gt_img_ids = list(set([anno['image_id'] for anno in cur_gt_annos]))
        cur_gt_images = [anno.copy() for anno in gt_images_all if anno['id'] in cur_gt_img_ids]

        cur_gt = gt_annos.copy()
        cur_gt['annotations'] = cur_gt_annos
        cur_gt['images'] = cur_gt_images

        # cur_det_results = [anno.copy() for anno in det_results if anno['image_id'] in cur_gt_img_ids]
        cur_det_results = [anno.copy() for anno in det_results if anno['image_id'] in cur_gt_img_ids and (min(anno['bbox'][:8][1::2]) >=start or max(anno['bbox'][:8][1::2])<=end)]
        cur_det_img_ids = [anno['image_id'] for anno in cur_det_results]
        print(len(cur_gt_annos), len(cur_det_results))
        print(len(set(cur_gt_img_ids)), len(set(cur_det_img_ids)))
        # with open(anno_path_cocoformat.split('.')[0]+'_{}_{}.json'.format(start, end), 'w') as f:
        #     json.dump(cur_gt, f)

        print('####################### Distance: {} - {} ####################### '.format(200-end, 200-start))
        run_polygon_eval(cur_gt, cur_det_results)
    return

if __name__ == '__main__':
    towns = {
        5: '/GPFS/data/yhu/Dataset/airsim_camera/airsim_camera_seg_15',
        6: '/GPFS/data/shfang/dataset/airsim_camera/airsim_camera_seg_town6_v2',
        4: '/GPFS/data/shfang/dataset/airsim_camera/airsim_camera_seg_town4_v2_40m/'
    }
    # anno_path_cocoformat = '/GPFS/data/yhu/code/BEV_Det/CoDet/exp/multiagent_det/GlobalCoord_Inter/gts_Global.json'
    # det_path_cocoformat = '/GPFS/data/yhu/code/BEV_Det/CoDet/exp/multiagent_det/GlobalCoord_Inter/results_Global.json'
    # anno_path_cocoformat = '/DATA7_DB7/data/shfang/airsim_camera_seg_15/multiagent_annotations/val_instances_global.json'
    # det_path_cocoformat = '/GPFS/data/yhu/code/CoDet/exp/multiagent_det/dla_multiagent_withwarp_GlobalCoord_Polygon_FeatMap_800_450/results.json'
    # run_polygon_eval(anno_path_cocoformat, det_path_cocoformat)

    # anno_path_cocoformat = '/GPFS/data/yhu/code/BEV_Det/CoDet/exp/multiagent_det/LocalCoord_repeat/gts_Global.json'
    # det_path_cocoformat = '/GPFS/data/yhu/code/BEV_Det/CoDet/exp/multiagent_det/LocalCoord_repeat/results_BEV_Polygon.json'
    # run_polygon_eval(anno_path_cocoformat, det_path_cocoformat)
    # get_splited_town(anno_path_cocoformat, det_path_cocoformat)

    # for town_id in [4,5,6]:
    #     anno_path_cocoformat = '/GPFS/data/yhu/code/BEV_Det/CoDet/exp/multiagent_det/LocalCoord_repeat/gts_BEV_{}.json'.format(town_id)
    #     det_path_cocoformat = '/GPFS/data/yhu/code/BEV_Det/CoDet/exp/multiagent_det/LocalCoord_repeat/results_BEV_{}.json'.format(town_id)
    #     run_polygon_eval(anno_path_cocoformat, det_path_cocoformat)

    # for town_id in [4,6]:
    #     print('####################### Town: {} ####################### '.format(town_id))
    #     data_dir = towns[town_id]
    #     anno_path_cocoformat = os.path.join(data_dir, 'multiagent_annotations/{}_val_instances_global_crop_woignoredbox.json'.format(40))
    #     det_path_cocoformat = '/GPFS/data/yhu/code/BEV_Det/CoDet/exp/multiagent_det/LocalCoord_repeat/results_BEV_Polygon_{}.json'.format(town_id)
    #     run_polygon_eval(anno_path_cocoformat, det_path_cocoformat)

    # anno_path_cocoformat = '/DB/public/uav_dataset/multiagent_annotations/40_val_instances_global_crop_woignoredbox.json'
    # det_path_cocoformat = '/GPFS/data/yhu/code/BEV_Det/CoDet/exp/multiagent_det/RealData_GlobalCoord_Early/results_Global.json'
    # run_polygon_eval(anno_path_cocoformat, det_path_cocoformat)

    # anno_path_cocoformat = '/GPFS/data/yhu/code/BEV_Det/CoDet/exp/multiagent_det/GlobalCoord_Inter_DADW_WeightedDepth_Revised_NoZsupervision/gts_Global.json'
    # det_path_cocoformat = '/GPFS/data/yhu/code/BEV_Det/CoDet/exp/multiagent_det/GlobalCoord_Inter_DADW_WeightedDepth_Revised_NoZsupervision/results_Global.json'
    # anno_path_cocoformat = '/GPFS/data/yhu/code/BEV_Det/CoDet/exp/multiagent_det/GlobalCoord_Inter_DADW_WeightedDepth_Revised/gts_Global.json'
    # det_path_cocoformat = '/GPFS/data/yhu/code/BEV_Det/CoDet/exp/multiagent_det/GlobalCoord_Inter_DADW_WeightedDepth_Revised/results_Global.json'
    # run_polygon_eval(anno_path_cocoformat, det_path_cocoformat)

    anno_path_cocoformat = '/GPFS/data/yhu/code/BEV_Det/CoDet/exp/multiagent_det/JointCoord_Inter_DADW_WeightedDepth/gts_Global.json'
    det_path_cocoformat = '/GPFS/data/yhu/code/BEV_Det/CoDet/exp/multiagent_det/JointCoord_Inter_DADW_WeightedDepth/results_Global.json'
    # split_accord_dis(anno_path_cocoformat, det_path_cocoformat)

    # anno_path_cocoformat = '/GPFS/data/yhu/code/BEV_Det/CoDet/exp/multiagent_det/GlobalCoord_Inter_HW_WeightedDepth_RevisedZ/gts_Global.json'
    # det_path_cocoformat = '/GPFS/data/yhu/code/BEV_Det/CoDet/exp/multiagent_det/GlobalCoord_Inter_HW_WeightedDepth_RevisedZ/results_Global.json'
    # split_accord_dis(anno_path_cocoformat, det_path_cocoformat)

    # anno_path_cocoformat = '/GPFS/data/yhu/code/BEV_Det/CoDet/exp/multiagent_det/GlobalCoord_Inter_DADW_Revised_Finetune/gts_Global.json'
    # det_path_cocoformat = '/GPFS/data/yhu/code/BEV_Det/CoDet/exp/multiagent_det/GlobalCoord_Inter_DADW_Revised_Finetune/results_Global.json'
    # split_accord_dis(anno_path_cocoformat, det_path_cocoformat)

    # anno_path_cocoformat = '/GPFS/data/yhu/code/BEV_Det/CoDet/exp/multiagent_det/GlobalCoord_Inter_DADW_WeightedDepth_Revised/gts_Global.json'
    # det_path_cocoformat = '/GPFS/data/yhu/code/BEV_Det/CoDet/exp/multiagent_det/GlobalCoord_Inter_DADW_WeightedDepth_Revised/results_Global.json'
    # split_accord_dis(anno_path_cocoformat, det_path_cocoformat)
    run_polygon_eval(anno_path_cocoformat, det_path_cocoformat)
