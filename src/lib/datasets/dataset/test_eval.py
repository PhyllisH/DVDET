'''
Author: yhu
Contact: phyllis1sjtu@outlook.com
LastEditTime: 2021-07-25 20:15:43
Description: 
'''

import json
import os
import re
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
from utils.eval_utils import eval_map

import cv2


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
                        annos[image_id][category_id] = np.array(annos[image_id][category_id]).reshape([1, len(annos[image_id][category_id])])
                    annotations[i].append(filter_polygon(annos[image_id][category_id]))
                else:
                    annotations[i].append([])
    return annotations

def run_polygon_eval(anno_path_cocoformat, det_path_cocoformat):
    with open(det_path_cocoformat, 'r') as f:
        det_results = json.load(f)

    with open(anno_path_cocoformat, 'r') as f:
        gt_annos = json.load(f)
    
    det_evalformat = todict(det_results)
    gt_evalformat = todict(gt_annos['annotations'])

    for image_id in det_evalformat:
        if image_id not in gt_evalformat:
            gt_evalformat[image_id] = {}

    det_evalformat = toevalformat(det_evalformat)
    gt_evalformat = toevalformat(gt_evalformat)

    print('####################### IOU 0.75 ####################### ')
    eval_map(det_evalformat, gt_evalformat, iou_thr=0.75, mode='area', nproc=8)

    print('####################### IOU 0.5 ####################### ')
    eval_map(det_evalformat, gt_evalformat, iou_thr=0.5, mode='area', nproc=8)
    # eval_map(det_evalformat, gt_evalformat, iou_thr=0.5, mode='ap', nproc=8)

    print('####################### IOU 0.25 ####################### ')
    eval_map(det_evalformat, gt_evalformat, iou_thr=0.25, mode='area', nproc=8)
    # eval_map(det_evalformat, gt_evalformat, iou_thr=0.25, mode='ap', nproc=8)

if __name__ == '__main__':
    anno_path_cocoformat = '/DATA7_DB7/data/shfang/airsim_camera_seg_15/multiagent_annotations/val_instances_global.json'
    det_path_cocoformat = '/GPFS/data/yhu/code/CoDet/exp/multiagent_det/dla_multiagent_withwarp_GlobalCoord_Polygon_FeatMap_800_450/results.json'
    # run_polygon_eval(anno_path_cocoformat, det_path_cocoformat)