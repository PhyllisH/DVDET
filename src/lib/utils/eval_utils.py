'''
Author: yhu
Contact: phyllis1sjtu@outlook.com
LastEditTime: 2021-07-23 19:39:36
Description: 
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os
from multiprocessing import Pool
from terminaltables import AsciiTable
from .post_process import compute_iou, get_corners
from shapely.geometry import Polygon


def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales).
    Args:
        recalls (ndarray): shape (num_scales, num_dets) or (num_dets, )
        precisions (ndarray): shape (num_scales, num_dets) or (num_dets, )
        mode (str): 'area' or 'ap', 'area' means calculating the area
            under precision-recall curve, 'ap' means calculating
            the average precision of recalls at [0.5, ..., 0.95]
    Returns:
        float or ndarray: calculated average precision
    """
    no_scale = False
    if recalls.ndim == 1:
        no_scale = True
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == 'ap':
        for i in range(num_scales):
            for thr in np.arange(0.5, 1, 0.05):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
            ap /= 10
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "ap" are supported')
    if no_scale:
        ap = ap[0]
    return ap


def tpfp_default(det_bboxes,
                 gt_bboxes,
                 iou_thr=0.5,
                 area_ranges=None):
    """Check if detected bboxes are true positive or false positive.
    Args:
        det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 4). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        area_ranges (list[tuple] | None): Range of bbox areas to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. Default: None.
    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
            each array is (num_scales, m).
    """
    num_dets = len(det_bboxes)
    num_gts = len(gt_bboxes)

    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)
    
    if len(det_bboxes) == 0:
        fp[...] = 1
        return tp, fp
    
    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if len(gt_bboxes) == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            if len(det_bboxes.shape) == 2:
                polygons = det_bboxes[:,:8].reshape(-1,4,2).transpose(0,2,1)
            elif len(det_bboxes.shape) == 1:
                polygons = det_bboxes[:8].reshape(-1,4,2).transpose(0,2,1)
            else:
                polygons = det_bboxes[0][0][:8].reshape(-1,4,2).transpose(0,2,1)
            h_w = np.sort(np.sqrt(np.square(polygons[:,:,1:] - polygons[:,:,0:1].repeat(3, axis=2)).sum(axis=1)), axis=-1)[:,:2]
            det_areas = h_w[:,0] * h_w[:,1]
            for i, (min_area, max_area) in enumerate(area_ranges):
                fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
        return tp, fp
    
    # an indicator of ignored gts
    gt_ignore_inds = np.zeros(gt_bboxes.shape[0], dtype=np.bool)

    gt_corners = np.zeros((gt_bboxes.shape[0], 4, 2), dtype=np.float32)
    pred_corners = np.zeros((det_bboxes.shape[0], 4, 2), dtype=np.float32)

    gt_corners = get_corners(gt_bboxes[:, :8].reshape(gt_bboxes.shape[0], 4, 2))
    pred_corners = get_corners(det_bboxes[:, :8].reshape(det_bboxes.shape[0], 4, 2))

    # print('gt: ', gt_corners[0])
    # print('pred: ', pred_corners[0])

    gt_box = np.array([Polygon([(box[i,0], box[i,1]) for i in range(4)]).buffer(0.01) for box in gt_corners])
    pred_box = np.array([Polygon([(box[i,0], box[i,1]) for i in range(4)]).buffer(0.01) for box in pred_corners])
    save_flag = False	
    # print('gt', gt_corners)
    # print('pred', pred_corners)
    for gt in gt_box:
        iou = np.array(compute_iou(gt, pred_box))
        if save_flag == False:
           box_iou = iou
           save_flag = True
        else:
           box_iou = np.vstack((box_iou, iou))

    ious = box_iou.T
    if len(ious.shape) < 2:
        ious = ious[:,np.newaxis]
    # ious = bbox_overlaps(det_bboxes, gt_bboxes)
    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)
    # for each det, which gt overlaps most with it
    ious_argmax = ious.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            polygons = gt_bboxes[:,:8].reshape(-1,4,2).transpose(0,2,1)
            h_w = np.sort(np.sqrt(np.square(polygons[:,:,1:] - polygons[:,:,0:1].repeat(3, axis=2)).sum(axis=1)), axis=-1)[:,:2]
            gt_areas = h_w[:,0] * h_w[:,1]
            gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not (gt_ignore_inds[matched_gt]
                        or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                    else:
                        fp[k, i] = 1
                # otherwise ignore this detected bbox, tp = 0, fp = 0
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :4]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    return tp, fp
    
def eval_map(det_results,
             annotations,
             iou_thr=0.5,
             mode='ap',
             nproc=4,
             print_flag=True):
    """Evaluate mAP of a dataset.
    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:
            - `bboxes`: numpy array of shape (n, 4*2)
            - `labels`: numpy array of shape (n, )
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.
    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    # print(len(det_results), len(annotations))
    assert len(det_results) == len(annotations)

    num_imgs = len(det_results)
    num_classes = len(det_results[0])  # positive class num
    # area_ranges = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
    area_ranges = [[0 ** 2, 1e5 ** 2]]
    # areaRngLbl = ['all', 'small', 'medium', 'large']
    num_scales = len(area_ranges) if area_ranges is not None else 1

    pool = Pool(nproc)
    eval_results = []
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets = [img_res[i] for img_res in det_results]
        cls_gts = [img_res[i] for img_res in annotations]
        # compute tp and fp for each image with multiple processes
        tpfp = pool.starmap(
            tpfp_default,
            zip(cls_dets, cls_gts,
                [iou_thr for _ in range(num_imgs)],
                [area_ranges for _ in range(num_imgs)]))
        tp, fp = tuple(zip(*tpfp))
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for j, bbox in enumerate(cls_gts):
            if len(bbox) == 0:
                continue
            if area_ranges is None:
                num_gts[0] += len(bbox)
            else:
                polygons = bbox[:,:8].reshape(-1,4,2).transpose(0,2,1)
                h_w = np.sort(np.sqrt(np.square(polygons[:,:,1:] - polygons[:,:,0:1].repeat(3, axis=2)).sum(axis=1)), axis=-1)[:,:2]
                gt_areas = h_w[:,0] * h_w[:,1]
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if area_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
    pool.close()
    if area_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0

    if print_flag:
        print_map_summary(mean_ap, eval_results, area_ranges)

    return mean_ap, eval_results

def print_map_summary(mean_ap,
                      results,
                      scale_ranges=None):
    """Print mAP and results of each class.
    A table will be printed to show the gts/dets/recall/AP of each class and
    the mAP.
    Args:
        mean_ap (float): Calculated from `eval_map()`.
        results (list[dict]): Calculated from `eval_map()`.
        dataset (list[str] | str | None): Dataset name or dataset classes.
        scale_ranges (list[tuple] | None): Range of scales to be evaluated.
    """
    if isinstance(results[0]['ap'], np.ndarray):
        num_scales = len(results[0]['ap'])
    else:
        num_scales = 1

    if scale_ranges is not None:
        assert len(scale_ranges) == num_scales

    num_classes = len(results)

    recalls = np.zeros((num_scales, num_classes), dtype=np.float32)
    aps = np.zeros((num_scales, num_classes), dtype=np.float32)
    num_gts = np.zeros((num_scales, num_classes), dtype=int)
    for i, cls_result in enumerate(results):
        if cls_result['recall'].size > 0:
            recalls[:, i] = np.array(cls_result['recall'], ndmin=2)[:, -1]
        aps[:, i] = cls_result['ap']
        num_gts[:, i] = cls_result['num_gts']

    label_names = [str(i) for i in range(num_classes)]

    if not isinstance(mean_ap, list):
        mean_ap = [mean_ap]

    header = ['class', 'gts', 'dets', 'recall', 'ap']
    for i in range(num_scales):
        if scale_ranges is not None:
            print(f'Scale range {scale_ranges[i]}')
        table_data = [header]
        for j in range(num_classes):
            row_data = [
                label_names[j], num_gts[i, j], results[j]['num_dets'],
                f'{recalls[i, j]:.3f}', f'{aps[i, j]:.3f}'
            ]
            table_data.append(row_data)
        table_data.append(['mAP', '', '', '', f'{mean_ap[i]:.3f}'])
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print('\n' + table.table)
