from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from operator import le

import numpy as np
import copy
from numpy.lib.arraysetops import isin
from shapely.geometry import Polygon
from .image import transform_preds
from .ddd_utils import ddd2locrot


def get_pred_depth(depth):
    return depth


def get_alpha(rot):
    # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # return rot[:, 0]
    idx = rot[:, 1] > rot[:, 5]
    alpha1 = np.arctan2(rot[:, 2], rot[:, 3]) + (-0.5 * np.pi)
    alpha2 = np.arctan2(rot[:, 6], rot[:, 7]) + (0.5 * np.pi)
    return alpha1 * idx + alpha2 * (1 - idx)


def ddd_post_process_2d(dets, c, s, opt):
    # dets: batch x max_dets x dim
    # return 1-based class det list
    ret = []
    include_wh = dets.shape[2] > 16
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, :2] = transform_preds(
            dets[i, :, 0:2], c[i], s[i], (opt.output_w, opt.output_h))
        classes = dets[i, :, -1]
        for j in range(opt.num_classes):
            inds = (classes == j)
            top_preds[j + 1] = np.concatenate([
                dets[i, inds, :3].astype(np.float32),
                get_alpha(dets[i, inds, 3:11])[:, np.newaxis].astype(np.float32),
                get_pred_depth(dets[i, inds, 11:12]).astype(np.float32),
                dets[i, inds, 12:15].astype(np.float32)], axis=1)
            if include_wh:
                top_preds[j + 1] = np.concatenate([
                    top_preds[j + 1],
                    transform_preds(
                        dets[i, inds, 15:17], c[i], s[i], (opt.output_w, opt.output_h))
                        .astype(np.float32)], axis=1)
        ret.append(top_preds)
    return ret


def ddd_post_process_3d(dets, calibs):
    # dets: batch x max_dets x dim
    # return 1-based class det list
    ret = []
    for i in range(len(dets)):
        preds = {}
        for cls_ind in dets[i].keys():
            preds[cls_ind] = []
            for j in range(len(dets[i][cls_ind])):
                center = dets[i][cls_ind][j][:2]
                score = dets[i][cls_ind][j][2]
                alpha = dets[i][cls_ind][j][3]
                depth = dets[i][cls_ind][j][4]
                dimensions = dets[i][cls_ind][j][5:8]
                wh = dets[i][cls_ind][j][8:10]
                locations, rotation_y = ddd2locrot(
                    center, alpha, dimensions, depth, calibs[0])
                bbox = [center[0] - wh[0] / 2, center[1] - wh[1] / 2,
                        center[0] + wh[0] / 2, center[1] + wh[1] / 2]
                pred = [alpha] + bbox + dimensions.tolist() + \
                       locations.tolist() + [rotation_y, score]
                preds[cls_ind].append(pred)
            preds[cls_ind] = np.array(preds[cls_ind], dtype=np.float32)
        ret.append(preds)
    return ret


def ddd_post_process(dets, c, s, calibs, opt):
    # dets: batch x max_dets x dim
    # return 1-based class det list
    dets = ddd_post_process_2d(dets, c, s, opt)
    dets = ddd_post_process_3d(dets, calibs)
    return dets


def ctdet_post_process(dets, c, s, h, w, num_classes):
    # dets: batch x max_dets x dim
    # return 1-based class det dict
    dims = dets.shape[-1]
    ret = []
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, :2] = transform_preds(
            dets[i, :, 0:2], c[i], s[i], (w, h))
        dets[i, :, 2:4] = transform_preds(
            dets[i, :, 2:4], c[i], s[i], (w, h))
        if dims > 6:
            dets[i, :, 4:6] = transform_preds(
                dets[i, :, 4:6], c[i], s[i], (w, h))
            dets[i, :, 6:8] = transform_preds(
                dets[i, :, 6:8], c[i], s[i], (w, h))
        classes = dets[i, :, -1]
        for j in range(num_classes):
            inds = (classes == j)
            top_preds[j + 1] = np.concatenate([
                dets[i, inds, :(dims-2)].astype(np.float32),
                dets[i, inds, (dims-2):(dims-1)].astype(np.float32)], axis=1).tolist()
        ret.append(top_preds)
    return ret


def multi_pose_post_process(dets, c, s, h, w):
    # dets: batch x max_dets x 40
    # return list of 39 in image coord
    ret = []
    for i in range(dets.shape[0]):
        bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h))
        pts = transform_preds(dets[i, :, 5:39].reshape(-1, 2), c[i], s[i], (w, h))
        top_preds = np.concatenate(
            [bbox.reshape(-1, 4), dets[i, :, 4:5],
             pts.reshape(-1, 34)], axis=1).astype(np.float32).tolist()
        ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
    return ret


###################### Polygon Detection #####################
def get_corners(polygon):
    def _get_corners(polygon):
        # print(polygon.shape)
        # print(polygon)
        x_sort = np.argsort(polygon[0])
        lefts = polygon[:, x_sort[:2]]
        rights = polygon[:, x_sort[2:][::-1]]
        if abs(lefts[1,0] - lefts[1,1]) > 1e-6:
            lefts = lefts[:, np.argsort(lefts[1,:])]
        if abs(rights[1,0] - rights[1,1]) > 1e-6:
            rights = rights[:, np.argsort(rights[1,:])[::-1]]
        ordered_polygon = np.concatenate([lefts, rights], axis=-1)
        # print(ordered_polygon)
        return ordered_polygon
    polygon = copy.deepcopy(polygon)
    polygon = [_get_corners(x.T).T[None,] for x in polygon]
    polygon = np.concatenate(polygon, axis=0)
    # left = np.argmin(polygon[:,:,0], axis=1)
    # right = np.argmax(polygon[:,:,0], axis=1)
    # up = np.argmin(polygon[:,:,1], axis=1)
    # bottom = np.argmax(polygon[:,:,1], axis=1)
    # n_index = list(range(len(polygon)))
    # polygon = np.concatenate([polygon[n_index,left,:].reshape(-1,1,2), polygon[n_index,bottom,:].reshape(-1,1,2),\
    #                                 polygon[n_index,right,:].reshape(-1,1,2), polygon[n_index,up,:].reshape(-1,1,2)], axis=1)
    return polygon

def convert_format(boxes_array):
    """
    :param array: an array of shape [# bboxs, 4, 2]
    :return: a shapely.geometry.Polygon object
    """

    polygons = [Polygon([(box[i*2], box[i*2+1]) for i in range(4)]) for box in boxes_array]
    return np.array(polygons)

def compute_iou(box, boxes):
    """Calculates IoU of the given box with the array of the given boxes.
    box: a polygon
    boxes: a vector of polygons
    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    iou = [box.intersection(b).area / (box.union(b).area+1e-6) for b in boxes]
    # iou = []
    # for b in boxes:
    #     try:
    #         a = box.intersection(b)
    #     except:
    #         a = 0.0
    #     if isinstance(a, float):
    #         iou.append(a)
    #     else:
    #         iou.append(a.area / (box.union(b).area+1e-6))
    #     print(iou)
    return np.array(iou, dtype=np.float32)

def polygon_nms(detections, threshold):
    """Performs non-maximum suppression and returns indices of kept boxes.
    detections: [N, 9]   (x,y)*4 and score
    threshold: Float. IoU threshold to use for filtering.
    return an numpy array of the positions of picks
    """
    assert detections.shape[0] > 0
    if detections.dtype.kind != "f":
        detections = detections.astype(np.float32)

    boxes = detections[:, :8]
    scores = detections[:, -1]

    polygons = convert_format(boxes)
    
    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    # print('ori: ',len(ixs))
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(polygons[i], polygons[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indices into ixs[1:], so add 1 to get
        # indices into ixs.
        
        remove_ixs = np.where(iou > threshold)[0] + 1
        
        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)

    # debug
    for i in range(len(pick)):
        iou = compute_iou(polygons[pick[i]],polygons[pick])
        # print(iou)
    print('selected: ', len(pick))

    detections = detections[pick]
    return
    