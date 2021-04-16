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

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, transform_matrix


def visualize_image_with_bbox():
    coco_ = coco.COCO('C:\\Users\\35387\\Desktop\\airsim_camera_demo\\airsim_instances_train.json')
    dataset_dir = "C:\\Users\\35387\\Desktop\\airsim_camera_demo\\"
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


def visualize_result():
    coco_ = coco.COCO('C:\\Users\\35387\\Desktop\\airsim_camera_demo\\airsim_instances_train.json')
    dataset_dir = "C:\\Users\\35387\\Desktop\\airsim_camera_demo\\"
    catIds = coco_.getCatIds()
    imgIds = coco_.getImgIds()

    # resFile
    result_path = "C:\\Users\\35387\\Desktop\\results.json"
    res_annos_all = json.load(open(result_path))

    for img_id in range(1, 100):
        img = coco_.loadImgs(img_id)[0]
        annIds = coco_.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        annos = coco_.loadAnns(annIds)
        res_annos = [anno for anno in res_annos_all if anno['image_id'] == img_id]

        image = cv2.imread(os.path.join(dataset_dir, img['file_name']))

        for anno in annos:
            bbox = anno['bbox']
            x, y, w, h = bbox
            anno_image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 1)
            cv2.imshow('demo', anno_image)
        for anno in res_annos:
            if anno['score'] > 0.5:
                bbox = anno['bbox']
                x, y, w, h = bbox
                anno_image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 1)
                cv2.imshow('demo', anno_image)

        cv2.waitKey()


if __name__ == '__main__':
    visualize_result()
