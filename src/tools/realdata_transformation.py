import os
import torch
import kornia
import re
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import sys
import numpy as np
import re
from xml.dom.minidom import parse
from pyquaternion import Quaternion
import math
from scipy.spatial.transform import Rotation as R

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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


def euler2quaternion(yaw, pitch, roll):
    cy = np.cos(yaw * 0.5 * np.pi / 180.0)
    sy = np.sin(yaw * 0.5 * np.pi / 180.0)
    cp = np.cos(pitch * 0.5 * np.pi / 180.0)
    sp = np.sin(pitch * 0.5 * np.pi / 180.0)
    cr = np.cos(roll * 0.5 * np.pi / 180.0)
    sr = np.sin(roll * 0.5 * np.pi / 180.0)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return w, x, y, z


def get_origin_enu(xml_file):
    domTree = parse(xml_file)
    rootNode = domTree.documentElement

    SRS = rootNode.getElementsByTagName("SRS")[0].childNodes[0].data
    SRSOrigin = rootNode.getElementsByTagName("SRSOrigin")[0].childNodes[0].data

    data = re.split(r'[:|,]', SRS)
    lon, lat = float(data[2]), float(data[1])
    z = float(SRSOrigin.split(',')[-1])
    return lon, lat, z


def gps_to_decimal(lat_str):
    pattern = r'[°|′|″]'
    arg = re.split(pattern, lat_str)
    return float(arg[0]) + ((float(arg[1]) + (float(arg[2]) / 60)) / 60)


def gps_to_location(origin, long, lat):
    unit_lat = 6371000 * np.pi * 2 / 360
    unit_long = np.cos(np.deg2rad(origin[1])) * 6371000 * np.pi * 2 / 360
    x = unit_long * (long - origin[0])
    y = unit_lat * (lat - origin[1])
    return x, y


def get_imgcoord2worldgrid_matrices_dji(tranlation, rotation_matrix, camera_intrinsic, worldgrid2worldcoord_mat, z0=0):
    im_position = tranlation.copy()
    im_position = np.array(im_position).reshape((3, 1))
    im_position[2] = im_position[2] - z0

    extrinsic_mat = np.hstack((rotation_matrix, - rotation_matrix @ im_position))
    project_mat = camera_intrinsic @ extrinsic_mat
    project_mat = np.delete(project_mat, 2, 1)
    project_mat = project_mat @ worldgrid2worldcoord_mat
    project_mat = np.linalg.inv(project_mat)
    return project_mat

def get_imgcoord_matrices_dji(tranlation, rotation_matrix, camera_intrinsic, z0=0):
    im_position = tranlation.copy()
    im_position = np.array(im_position).reshape((3, 1))
    im_position[2] = im_position[2] - z0

    extrinsic_mat = np.hstack((rotation_matrix, - rotation_matrix @ im_position))
    project_mat = camera_intrinsic @ extrinsic_mat
    project_mat = np.delete(project_mat, 2, 1)
    return project_mat

def world2img(points, rotation_all, camera_intrinsic, ori_camera_position, offset):
    for j in range(3):
        points[j, :] = points[j, :] - ori_camera_position[j] - offset[j]
    points[:3, :] = np.dot(rotation_all, points[:3, :])

    # camera coord to perspective view
    depths = points[2, :]
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 0)
    points = points[:, mask]
    points = np.dot(camera_intrinsic, points)
    img_pixel = points / points[2:3, :].repeat(3, 0).reshape(3, points.shape[1])
    return img_pixel

def get_box_corners(item):
    """
    return boxes: (3, 8)
    """
    template = np.array(
                    [[1, 1, -1],
                    [-1, 1, -1],
                    [-1, -1, -1],
                    [1, -1, -1],
                    [1, 1, 1],
                    [-1, 1, 1],
                    [-1, -1, 1],
                    [1, -1, 1]]
                ) * 0.5
    
    position = np.array(item['position'])
    rotation = np.array(item['rotation'])
    dimension = np.array(item['dimension'])

    r = R.from_euler('YZX', rotation)
    corner = (template * dimension).dot(r.as_matrix()) + position
    return corner.T

def get_dji_pic_info(filename, img_csv_file, offset, origin_point):
    """

    :param filename:
    :param img_csv_file:
    :param offset:
    :param origin_point:
    :return: camera_position, camera_rotation_matrix
    """
    image_data = pd.read_csv(img_csv_file, header=0, encoding='gbk')
    img_info_line = image_data[image_data["文件名"] == filename]

    camera_lat = gps_to_decimal(img_info_line['GPS 纬度'].item())
    camera_lon = gps_to_decimal(img_info_line['GPS 经度'].item())
    camera_height = img_info_line['GPS 高度'].item()
    camera_height = float(camera_height[:-1])

    camera_location = gps_to_location(origin_point, camera_lon, camera_lat)
    camera_position = [camera_location[0], camera_location[1], float(camera_height - origin_point[2])]
    camera_position = np.array(camera_position)

    gimbal_yaw = img_info_line["相机云台偏航角"].item()
    gimbal_pitch = img_info_line["相机云台俯仰角"].item()
    gimbal_roll = img_info_line["相机云台横滚角"].item()

    rotation_1 = Quaternion(euler2quaternion(- (gimbal_roll + offset[5]), - (gimbal_yaw + offset[4]), 90))
    rotation_2 = Quaternion(euler2quaternion(0, 0, - (gimbal_pitch + offset[3])))
    rotation_all = rotation_2.rotation_matrix @ rotation_1.rotation_matrix

    return camera_position, rotation_all


def re_range(yaw):
    if yaw < math.pi/2.0 and yaw > -(math.pi)/2.0:
        return yaw
    elif yaw > math.pi/2.0:
        return yaw - math.pi
    elif yaw < -(math.pi)/2.0:
        return yaw + math.pi

def get_crop_shift_mat(tranlation, rotation, map_scale_w=1, map_scale_h=1, world_X_left=200, world_Y_left=200):
    im_position = tranlation.copy()
    im_position[-1] = 1
    world_mat = np.array([[1/map_scale_w, 0, 0], [0, 1/map_scale_h, 0], [0, 0, 1]]) @ \
                    np.array([[1, 0, world_X_left], [0, 1, world_Y_left], [0, 0, 1]])
    grid_center = world_mat @ im_position # [x, y, 1]
    # print(grid_center)
    yaw, _, _ = Quaternion(matrix=rotation).yaw_pitch_roll
    # print([int(x*180/math.pi) for x in Quaternion(matrix=rotation).yaw_pitch_roll])
    # yaw -= math.pi/4.0
    # yaw = re_range(yaw)
    # yaw -= math.pi/4.0
    yaw = -yaw + math.pi

    x_shift = 65/map_scale_w # 250/map_scale_w
    y_shift = 94/map_scale_h # 350/map_scale_h

    shift_mat = np.array([[1, 0, -x_shift], [0, 1, -y_shift], [0, 0, 1]])
    rotat_mat = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]) + \
                    np.array([[0, 0, grid_center[0]], [0, 0, grid_center[1]], [0, 0, 0]])
    trans_mat = np.linalg.inv(rotat_mat @ shift_mat)
    return trans_mat, int(Quaternion(matrix=rotation).yaw_pitch_roll[0]*180/math.pi)


if __name__ == '__main__':
    root_dir = '/DB/public/uav_dataset'
    camera_intrinsic = np.array([[486.023, 0, 359.066],
                                 [0, 486.105, 240.959],
                                 [0, 0, 1]])

    scene_no = 5
    offset = scene_offset[scene_no]
    scene_dir = os.path.join(root_dir, 'scene_{}_lidar'.format(scene_no))
    img_path = os.path.join(root_dir, 'image', '{}'.format(scene_no))
    xml_file = os.path.join(scene_dir, 'metadata.xml')  # a xml file includes the origin point coordinate in ENU
    img_info_file = os.path.join(root_dir, 'image', '{}.csv'.format(scene_no))  # a csv file

    scene_origin = get_origin_enu(xml_file)  # scene5 (121.42091098612988, 31.02890249679692, 45.19054702894702)

    images = os.listdir(img_path)
    images = [x for x in images if x.endswith('JPG')]
    # for img_name in images:
    #     print(img_name)
    #     # img_name = 'DJI_20211102111329_0007.JPG'
    #     camera_position, camera_rotation = get_dji_pic_info(img_name, img_info_file, offset, scene_origin)

    #     ground_height = -30.0  # car center height of the scene
    #     camera_position[2] = float(- ground_height)

    #     world_X_left=200
    #     world_Y_left=200
    #     scale = 1
    #     map_scale_h = 1/4 * scale
    #     map_scale_w = 1/4 * scale
    #     worldgrid2worldcoord_mat = np.array([[map_scale_w, 0, -world_X_left], [0, map_scale_h, -world_Y_left], [0, 0, 1]])
    #     # worldgrid2worldcoord_mat = np.array([[1, 0, -world_X_left], [0, 1, -world_Y_left], [0, 0, 1]])
    #     project_mat = get_imgcoord2worldgrid_matrices_dji(camera_position.copy(),
    #                                                     camera_rotation.copy(),
    #                                                     camera_intrinsic,
    #                                                     worldgrid2worldcoord_mat,
    #                                                     z0=0)

    #     img: np.ndarray = cv2.imread('{}/{}'.format(img_path, img_name))
    #     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     img = cv2.resize(img, (720, 480))
    #     # plt.imshow(img)
    #     # plt.show()
    #     data = kornia.image_to_tensor(img, keepdim=False)
    #     data_warp = kornia.warp_perspective(data.float(),
    #                                         torch.tensor(project_mat).repeat([1, 1, 1]).float(),
    #                                         dsize=(int(400/map_scale_h), int(400/map_scale_w)))

    #     # convert back to numpy
    #     img_warp = kornia.tensor_to_image(data_warp.byte())
    #     # print(img_warp.shape)
    #     img_warp = img_warp.transpose((1, 0, 2))
    #     plt.imshow(img_warp)
    #     plt.show()

    #     cur_shift_mat, yaw = get_crop_shift_mat(tranlation=camera_position.copy(), \
    #                                 rotation=camera_rotation.copy(), \
    #                                 map_scale_w=map_scale_w, map_scale_h=map_scale_h, \
    #                                 world_X_left=world_X_left, world_Y_left=world_Y_left)
    #     proj_mat = project_mat
    #     img_warp_center = kornia.image_to_tensor(img, keepdim=False)
    #     img_warp_center = kornia.warp_perspective(img_warp_center.float(),
    #                                         torch.tensor(cur_shift_mat@proj_mat).repeat([1, 1, 1]).float(),
    #                                         dsize=(int(96/(map_scale_h)), int(128/(map_scale_w))))
    #     img_warp_center = kornia.tensor_to_image(img_warp_center.byte())
    #     cv2.imwrite('realdata/{}_{}'.format(yaw,img_name), img_warp_center)
    #     import ipdb; ipdb.set_trace()
