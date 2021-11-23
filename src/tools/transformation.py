import numpy as np
import math
from pyquaternion import Quaternion


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
    # roll, pitch, yaw = quaternion2euler(rotation)
    yaw, pitch, roll = Quaternion(rotation).yaw_pitch_roll
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


def get_vehicle_coord(anno_data):
    translation = anno_data["translation"]
    size = anno_data["size"]
    a = size[0]
    size[0] = size[1]
    size[1] = a
    rotation = anno_data["rotation"]

    # cords the bounding box of a vehicle
    cords = np.zeros((8, 4))
    cords[0, :] = np.array([ size[0] / 2,  size[1] / 2, -size[2] / 2, 1])
    cords[1, :] = np.array([-size[0] / 2,  size[1] / 2, -size[2] / 2, 1])
    cords[2, :] = np.array([-size[0] / 2, -size[1] / 2, -size[2] / 2, 1])
    cords[3, :] = np.array([ size[0] / 2, -size[1] / 2, -size[2] / 2, 1])
    cords[4, :] = np.array([ size[0] / 2,  size[1] / 2,  size[2] / 2, 1])
    cords[5, :] = np.array([-size[0] / 2,  size[1] / 2,  size[2] / 2, 1])
    cords[6, :] = np.array([-size[0] / 2, -size[1] / 2,  size[2] / 2, 1])
    cords[7, :] = np.array([ size[0] / 2, -size[1] / 2,  size[2] / 2, 1])

    vehicle_world_matrix = _get_rotation_matrix(translation, rotation)

    world_cords = np.dot(vehicle_world_matrix, np.transpose(cords))
    return np.array(world_cords)

def get_2d_bounding_box(cords):
    """
    transform the 3D bounding box to 2D
    :param cords: <3, 8> the first channel: x, y, z; the second channel is the points amount
    :return <4, > 2D bounding box (x, y, w, h)
    """
    x_min = min(cords[0])
    x_max = max(cords[0])
    y_min = min(cords[1])
    y_max = max(cords[1])
    return x_min, y_min, x_max - x_min, y_max - y_min

def get_2d_polygon(cords):
    """
    transform the 3D bounding box to 2D
    :param cords: <3, 8> the first channel: x, y, z; the second channel is the points amount
    :return <4, > 2D bounding box (x, y, w, h)
    """
    x_sort = np.argsort(cords[0])
    lefts = cords[:2, x_sort[:2]]
    rights = cords[:2, x_sort[2:]]
    if abs(lefts[1,0] - lefts[1,1]) > 1e-6:
        lefts = lefts[:, np.argsort(lefts[1,:])]
    if abs(rights[1,0] - rights[1,1]) > 1e-6:
        rights = rights[:, np.argsort(rights[1,:])[::-1]]
    ordered_polygon = list(np.concatenate([lefts, rights], axis=-1).T.reshape(-1,))
    return ordered_polygon

def get_angle_polygon(polygon):
    """
    transform the 2D polygon to (x, y, w, h, cos, sin)
    :param polygon: <3, 8> the first channel: x, y, z; the second channel is the points amount
    :return <6, > 2D bounding box (x, y, w, h, cos, sin)
    """
    def _nor_theta(theta):
        if theta > math.radians(45):
            theta -= math.radians(90)
            theta = _nor_theta(theta)
        elif theta <= math.radians(-45):
            theta += math.radians(90)
            theta = _nor_theta(theta)
        return theta

    def _calc_bearing(point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        theta = math.atan2(y2 - y1, x2 - x1)
        theta = _nor_theta(theta)
        return theta
    
    def _get_corners(polygon):
        x_sort = np.argsort(polygon[0])
        lefts = polygon[:, x_sort[:2]]
        rights = polygon[:, x_sort[2:]]
        if abs(lefts[1,0] - lefts[1,1]) > 1e-6:
            lefts = lefts[:, np.argsort(lefts[1,:])]
        if abs(rights[1,0] - rights[1,1]) > 1e-6:
            rights = rights[:, np.argsort(rights[1,:])[::-1]]
        ordered_polygon = list(np.concatenate([lefts, rights], axis=-1).reshape(-1,))
        return ordered_polygon

    corners = np.array(_get_corners(polygon)).reshape(2,4).T    # (4,2)
    center = np.mean(np.array(corners), 0)
    theta = _calc_bearing(corners[0], corners[1])
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
    out_points = np.matmul(corners - center, rotation) + center
    x, y = list(out_points[0, :])
    w, h = [int(x) for x in list(out_points[2, :] - out_points[0, :])]

    # from matplotlib import pyplot as plt
    # from shapely.geometry.polygon import Polygon
    # from descartes import PolygonPatch
    # fig = plt.figure(1, figsize=(5,5), dpi=90)
    # corners = corners * 0.05
    # print(corners)
    # ring_mixed = Polygon([tuple(corners[0]), tuple(corners[1]), tuple(corners[2]), tuple(corners[3])])
    # ax = fig.add_subplot(111)
    # ring_patch = PolygonPatch(ring_mixed)
    # ax.add_patch(ring_patch)
    # ax.set_title('Filled Polygon')
    # xrange = [int(corners[:,0].min())-1, int(corners[:,0].max())+1]
    # yrange = [int(corners[:,1].min())-1, int(corners[:,1].max())+1]
    # ax.set_xlim(*xrange)
    # ax.set_xticks(list(range(*xrange)) + [xrange[-1]])
    # ax.set_ylim(*yrange)
    # ax.set_yticks(list(range(*yrange)) + [yrange[-1]])
    # ax.set_aspect(1)
    # plt.savefig('corner_polygon.png')
    # fig = plt.figure(1, figsize=(5,5), dpi=90)
    # out_points = out_points * 0.05
    # print(out_points)
    # ring_mixed = Polygon([tuple(out_points[0]), tuple(out_points[1]), tuple(out_points[2]), tuple(out_points[3])])
    # ax = fig.add_subplot(111)
    # ring_patch = PolygonPatch(ring_mixed)
    # ax.add_patch(ring_patch)
    # xrange = [int(out_points[:,0].min())-1, int(out_points[:,0].max())+1]
    # yrange = [int(out_points[:,1].min())-1, int(out_points[:,1].max())+1]
    # ax.set_xlim(*xrange)
    # ax.set_xticks(list(range(*xrange)) + [xrange[-1]])
    # ax.set_ylim(*yrange)
    # ax.set_yticks(list(range(*yrange)) + [yrange[-1]])
    # ax.set_aspect(1)
    # plt.savefig('trans_polygon.png')
    out_points = list(out_points.reshape(-1,))
    # return [x, y, w, h, np.sin(theta), np.cos(theta)], out_points
    return [x, y, w, h, np.sin(theta), np.cos(theta)], list(corners.reshape(-1,))

def global_points_to_image(global_points, translation, rotation, camera_intrinsic):
    """
    transform global points (x,y,z) to image (h,w,1)
    :param global_points: <3, n> points in global coordinate
    :param translation: translation of airsim camera
    :param rotation: rotation of airsim camera
    :param camera_intrinsic:
    :return: <3, n> image pixels
    """
    # in airsim coordinate system, +Z is down
    translation[2] = -translation[2]
    im_position = np.array(translation).reshape((3, 1))
    rotation[3] = - rotation[3]
    im_rotation = Quaternion(rotation)
    reverse_matrix = np.eye(3)
    reverse_matrix[0, 0] = -1

    points = global_points - im_position
    points = np.dot(im_rotation.rotation_matrix, points)
    points = np.dot(Quaternion([0.5, -0.5, 0.5, -0.5]).rotation_matrix.T, points)
    points = np.dot(reverse_matrix, points)

    depths = points[2, :]
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 0)
    points = points[:, mask]

    points = np.dot(camera_intrinsic, points)
    # normalize
    img_pixel = points / points[2:3, :].repeat(3, 0).reshape(3, points.shape[1])

    return img_pixel


def image_points_to_global(image_points, translation, rotation, camera_intrinsic, z0=0):
    """
    transform image (h,w,1) to global points (x,y,z)
    :param image_points: <3, n> image pixels, the third dimension is 1
    :param translation: translation of airsim camera
    :param rotation: rotation of airsim camera
    :param camera_intrinsic:
    :param z0: the default height of vehicles in global coordinates
    :return: <3, n> points in global coordinate
    """
    # in airsim coordinate system, +Z is down
    im_position = translation
    im_rotation = rotation
    im_position[2] = - im_position[2]
    im_position = np.array(im_position).reshape((3, 1))
    im_rotation[3] = - im_rotation[3]
    im_rotation = Quaternion(im_rotation)
    reverse_matrix = np.eye(3)
    reverse_matrix[0, 0] = -1

    mat = np.dot(im_rotation.rotation_matrix.T, Quaternion([0.5, -0.5, 0.5, -0.5]).rotation_matrix)
    mat = np.dot(mat, reverse_matrix)
    mat = np.dot(mat, np.linalg.inv(camera_intrinsic))

    points_ = np.dot(mat, image_points)
    d = (z0 - im_position[2]) / points_[2, :]
    restore_global = im_position + np.dot(mat, image_points) * d

    return restore_global

def get_imagecoord_from_worldcoord(world_coord, tranlation, rotation, camera_intrinsic, ignore_z=False):
    im_position = tranlation.copy()
    im_position[2] = - im_position[2]
    im_position = np.array(im_position).reshape((3, 1))
    im_rotation = rotation.copy()
    im_rotation[3] = - im_rotation[3]
    im_rotation = Quaternion(im_rotation)
    reverse_matrix = np.eye(3)
    reverse_matrix[0, 0] = -1

    mat = reverse_matrix @ Quaternion([0.5, -0.5, 0.5, -0.5]).rotation_matrix.T
    extrinsic_mat = np.hstack((im_rotation.rotation_matrix, - im_rotation.rotation_matrix @ im_position))

    if ignore_z:
        # Ignore z
        extrinsic_mat = np.delete(extrinsic_mat, 2, 1)
        project_mat = camera_intrinsic @ mat @ extrinsic_mat
        image_coord = project_mat @ np.concatenate([world_coord[:2], np.ones([1, world_coord.shape[-1]])], axis=0)
    else:
        # Consider z
        project_mat = camera_intrinsic @ mat @ extrinsic_mat
        image_coord = project_mat @ np.concatenate([world_coord, np.ones([1, world_coord.shape[-1]])], axis=0)

    image_coord = image_coord[:3] / image_coord[2, :]
    return image_coord


def get_worldcoord_from_imagecoord(image_coord, tranlation, rotation, camera_intrinsic, z0=0):
    im_position = tranlation.copy()
    im_position[2] = - im_position[2]
    im_position = np.array(im_position).reshape((3, 1))
    im_rotation = rotation.copy()
    im_rotation[3] = - im_rotation[3]
    im_rotation = Quaternion(im_rotation)
    reverse_matrix = np.eye(3)
    reverse_matrix[0, 0] = -1

    mat = reverse_matrix @ Quaternion([0.5, -0.5, 0.5, -0.5]).rotation_matrix.T

    if isinstance(z0, np.ndarray):
        extrinsic_mat = im_rotation.rotation_matrix
        project_mat = camera_intrinsic @ mat @ extrinsic_mat
        project_mat = np.linalg.inv(project_mat)
        world_coord = project_mat @ image_coord
        d = (z0 - im_position[2]) / world_coord[2, :]
        world_coord = im_position + world_coord * d
    else:
        extrinsic_mat = np.hstack((im_rotation.rotation_matrix, - im_rotation.rotation_matrix @ im_position))
        extrinsic_mat = np.delete(extrinsic_mat, 2, 1)
        project_mat = camera_intrinsic @ mat @ extrinsic_mat
        project_mat = np.linalg.inv(project_mat)
        image_coord = np.concatenate([image_coord[:2], np.ones([1, image_coord.shape[1]])], axis=0)
        world_coord = project_mat @ image_coord
        world_coord = world_coord / world_coord[2, :]
    return world_coord

def WorldCoord2WorldGrid(coord, scale_w=1, scale_h=1, world_X_left=200, world_Y_left=250):
    x = (coord[0:1] + world_X_left) * scale_w
    y = (coord[1:2] + world_Y_left) * scale_h
    return np.concatenate([x, y], axis=0)

def get_crop_shift_mat(tranlation, rotation, sensor_type, map_scale_w=1, map_scale_h=1, world_X_left=200, world_Y_left=250):
    im_position = tranlation.copy()
    im_position[-1] = 1
    world_mat = np.array([[1/map_scale_w, 0, 0], [0, 1/map_scale_h, 0], [0, 0, 1]]) @ \
                    np.array([[1, 0, world_X_left], [0, 1, world_Y_left], [0, 0, 1]])
    grid_center = world_mat @ im_position # [x, y, 1]

    yaw, _, _ = Quaternion(rotation).yaw_pitch_roll
    yaw += math.pi/2.0

    if sensor_type == 'BOTTOM':
        x_shift = 176/map_scale_w # 250/map_scale_w
        y_shift = 96/map_scale_h # 250/map_scale_h
    else:
        x_shift = 176/map_scale_w # 250/map_scale_w
        y_shift = 196/map_scale_h # 350/map_scale_h

    shift_mat = np.array([[1, 0, -x_shift], [0, 1, -y_shift], [0, 0, 1]])
    rotat_mat = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]) + \
                    np.array([[0, 0, grid_center[0]], [0, 0, grid_center[1]], [0, 0, 0]])
    trans_mat = np.linalg.inv(rotat_mat @ shift_mat)
    return trans_mat

def get_shift_coord(coord, project_mat):
    image_coord = project_mat @ np.concatenate([coord[:2], np.ones([1, coord.shape[-1]])], axis=0)
    image_coord = image_coord[:3] / image_coord[2, :]
    return image_coord[:2]


def get_imgcoord2worldgrid_matrices(tranlation, rotation, camera_intrinsic, worldgrid2worldcoord_mat):
    im_position = tranlation.copy()
    im_position[2] = - im_position[2]
    im_position = np.array(im_position).reshape((3, 1))
    im_rotation = rotation.copy()
    im_rotation[3] = - im_rotation[3]
    im_rotation = Quaternion(im_rotation)
    reverse_matrix = np.eye(3)
    reverse_matrix[0, 0] = -1

    mat = reverse_matrix @ Quaternion([0.5, -0.5, 0.5, -0.5]).rotation_matrix.T
    extrinsic_mat = np.hstack((im_rotation.rotation_matrix, - im_rotation.rotation_matrix @ im_position))
    extrinsic_mat = np.delete(extrinsic_mat, 2, 1)
    project_mat = camera_intrinsic @ mat @ extrinsic_mat
    project_mat = project_mat @ worldgrid2worldcoord_mat
    project_mat = np.linalg.inv(project_mat)
    return project_mat

def get_imgcoord_matrices(tranlation, rotation, camera_intrinsic, z0=0):
    im_position = tranlation.copy()
    im_position[2] = - im_position[2]
    im_position = np.array(im_position).reshape((3, 1))
    im_rotation = rotation.copy()
    im_rotation[3] = - im_rotation[3]
    im_rotation = Quaternion(im_rotation)
    reverse_matrix = np.eye(3)
    reverse_matrix[0, 0] = -1

    im_position[2] = im_position[2] - z0
    
    mat = reverse_matrix @ Quaternion([0.5, -0.5, 0.5, -0.5]).rotation_matrix.T
    extrinsic_mat = np.hstack((im_rotation.rotation_matrix, - im_rotation.rotation_matrix @ im_position))
    extrinsic_mat = np.delete(extrinsic_mat, 2, 1)
    project_mat = camera_intrinsic @ mat @ extrinsic_mat
    return project_mat