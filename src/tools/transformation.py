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

