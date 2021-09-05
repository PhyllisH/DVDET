'''
Author: yhu
Contact: phyllis1sjtu@outlook.com
LastEditTime: 2021-06-22 18:40:15
Description: 
'''
from kornia.geometry.transform.affwarp import scale
import torch
import kornia
import os
import cv2
import numpy as np
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
import math

from collections import OrderedDict
from transformation import get_vehicle_coord, WorldCoord2WorldGrid, get_shift_coord, get_crop_shift_mat


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
    # project_mat = mat @ extrinsic_mat
    # print(project_mat.shape)
    return project_mat

# def get_shift_mat(tranlation, rotation, d_type, map_scale_w=1, map_scale_h=1, world_shift_x=200, world_shift_y=250):
#     im_position = tranlation.copy()
#     im_position[-1] = 1
#     world_mat = np.array([[1/map_scale_w, 0, world_shift_x/map_scale_w], [0, 1/map_scale_h, world_shift_y/map_scale_h], [0, 0, 1]])
#     grid_center = world_mat @ im_position # [x, y, 1]
#     # print('center: ', updated_center)

#     yaw, _, _ = Quaternion(rotation).yaw_pitch_roll
#     yaw += math.pi/2.0
#     # rotat_mat = np.array([[math.cos(yaw), math.sin(-yaw), (1-math.cos(yaw))*250+250*math.sin(yaw)], [math.sin(yaw), math.cos(yaw), (1-math.cos(yaw))*250-250*math.sin(yaw)], [0, 0, 1]])
#     rotat_mat = np.array([[math.cos(yaw), math.sin(-yaw), (1-math.cos(yaw))*grid_center[0]+grid_center[1]*math.sin(yaw)], [math.sin(yaw), math.cos(yaw), (1-math.cos(yaw))*grid_center[1]-grid_center[0]*math.sin(yaw)], [0, 0, 1]])
#     # print('rot_mat: ', rotat_mat)

#     if d_type == 'BOTTOM':
#         x_shift = grid_center[0]-250/map_scale_w
#         y_shift = grid_center[1]-250/map_scale_h
#     else:
#         x_shift = grid_center[0]-250/map_scale_w
#         y_shift = grid_center[1]-350/map_scale_h
#         rotation_mat = Quaternion(rotation).rotation_matrix
#         if abs(rotation_mat[0,0] - 0.7) < 0.1 and abs(rotation_mat[1,0]) < 0.1:
#             d_type = 'RIGHT'
#         elif abs(rotation_mat[0,0]) < 0.1 and abs(rotation_mat[1,0]+0.7) < 0.1:
#             d_type = 'UP'
#         elif abs(rotation_mat[0,0]+0.7) < 0.1 and abs(rotation_mat[1,0]) < 0.1:
#             d_type = 'LEFT'
#         elif abs(rotation_mat[0,0]) < 0.1 and abs(rotation_mat[1,0]-0.7) < 0.1:
#             d_type = 'DOWN'
#     updated_shift = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]) @ np.array([x_shift, y_shift])
#     shift_mat = np.array([[1, 0, updated_shift[0]], [0, 1, updated_shift[1]], [0, 0, 1]])
#     trans_mat = rotat_mat
#     return trans_mat, d_type, shift_mat, yaw

def get_shift_mat(tranlation, rotation, d_type, map_scale_w=1, map_scale_h=1, world_shift_x=200, world_shift_y=250):
    im_position = tranlation.copy()
    im_position[-1] = 1
    world_mat = np.array([[1/map_scale_w, 0, 0], [0, 1/map_scale_h, 0], [0, 0, 1]]) @ \
                    np.array([[1, 0, world_shift_x], [0, 1, world_shift_y], [0, 0, 1]])
    grid_center = world_mat @ im_position # [x, y, 1]
    # print('center: ', updated_center)

    yaw, _, _ = Quaternion(rotation).yaw_pitch_roll
    yaw += math.pi/2.0

    if d_type == 'BOTTOM':
        x_shift = 176/map_scale_w # 250/map_scale_w
        y_shift = 96/map_scale_h # 250/map_scale_h
    else:
        x_shift = 176/map_scale_w # 250/map_scale_w
        y_shift = 196/map_scale_h # 350/map_scale_h
        rotation_mat = Quaternion(rotation).rotation_matrix
        if abs(rotation_mat[0,0] - 0.7) < 0.1 and abs(rotation_mat[1,0]) < 0.1:
            d_type = 'RIGHT'
        elif abs(rotation_mat[0,0]) < 0.1 and abs(rotation_mat[1,0]+0.7) < 0.1:
            d_type = 'UP'
        elif abs(rotation_mat[0,0]+0.7) < 0.1 and abs(rotation_mat[1,0]) < 0.1:
            d_type = 'LEFT'
        elif abs(rotation_mat[0,0]) < 0.1 and abs(rotation_mat[1,0]-0.7) < 0.1:
            d_type = 'DOWN'
    shift_mat = np.array([[1, 0, -x_shift], [0, 1, -y_shift], [0, 0, 1]])
    rotat_mat = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]) + \
                    np.array([[0, 0, grid_center[0]], [0, 0, grid_center[1]], [0, 0, 0]])
    
    # updated_shift = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]) @ np.array([(500-352)/(2*map_scale_w), (500-192)/(2*map_scale_h)])
    # shift_mat_2 = np.array([[1, 0, updated_shift[0]], [0, 1, updated_shift[1]], [0, 0, 1]])
    trans_mat = rotat_mat @ shift_mat
    return trans_mat, d_type, shift_mat, yaw

def test():
    data_dir = '/DATA5_DB8/data/public/airsim_camera/airsim_camera_10scene'
    camera_intrinsic = [[400.0, 0.0, 400.0],
                        [0.0, 400.0, 225.0],
                        [0.0, 0.0, 1.0]]
    camera_intrinsic = np.array(camera_intrinsic)
    reverse_matrix = np.eye(3)
    reverse_matrix[0, 0] = -1

    world_shift_x = 200
    world_shift_y = 250

    # get test data
    nusc = NuScenes(version='v1.0-mini', dataroot=data_dir, verbose=True)
    count = 0
    for cur_sample in nusc.sample:
        # cur_sample = nusc.sample[8]
        sample_data = cur_sample["data"]
        sensors = ['CAM_FRONT_id_0', 'CAM_LEFT_id_0', 'CAM_BOTTOM_id_0', 'CAM_BACK_id_0', 'CAM_RIGHT_id_0']
        sensors = []
        for cam in ['FRONT', 'LEFT', 'BOTTOM', 'BACK', 'RIGHT']:
            for i in range(5):
                sensors.append('CAM_{}_id_{}'.format(cam, i))

        prev_proj_mat = []
        prev_img = []
        prev_img_g = []
        count += 1
        
        for i, sensor in enumerate(sensors):
            d_type = sensor.split('_')[1]
            sensor_record = nusc.get("sample_data", sample_data[sensor])
            # print(sensor_record)
            calibrated_record = nusc.get("calibrated_sensor", sensor_record["calibrated_sensor_token"])

            img: np.ndarray = cv2.imread(os.path.join(nusc.dataroot, sensor_record['filename']))
            # print(img.shape)
            scale = 4
            # map_scale_h = 500 / (img.shape[0] // scale)
            # map_scale_w = 500 / (img.shape[1] // scale)
            map_scale_h = 1/4 * scale
            map_scale_w = 1/4 * scale
            img = cv2.resize(img, dsize=(int(img.shape[1]/scale), int(img.shape[0]/scale)))
            # img = cv2.resize(img, dsize=(img.shape[1], img.shape[0]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            worldgrid2worldcoord_mat = np.array([[map_scale_w, 0, -world_shift_x], [0, map_scale_h, -world_shift_y], [0, 0, 1]])

            project_mat = get_imgcoord2worldgrid_matrices(calibrated_record["translation"].copy(),
                                                        calibrated_record["rotation"].copy(),
                                                        camera_intrinsic,
                                                        worldgrid2worldcoord_mat)
            cv2.imwrite('ori.png', img)

            map_zoom_mat = np.diag(np.append([1, 1], [1]))
            img_zoom_mat = np.diag(np.append([scale, scale], [1]))

            data = kornia.image_to_tensor(img, keepdim=False) # [1, c, h, w ]
            data_warp = kornia.warp_perspective(data.float(),
                                                torch.tensor( map_zoom_mat @ project_mat @ img_zoom_mat).repeat([1, 1, 1]).float(),
                                                dsize=(int(500/map_scale_h), int(500/map_scale_w)))
            # print(data_warp.nonzero().max(dim=0)[0])
            img_warp = kornia.tensor_to_image(data_warp.byte())
            cv2.imwrite('img_warp.png', img_warp)

            # w_range = data_warp.max(dim=1)[0].squeeze(0).max(dim=0)[0].nonzero()
            # h_range = data_warp.max(dim=1)[0].squeeze(0).max(dim=1)[0].nonzero()
            # print('rotation: ', calibrated_record["rotation"].copy())
            # print('rotation: ', Quaternion(calibrated_record["rotation"].copy()).rotation_matrix)
            # print('rotation yaw_pitch_roll: ', Quaternion(calibrated_record["rotation"].copy()).yaw_pitch_roll)
            # print('Global: X {}-{}: {}, Y {}-{}: {}'.format(w_range[0].item(), w_range[-1].item(), w_range.shape[0], \
                                                            # h_range[0].item(), h_range[-1].item(), h_range.shape[0]))
            
            trans_mat, d_type, shift_mat, yaw = get_shift_mat(calibrated_record["translation"].copy(), calibrated_record["rotation"].copy(), d_type, map_scale_w, map_scale_h, world_shift_x, world_shift_y)
            proj_mat = map_zoom_mat @ project_mat @ img_zoom_mat
            reverse_mat = np.linalg.inv(trans_mat)
            reverse_mat_2 = np.linalg.inv(shift_mat @ trans_mat)
            # print(reverse_mat)
            img_warp_center = kornia.image_to_tensor(img, keepdim=False)
            img_warp_center = kornia.warp_perspective(img_warp_center.float(),
                                                torch.tensor(reverse_mat@proj_mat).repeat([1, 1, 1]).float(),
                                                dsize=(int(192/(map_scale_h)), int(352/(map_scale_w))))
            # img_warp_center = kornia.center_crop(img_warp_center, (200, 350))
            img_warp_center = kornia.tensor_to_image(img_warp_center.byte())
            if not os.path.exists('/GPFS/data/yhu/code/BEV_Det/CoDet/src/tools/warp/{}'.format(d_type)):
                os.makedirs('/GPFS/data/yhu/code/BEV_Det/CoDet/src/tools/warp/{}'.format(d_type))
            cv2.imwrite('warp/{}/img_warp_center_{}_{}.png'.format(d_type, sensor, count), img_warp_center)
            
            cv2.imwrite('warp/{}/img_warp_center_{}_{}_ori.png'.format(d_type, sensor, count), img_warp)
            
            cv2.imwrite('img_warp_center.png', img_warp_center)

            vehicle_cords = []
            for anno_token in cur_sample["anns"]:
                anno_data = nusc.get("sample_annotation", anno_token)
                vehicle_cord = get_vehicle_coord(anno_data)
                vehicle_cords.append(vehicle_cord)
            
            shift_mats = OrderedDict()
            for cur_scale in [1/2, 1, 2, 4, 8, 16]:
                cur_shift_mat = get_crop_shift_mat(tranlation=calibrated_record["translation"].copy(), \
                                                rotation=calibrated_record["rotation"].copy(), \
                                                sensor_type=sensor.split('_')[1], map_scale_w=cur_scale, map_scale_h=cur_scale, \
                                                world_X_left=world_shift_x, world_Y_left=world_shift_y)
                shift_mats[cur_scale] = cur_shift_mat

            img_warp_center = cv2.cvtColor(img_warp_center, cv2.COLOR_BGR2RGB)
            img_warp_center = cv2.cvtColor(img_warp_center, cv2.COLOR_RGB2BGR)

            for vehicle_cord in vehicle_cords:
                vehicle_grid = WorldCoord2WorldGrid(vehicle_cord[:3, :], scale_w=1, scale_h=1, world_X_left=world_shift_x, world_Y_left=world_shift_y)
                updated_vehicle_grid = get_shift_coord(vehicle_grid, shift_mats[1])
                ori_vehicle_grid = get_shift_coord(updated_vehicle_grid, np.linalg.inv(shift_mats[1]))
                import ipdb; ipdb.set_trace()
                bbox_g = updated_vehicle_grid.T
                img_warp_center_bbox = cv2.polylines(img_warp_center, pts=np.int32([bbox_g.reshape(-1, 1, 2)]), isClosed=True, color=(0, 0, 255), thickness=1)
            cv2.imwrite('img_warp_center_bbox.png', img_warp_center_bbox)
            import ipdb; ipdb.set_trace()


            img_warp_center = kornia.image_to_tensor(img, keepdim=False)
            img_warp_center = kornia.warp_perspective(img_warp_center.float(),
                                                torch.tensor(reverse_mat_2@proj_mat).repeat([1, 1, 1]).float(),
                                                dsize=(int(192/(map_scale_h)), int(352/(map_scale_w))))
            # img_warp_center = kornia.center_crop(img_warp_center, (192/(map_scale_h), 352/(map_scale_w)))
            img_warp_center = kornia.tensor_to_image(img_warp_center.byte())
            cv2.imwrite('warp/{}/img_warp_center_{}_{}_{:.02f}_{:.02f}_{:.02f}.png'.format(d_type, sensor, count, yaw, shift_mat[0,-1], shift_mat[1,-1]), img_warp_center)
            cv2.imwrite('img_warp_center_shift.png', img_warp_center)

            img_warp_back = kornia.image_to_tensor(img_warp, keepdim=False)
            img_warp_back = kornia.warp_perspective(img_warp_back.float(),
                                                torch.tensor(np.linalg.inv(map_zoom_mat @ project_mat @ img_zoom_mat)).repeat([1, 1, 1]).float(),
                                                dsize=(int(450/scale), int(800/scale)))
            img_warp_back = kornia.tensor_to_image(img_warp_back.byte())
            cv2.imwrite('img_warp_back.png', img_warp_back)

            prev_proj_mat.append(project_mat)
            prev_img.append(img)
            prev_img_g.append(img_warp)

            import ipdb; ipdb.set_trace()

            
            # create the plot
            # if len(prev_proj_mat) > 1:
            #     fig, axs = plt.subplots(3, 2, figsize=(16, 10))

            #     for i in range(2):
            #         axs[i][0].axis('off')
            #         axs[i][0].set_title('image source')
            #         axs[i][0].imshow(prev_img[-i-1])

            #         axs[i][1].axis('off')
            #         axs[i][1].set_title('image destination')
            #         axs[i][1].imshow(prev_img_g[-i-1])

            #     reverse_mat = np.linalg.inv(prev_proj_mat[-2])
            #     # reverse_mat = prev_cam_external_mat[-2] @ worldgrid2worldcoord_mat
            #     img_warp_to_prev = kornia.image_to_tensor(prev_img_g[-1], keepdim=False)
            #     img_warp_to_prev = kornia.warp_perspective(img_warp_to_prev.float(),
            #                                         torch.tensor(reverse_mat).repeat([1, 1, 1]).float(),
            #                                         dsize=(450, 800))
            #     # reverse_mat = np.linalg.inv(map_zoom_mat @ prev_proj_mat[-2] @ img_zoom_mat)
            #     # proj_mat = map_zoom_mat @ prev_proj_mat[-1] @ img_zoom_mat
            #     # img_warp_to_prev = kornia.image_to_tensor(prev_img[-1], keepdim=False)
            #     # img_warp_to_prev = kornia.warp_perspective(img_warp_to_prev.float(),
            #     #                                         torch.tensor(reverse_mat@proj_mat).repeat([1, 1, 1]).float(),
            #     #                                         dsize=(int(450/scale), int(800/scale)))
            #     img_warp_to_prev = kornia.tensor_to_image(img_warp_to_prev.byte())
                
            #     axs[2][0].axis('off')
            #     axs[2][0].set_title('source image in target coord')
            #     axs[2][0].imshow(img_warp_to_prev)
            # else:
            #     fig, axs = plt.subplots(1, 3, figsize=(16, 10))
            #     axs = axs.ravel()

            #     axs[0].axis('off')
            #     axs[0].set_title('image source')
            #     axs[0].imshow(img)

            #     axs[1].axis('off')
            #     axs[1].set_title('image destination')
            #     axs[1].imshow(img_warp)
                
            #     axs[2].axis('off')
            #     axs[2].set_title('image wrap back')
            #     axs[2].imshow(img_warp_back)
            # # plt.show()
            # plt.savefig('trans_{}.png'.format(sensor))


if __name__ == '__main__':
    test()
