'''
Author: yhu
Contact: phyllis1sjtu@outlook.com
LastEditTime: 2021-06-22 18:40:15
Description: 
'''
import torch
import kornia
import os
import cv2
import numpy as np
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes


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


def test():
    data_dir = '/DATA5_DB8/data/public/airsim_camera/airsim_camera_10scene'
    camera_intrinsic = [[400.0, 0.0, 400.0],
                        [0.0, 400.0, 225.0],
                        [0.0, 0.0, 1.0]]
    camera_intrinsic = np.array(camera_intrinsic)
    reverse_matrix = np.eye(3)
    reverse_matrix[0, 0] = -1

    # get test data
    nusc = NuScenes(version='v1.0-mini', dataroot=data_dir, verbose=True)
    for cur_sample in nusc.sample:
        # cur_sample = nusc.sample[8]
        sample_data = cur_sample["data"]
        sensors = ['CAM_FRONT_id_0', 'CAM_LEFT_id_0', 'CAM_BOTTOM_id_0']

        prev_proj_mat = []
        prev_img = []
        prev_img_g = []
        for i, sensor in enumerate(sensors):
            sensor_record = nusc.get("sample_data", sample_data[sensor])
            # print(sensor_record)
            calibrated_record = nusc.get("calibrated_sensor", sensor_record["calibrated_sensor_token"])

            img: np.ndarray = cv2.imread(os.path.join(nusc.dataroot, sensor_record['filename']))
            print(img.shape)
            scale = 1/2
            map_scale_h = 500 / (img.shape[0] // scale)
            map_scale_w = 500 / (img.shape[1] // scale)
            img = cv2.resize(img, dsize=(int(img.shape[1]/scale), int(img.shape[0]/scale)))
            # img = cv2.resize(img, dsize=(img.shape[1], img.shape[0]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            worldgrid2worldcoord_mat = np.array([[map_scale_w, 0, -200], [0, map_scale_h, -250], [0, 0, 1]])

            project_mat = get_imgcoord2worldgrid_matrices(calibrated_record["translation"].copy(),
                                                        calibrated_record["rotation"].copy(),
                                                        camera_intrinsic,
                                                        worldgrid2worldcoord_mat)

            cv2.imwrite('ori.png', img)

            # map_zoom_mat = np.diag(np.append([1/map_scale_h, 1/map_scale_w], [1]))
            map_zoom_mat = np.diag(np.append([1, 1], [1]))
            img_zoom_mat = np.diag(np.append([scale, scale], [1]))
            # img_zoom_mat = np.diag(np.append([1, 1], [1]))

            data = kornia.image_to_tensor(img, keepdim=False) # [1, c, h, w ]
            data_warp = kornia.warp_perspective(data.float(),
                                                torch.tensor( map_zoom_mat @ project_mat @ img_zoom_mat).repeat([1, 1, 1]).float(),
                                                dsize=(int(500/map_scale_h), int(500/map_scale_w)))
            print(data_warp.nonzero().max(dim=0)[0])

            # convert back to numpy
            img_warp = kornia.tensor_to_image(data_warp.byte())

            cv2.imwrite('img_warp.png', img_warp)

            img_warp_back = kornia.image_to_tensor(img_warp, keepdim=False)
            img_warp_back = kornia.warp_perspective(img_warp_back.float(),
                                                torch.tensor(np.linalg.inv(map_zoom_mat @ project_mat @ img_zoom_mat)).repeat([1, 1, 1]).float(),
                                                dsize=(int(450/scale), int(800/scale)))
            # img_warp_back = kornia.warp_perspective(img_warp_back.float(),
            #                                     torch.tensor(np.linalg.inv(project_mat @ img_zoom_mat)).repeat([1, 1, 1]).float(),
            #                                     dsize=(450, 800))
            img_warp_back = kornia.tensor_to_image(img_warp_back.byte())

            cv2.imwrite('img_warp_back.png', img_warp_back)
            import ipdb; ipdb.set_trace()

            prev_proj_mat.append(project_mat)
            prev_img.append(img)
            prev_img_g.append(img_warp)

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

            #     img_warp_to_prev = kornia.image_to_tensor(prev_img_g[-1], keepdim=False)
            #     img_warp_to_prev = kornia.warp_perspective(img_warp_to_prev.float(),
            #                                         torch.tensor(np.linalg.inv(prev_proj_mat[-2])).repeat([1, 1, 1]).float(),
            #                                         dsize=(450, 800))
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
