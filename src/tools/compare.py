import cv2
import numpy as np
import os

# models = os.listdir('.')
# models = list(filter(lambda x: x.startswith('40'), models))
# modes = [x.split('_')[-1] for x in models]
save_dir = '/GPFS/data/yhu/code/BEV_Det/CoDet/exp/multiagent_det'
cams = ['FRONT', 'BACK', 'LEFT', 'RIGHT']

modes = ['NO_MESSAGE', 'V2V']
models = ['{}/vis'.format(mode) for mode in modes]

scenes = os.listdir(os.path.join(save_dir, models[0]))

save_path = os.path.join(save_dir, 'Comparision')
if not os.path.exists(save_path):
    os.mkdir(save_path)
padding = None
for scene in scenes:
    for cam in cams:
        for uav in range(5):
            cur_images = [os.path.join(save_dir, model, scene, '{}_{}_pred.png'.format(cam, uav)) for model in models]
            cur_images_g = [os.path.join(save_dir, model, scene, '{}_{}_pred_g.png'.format(cam, uav)) for model in models]
            
            if padding is None:
                padding = np.ones([20, np.array(cv2.imread(cur_images[0])).shape[1], 3], dtype=np.uint8) * 255
                padding_g = np.ones([20, np.array(cv2.imread(cur_images_g[0])).shape[1], 3], dtype=np.uint8) * 255

            cur_images = np.concatenate([np.concatenate([cv2.imread(x), padding], axis=0) for x in cur_images], axis=0)
            cur_images_g = np.concatenate([np.concatenate([cv2.imread(x), padding_g], axis=0) for x in cur_images_g], axis=0)

            cv2.imwrite(os.path.join(save_path, '{}_{}_{}.png'.format(scene, cam, uav)), cur_images)
            cv2.imwrite(os.path.join(save_path, '{}_{}_{}_g.png'.format(scene, cam, uav)), cur_images_g)

