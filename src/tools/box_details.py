import pickle as pkl
import os
import numpy as np
import json

data_dirs = ['/GPFS/data/yhu/Dataset/airsim_camera/airsim_camera_seg_15',
                '/GPFS/data/shfang/dataset/airsim_camera/airsim_camera_seg_town6_v2',
                '/GPFS/data/shfang/dataset/airsim_camera/airsim_camera_seg_town4_v2_40m']
splits = ['train', 'val']

for split in splits:
    sample_amount = 0
    box_amount = 0
    for data_dir in data_dirs:
        out_sample_path = os.path.join(data_dir, 'multiagent_annotations/{}_{}_instances_global_crop_woignoredbox.json'.format(40, split))
        print(out_sample_path)
        with open(out_sample_path, 'r') as f:
            out_sample = json.load(f)
        print(data_dir, len(out_sample['images']), len(out_sample['annotations']))
        sample_amount += len(out_sample['images'])
        box_amount += len(out_sample['annotations'])
    print(sample_amount, box_amount)

import ipdb; ipdb.set_trace()

vehicles_z = []
for data_dir in data_dirs:
    out_sample_path = os.path.join(data_dir, 'multiagent_annotations/{}_{}_instances_sample.pkl'.format(40, split))

    with open(out_sample_path, 'rb') as f:
        data = pkl.load(f)
    print(len(data['samples']))
    for sample in data['samples']:
        for k in sample:
            if k.startswith('vehicle'):
                pass
            else:
                if len(sample[k]['vehicles_z']) > 1:
                    vehicles_z.append(sample[k]['vehicles_z'])

vehicles_z = np.concatenate(vehicles_z, axis=0)

mean_z = np.mean(vehicles_z, axis=1)
mean_z.sort()
min_z = vehicles_z[:,0]
min_z.sort()
max_z = vehicles_z[:,1]
max_z.sort()

split_5 = np.linspace(0, len(vehicles_z), num=6)[:-1]
split_5 = [int(x) for x in split_5]

print('Equal Frequency Split 5')
print([min_z[i] for i in split_5])
print([mean_z[i] for i in split_5])
print([max_z[i] for i in split_5])

print('Equal Gap Split 5')
print(np.histogram(min_z, bins=5))
print(np.histogram(mean_z, bins=5))
print(np.histogram(max_z, bins=5))