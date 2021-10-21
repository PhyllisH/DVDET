import pickle as pkl
import os
import numpy as np


data_dir = '/GPFS/data/yhu/Dataset/airsim_camera/airsim_camera_seg_15'
out_sample_path = os.path.join(data_dir, 'multiagent_annotations/{}_{}_instances_sample.pkl'.format(40, 'train'))

with open(out_sample_path, 'rb') as f:
    data = pkl.load(f)

vehicles_z = []
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