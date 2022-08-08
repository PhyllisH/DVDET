from certifi import where
from matplotlib import markers
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import math

pose_error_std = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

# No_Message = [0.5784, 0.5529, 0.4658, 0.3358, 0.2385, 0.1699, 0.1182]
# V2V = [0.5982,0.5756, 0.4900, 0.3701, 0.2713, 0.2045, 0.1447]
# When2com = [0.6163, 0.6006, 0.5244, 0.4036, 0.3017, 0.2233, 0.1574]
# DiscoNet = [0.5974, 0.5807, 0.5120, 0.4005, 0.2986, 0.2247, 0.1583]
# where2comm = [0.6486, 0.6276, 0.5569, 0.4284, 0.3269, 0.2482, 0.1739]


No_Message = [0.5784, 0.5784, 0.5784, 0.5784, 0.5784, 0.5784, 0.5784]
V2V = [0.5982, 0.5942, 0.5885, 0.5789, 0.5641, 0.5487, 0.5368]
When2com = [0.6163, 0.6163, 0.6154, 0.6114, 0.6074, 0.6023, 0.5981]
DiscoNet = [0.5974, 0.5961, 0.5951, 0.5888, 0.5806, 0.5702, 0.5635]
where2comm = [0.6486, 0.6471, 0.6436, 0.6398, 0.6322, 0.6207, 0.6127]


fontsize = 20
label_size = 18
legend_size = 15
tick_size = 20
labelsize = 20
point_size = 100
figsize=(5.5,5.5)
params = {
    # 'legend.fontsize': 'x-large',
        # 'figure.figsize': (9, 7),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)


metric = 'AP@0.50'
fig = plt.figure()

plt.plot(pose_error_std, [x*100 for x in where2comm], label='Where2comm', linewidth=3, c='red')

plt.plot(pose_error_std, [x*100 for x in When2com], label='When2com', linewidth=3, c='violet')

plt.plot(pose_error_std, [x*100 for x in DiscoNet], label='DiscoNet', linewidth=3, c='olivedrab')

plt.plot(pose_error_std, [x*100 for x in V2V], label='V2VNet', linewidth=3, c='orange')

plt.plot(pose_error_std, [x*100 for x in No_Message], label='NoCollaboration', linewidth=3, c='mediumpurple', linestyle='--')

# plt.title('{}'.format(metric))
# plt.xlabel('AP')
plt.ylabel(metric, size=label_size)
plt.xlabel('Std of localization error (m)', size=label_size)
plt.legend(loc=3, prop={'size': legend_size})
# plt.ylim(0, 0.4)
plt.savefig('LocalizationError_UAV.png')



No_Message = [45.8, 45.8, 45.8, 45.8, 45.8, 45.8, 45.8]
V2V = [55.3, 54.6, 54.0, 52.1, 51.0, 50.7, 50.3]
When2com = [46.8, 46.8, 46.7, 46.4, 46.2, 46.1, 45.7]
DiscoNet = [58.0, 57.9, 57.7, 57.5, 57.3, 56.8, 56.0]
where2comm = [59.1, 58.8, 57.9, 57.7, 57.5, 57.3, 56.8]

fig = plt.figure()
plt.plot(pose_error_std, where2comm, label='Where2comm', linewidth=3, c='red')

plt.plot(pose_error_std, When2com, label='When2com', linewidth=3, c='violet')

plt.plot(pose_error_std, DiscoNet, label='DiscoNet', linewidth=3, c='olivedrab')

plt.plot(pose_error_std, V2V, label='V2VNet', linewidth=3, c='orange')

plt.plot(pose_error_std, No_Message, label='NoCollaboration', linewidth=3, c='mediumpurple', linestyle='--')
plt.ylabel(metric, size=label_size)
plt.xlabel('Std of localization error (m)', size=label_size)
plt.legend(loc=3, prop={'size': legend_size})
# plt.ylim(0, 0.4)
plt.savefig('LocalizationError_V2X.png')


No_Message = [22.66, 22.66, 22.66, 22.66, 22.66, 22.66, 22.66]
V2V = [37.47, 37.24, 35.87, 33.08, 30.52, 28.26, 25.79]
When2com = [19.53, 19.35, 18.78, 18.04, 17.07, 16.30, 15.26]
DiscoNet = [36.00, 35.69, 35.56, 33.16, 30.76, 29.49, 27.46]
where2comm = [47.30, 46.31, 45.45, 44.13, 41.83, 40.08, 38.30]
# where2comm = [47.30, 47.10, 46.37, 44.13, 43.02, 41.86, 40.32]

fig = plt.figure()

plt.plot(pose_error_std, where2comm, label='Where2comm', linewidth=3, c='red')

plt.plot(pose_error_std, When2com, label='When2com', linewidth=3, c='violet')

plt.plot(pose_error_std, DiscoNet, label='DiscoNet', linewidth=3, c='olivedrab')

plt.plot(pose_error_std, V2V, label='V2VNet', linewidth=3, c='orange')

plt.plot(pose_error_std, No_Message, label='NoCollaboration', linewidth=3, c='mediumpurple', linestyle='--')
plt.ylabel(metric, size=label_size)
plt.xlabel('Std of localization error (m)', size=label_size)
plt.legend(loc=3, prop={'size': legend_size})
# plt.ylim(0, 0.4)
plt.savefig('LocalizationError_OPV2V.png')

