from IPython.core import pylabtools
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np


# Performance & Distance 
Joint_DADW_Weighted = [
    [0.0278, 0.1299, 0.2387, 0.3186, 0.3084],
    [0.1051, 0.3430, 0.5019, 0.5964, 0.5315],
    [0.0064, 0.0740, 0.2026, 0.3152, 0.3215]
]

DADW_Weighted = [
    [0.0293, 0.1433, 0.2587, 0.3543, 0.3603],
    [0.1167, 0.3856, 0.5498, 0.6588, 0.6182],
    [0.0056, 0.0787, 0.2210, 0.3532, 0.3803]
]

# HW_Weighted = [
#     [0.0145, 0.0687, 0.1349, 0.2008, 0.1561],
#     [0.0553, 0.1993, 0.3149, 0.3980, 0.2872],
#     [0.0028, 0.0335, 0.0985, 0.1844, 0.1542]
# ]

# DADW = [
#     [0.0151, 0.0725, 0.1411, 0.1918, 0.1330],
#     [0.0553, 0.2012, 0.3150, 0.3673, 0.2420],
#     [0.0038, 0.0384, 0.1120, 0.1839, 0.1336]
# ]


distances = [200 - x for x in list(range(20, 220, 40))]


metrics = ['AP', 'AP@0.50', 'AP@0.75']
plt.tick_params(labelsize=20)
for i, metric in enumerate(metrics[:3]):
    # fig = plt.figure(figsize=(12,10))
    fig = plt.figure()
    plt.plot(distances, Joint_DADW_Weighted[i], label='Dual-View-BEV', linewidth=3)
    plt.plot(distances, DADW_Weighted[i], label='Single-View-BEV', linewidth=3)

    plt.title('{}'.format(metric))
    # plt.xlabel('AP')
    plt.ylabel(metric, size=14)
    plt.xlabel('Distance(m)', size=14)
    plt.legend()
    # plt.ylim(0, 0.4)
    plt.savefig('Joint_Performance_with_{}.png'.format(metric))

# metrics = ['AP', 'AP@0.50', 'AP@0.75']
# fig = plt.figure()
# for i, metric in enumerate(metrics[:3]):
#     plt.plot(distances, APs[i], label=metric)
#     # plt.title('{}'.format('Perception decay'))
#     plt.xlabel('Distance(m)')
#     plt.legend()
#     # plt.ylim(0.4, 1.0)
# plt.savefig('Performance_with_distance.png')

# metrics = ['AP', 'AP@0.50', 'AP@0.75']
# gain_dcns = []
# gain_caes = []
# for i, metric in enumerate(metrics[:3]):
#     gain_dcn = [x-y for x, y in zip(DADW_Weighted[i], HW_Weighted[i])]
#     gain_cae = [x-y for x, y in zip(DADW_Weighted[i], DADW[i])]
    
#     gain_dcns.append(gain_dcn)
#     gain_caes.append(gain_cae)


# metrics = ['AP', 'AP@0.50', 'AP@0.75']
# fig = plt.figure()
# for i, metric in enumerate(metrics[:3]):
#     # ratios = [x/y for x, y in zip(gain_dcns[i], HW_Weighted[i])]
#     # plt.plot(distances, ratios, label=metric)
#     plt.plot(distances, gain_dcns[i], label=metric)
#     # plt.title('{}'.format('Perception decay'))
#     plt.xlabel('AP')
#     plt.xlabel('Distance(m)')
#     plt.legend()
#     plt.ylim(0, 0.4)
# plt.savefig('Performance_with_dcn.png')


# metrics = ['AP', 'AP@0.50', 'AP@0.75']
# fig = plt.figure()
# for i, metric in enumerate(metrics[:3]):
#     # ratios = [x/y for x, y in zip(gain_caes[i], DADW[i])]
#     # plt.plot(distances, ratios, label=metric)
#     plt.plot(distances, gain_caes[i], label=metric)
#     # plt.title('{}'.format('Perception decay'))
#     plt.xlabel('AP')
#     plt.xlabel('Distance(m)')
#     plt.legend()
#     plt.ylim(0, 0.4)
# plt.savefig('Performance_with_cae.png')

# params = {
#     # 'legend.fontsize': 'x-large',
#         #   'figure.figsize': (15, 5),
#          'axes.labelsize': 'x-large',
#          'axes.titlesize':'x-large',
#          'xtick.labelsize':'x-large',
#          'ytick.labelsize':'x-large'}
# plt.rcParams.update(params)

# metrics = ['AP', 'AP@0.50', 'AP@0.75']
# plt.tick_params(labelsize=20)
# for i, metric in enumerate(metrics[:3]):
#     # fig = plt.figure(figsize=(12,10))
#     fig = plt.figure()
#     plt.plot(distances, DADW_Weighted[i], label='Inter-GDT-CAE', linewidth=3)
#     plt.plot(distances, HW_Weighted[i], label='Inter-GT-CAE', linewidth=3)
#     plt.plot(distances, DADW[i], label='Inter-GDT', linewidth=3)
#     plt.title('{}'.format(metric))
#     # plt.xlabel('AP')
#     plt.ylabel(metric, size=14)
#     plt.xlabel('Distance(m)', size=14)
#     plt.legend()
#     # plt.ylim(0, 0.4)
#     plt.savefig('Performance_with_{}.png'.format(metric))