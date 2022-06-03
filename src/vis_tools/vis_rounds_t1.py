from tkinter import font
from certifi import where
from matplotlib import markers
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import math

communication_thres = [0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.9, 1.0]
communication_thres = [0.0, 0.001, 0.01, 0.03, 0.06, 0.08, 0.1, 0.13, 0.16, 0.20, 0.24, 0.28, 1.0]
# Model Local Transformer (Train thre 0.03)
init_communication_rates = [1, 0.102008, 0.010344, 0.002625, 0.000936, 0.000497, 0.000319, 0.000125, 0.000002, 0]
Where2comm = [
    [0.3692, 0.3669, 0.3589, 0.3555, 0.3507, 0.3442, 0.3399, 0.3381, 0.3376, 0.3376],
    [0.6486, 0.6438, 0.6355, 0.6316, 0.6250, 0.6150, 0.6077, 0.6037, 0.6023, 0.6023],
    [0.3848, 0.3826, 0.3711, 0.3665, 0.3587, 0.3517, 0.3477, 0.3464, 0.3464, 0.3464]
]

GaussianSmooth_communication_rates = [1, 0.105872, 0.012726, 0.004085, 0.002319, 0.001844, 0.001515, 0.000936,  0.000497, 0.000319, 0.000125, 0]
GaussianSmooth_Where2comm = [
    [0.3692, 0.3679, 0.3653, 0.3620, 0.3587, 0.3564, 0.3537, 0.3507, 0.3442, 0.3399, 0.3381, 0.3376],
    [0.6486, 0.6448, 0.6411, 0.6374, 0.6342, 0.6312, 0.6287, 0.6250, 0.6150, 0.6077, 0.6037, 0.6023],
    [0.3848, 0.3835, 0.3815, 0.3764, 0.3713, 0.3674, 0.3628, 0.3587, 0.3517, 0.3477, 0.3464, 0.3464]
]

# GaussianSmooth_communication_rates_r2 = [1+0.087227, 0.012303+0.031158, 0.003941+0.009548, 0.002268+0.004244, 0.001812+0.002883, 0.001495+0.002139, 0.001098+0.001112, 0.000762+0.000736, 0.000503+0.000449, 0.000320+0.000157, 0.000126+0.000057, 0]
GaussianSmooth_communication_rates_r2 = [1+0.090380, 0.012726+0.014729, 0.004085+0.007200, 0.002319+0.002324, 0.001844+0.001191, 0.001515+0.000692, 0.001113+0.000500, 0.000776+0.000247, 0.000497, 0.000319, 0.000125, 0]
GaussianSmooth_Where2comm_r2 = [
    [0.3751, 0.3734, 0.3719, 0.3686, 0.3640, 0.3596, 0.3559, 0.3517, 0.3474, 0.3432, 0.3400, 0.3396],
    [0.6560, 0.6521, 0.6496, 0.6462, 0.6420, 0.6380, 0.6326, 0.6271, 0.6195, 0.6121, 0.6035, 0.6022],
    [0.3916, 0.3897, 0.3872, 0.3823, 0.3742, 0.3682, 0.3641, 0.3595, 0.3549, 0.3500, 0.3473, 0.3471]
]

r1_rates = [1.0, 0.102075, 0.012303, 0.003941, 0.001812, 0.001495, 0.001098, 0.000761, 0.000503, 0.000320, 0.000126, 0.000054, 0, 0]
GaussianSmooth_communication_rates_r3 = [0.079944, 0.018155, 0.018185, 0.005372, 0.001057, 0.000638, 0.000464, 0.000350, 0.000232, 0.000129, 0.000031, 0.000003, 1e-5, 0]
GaussianSmooth_communication_rates_r3 = [r0 + r for r0, r in zip(r1_rates, GaussianSmooth_communication_rates_r3)]
GaussianSmooth_Where2comm_r3 = [
    [0.3762, 0.3753, 0.3746, 0.3731, 0.3698, 0.3658, 0.3609, 0.3569, 0.3532, 0.3493, 0.3443, 0.3402, 0.3401, 0.3401],
    [0.6571, 0.6546, 0.6532, 0.6510, 0.6479, 0.6436, 0.6394, 0.6341, 0.6304, 0.6228, 0.6130, 0.6023, 0.6019, 0.6019],
    [0.3938, 0.3927, 0.3925, 0.3898, 0.3855, 0.3786, 0.3716, 0.3661, 0.3610, 0.3573, 0.3529, 0.3493, 0.3494, 0.3494]
]


# GaussianSmooth_communication_rates_r2 = [1+0.090380, 0.105872+0.023455, 0.012726+0.014729, 0.004085+0.007200, 0.002319+0.002324, 0.001844+0.001191, 0.001515+0.000692, 0.001113+0.000500, 0.000776+0.000247, 0.000497, 0.000319, 0.000125, 0]
# GaussianSmooth_Where2comm_r2 = [
#     [0.3763, 0.3757, 0.3742, 0.3731, 0.3699, 0.3649, 0.3590, 0.3544, 0.3505, 0.3442, 0.3399, 0.3381, 0.3376],
#     [0.6661, 0.6647, 0.6621, 0.6599, 0.6570, 0.6524, 0.6475, 0.6419, 0.6365, 0.6150, 0.6077, 0.6037, 0.6023],
#     [0.3913, 0.3911, 0.3895, 0.3876, 0.3832, 0.3750, 0.3646, 0.3597, 0.3546, 0.3517, 0.3477, 0.3464, 0.3464]
# ]


# r1_rates = [1.0, 0.102075, 0.012303, 0.003941, 0.002268, 0.001812, 0.001495, 0.001098, 0.000761, 0.000503, 0.000320, 0.000126, 0.000054, 0]
# GaussianSmooth_communication_rates_r3 = [0.079944, 0.018155, 0.018185, 0.005372, 0.001937, 0.001057, 0.000638, 0.000464, 0.000350, 0.000232, 0.000129, 0.000031, 0.000003, 0]
# GaussianSmooth_communication_rates_r3 = [r0 + r for r0, r in zip(r1_rates, GaussianSmooth_communication_rates_r3)]
# GaussianSmooth_Where2comm_r3 = [
#     [0.3762, 0.3753, 0.3746, 0.3731, 0.3706, 0.3698, 0.3658, 0.3609, 0.3569, 0.3532, 0.3493, 0.3443, 0.3402, 0.3401],
#     [0.6571, 0.6546, 0.6532, 0.6510, 0.6535, 0.6479, 0.6436, 0.6394, 0.6341, 0.6304, 0.6228, 0.6130, 0.6023, 0.6019],
#     [0.3938, 0.3927, 0.3925, 0.3898, 0.3844, 0.3855, 0.3786, 0.3716, 0.3661, 0.3610, 0.3573, 0.3529, 0.3493, 0.3494]
# ]

No_Message = [0.3057, 0.5767, 0.2952]
# Late_Fusion = [0.2843, 0.5756, 0.3492]
V2V = [0.3292, 0.5982, 0.3314]
When2com = [0.3364, 0.6163, 0.3355]
DiscoNet = [0.3113, 0.5974, 0.2971]
LateFusion = [0.3057, 0.5767, 0.2952]
# where2comm = [0.3738, 0.6539, 0.3853]
# where2comm = [0.3692, 0.6483, 0.3849]
where2comm = [GaussianSmooth_Where2comm_r3[0][0], GaussianSmooth_Where2comm_r3[1][0], GaussianSmooth_Where2comm_r3[2][0]]


where2comm_ATTEN = [0.3466, 0.6348, 0.3454]
where2comm_MHA = [0.3598, 0.6487, 0.3654]
where2comm_MHA_PE = [0.3619, 0.6434, 0.3735]
where2comm_MHA_PE_CS = [0.3692, 0.6483, 0.3849]


communication_volume = 192*352*64*5*4
max_volume = math.log2(communication_volume * 32 // 8)

box_num = 100*5*4
Late_cost = math.log2(box_num * 7 * 32 // 8)

fontsize = 20
label_size = 18
legend_size = 17
tick_size = 20
labelsize = 20
point_size = 100
# figsize=(7,6)
# figsize=(6.7,5.5)
figsize=(5.5,5.5)
params = {
    # 'legend.fontsize': 'x-large',
        # 'figure.figsize': (9, 7),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)


# Figure1: Compare with SOTAs
metrics = ['AP', 'AP@0.50', 'AP@0.70']
plt.tick_params(labelsize=labelsize)
for i, metric in enumerate(metrics[:3]):
    # fig = plt.figure(figsize=(12,10))
    fig = plt.figure(figsize=figsize)
    plt.scatter(max_volume, where2comm[i]* 100, label='Where2comm', linewidth=3, c='red', s=point_size)
    plt.text(max_volume-7, where2comm[i]* 100-1, 'Where2comm'.format(max_volume,where2comm[i]* 100), ha='center', va='bottom', fontsize=fontsize)
    plt.scatter(0, No_Message[i]* 100, label='NoCollaboration', linewidth=3, c='mediumpurple', s=point_size)
    plt.text(9, No_Message[i]* 100, 'NoCollaboration'.format(0,No_Message[i]* 100), ha='center', va='bottom', fontsize=fontsize)
    plt.scatter(max_volume+math.log2(3), V2V[i]* 100, label='V2V', linewidth=3, c='orange', s=point_size)
    plt.text(max_volume+math.log2(3)-0.5, V2V[i]* 100+0.2, 'V2V'.format(max_volume+math.log2(3),V2V[i]* 100), ha='center', va='bottom', fontsize=fontsize)
    plt.scatter(max_volume, DiscoNet[i]* 100, label='DiscoNet', linewidth=3, c='olivedrab', s=point_size)
    plt.text(max_volume-3, DiscoNet[i]* 100-0.8, 'DiscoNet'.format(max_volume,DiscoNet[i]* 100), ha='center', va='bottom', fontsize=fontsize)
    plt.scatter(max_volume, When2com[i]* 100, label='When2com', linewidth=3, c='violet', s=point_size)
    plt.text(max_volume+math.log2(0.7)-3, When2com[i]* 100-0.8, 'When2com'.format(max_volume,When2com[i]* 100), ha='center', va='bottom', fontsize=fontsize)
    # plt.scatter(0, Where2comm[i][-1]* 100, label='Where2comm(NoBandLimit)', linewidth=3, c='steelblue')
    # plt.text(0, Where2comm[i][-1]* 100, '({:.02f},{:.02f})'.format(0,Where2comm[i][-1]* 100), ha='center', va='bottom', fontsize=fontsize)
    plt.scatter(Late_cost, LateFusion[i]* 100, label='LateFusion', linewidth=3, c='steelblue', s=point_size)
    plt.text(Late_cost, LateFusion[i]* 100, 'LateFusion'.format(max_volume,LateFusion[i]* 100), ha='center', va='bottom', fontsize=fontsize)
    
    
    # communication_rates, multi_agent_withthre_results = Thre003_communication_rates, Thre003_multi_agent_withthre_results
    communication_rates, multi_agent_withthre_results = GaussianSmooth_communication_rates_r3, GaussianSmooth_Where2comm_r3
    multi_agent_withthre_results = [x * 100 for x in multi_agent_withthre_results[i]]
    communication_volume_bytes = [math.log2(communication_volume * 32 // 8 * r) for r in communication_rates[:-1]] + [0]
    plt.plot(communication_volume_bytes[:-1], multi_agent_withthre_results[:-1], label='Where2comm', linewidth=3)
    
    communication_rates = [1/x for x in communication_rates[:-1]] + [0]

    # r = 0.000195
    # plt.plot([max_volume, math.log2(communication_volume * 32 // 8 * r)], [When2com[i]* 100, When2com[i]* 100], label='When2com', linewidth=2, c='black')
    # plt.arrow(max_volume, When2com[i]* 100, math.log2(communication_volume * 32 // 8 * r), When2com[i]* 100, shape='full', lw=0, length_includes_head=True, head_width=.05)
    # plt.text(math.log2(communication_volume * 32 // 8 * r), When2com[i]* 100-0.8, 'When2com'.format(max_volume,When2com[i]* 100), ha='center', va='bottom', fontsize=fontsize)
    
    # plt.title('{}'.format(metric))
    # plt.xlabel('AP')
    plt.ylabel(metric, size=label_size)
    plt.xlabel('CommunicationVolume(log2)', size=label_size)
    # plt.legend(loc=2,prop={'size': legend_size})
    # plt.ylim(0, 0.4)
    plt.savefig('SOTA_Performance_{}_vs_Commcost.png'.format(metric))

    
# Figure3: Gaussian Smooth
metrics = ['AP', 'AP@0.50', 'AP@0.70']
plt.tick_params(labelsize=labelsize)
for i, metric in enumerate(metrics[:3]):
    # fig = plt.figure(figsize=(7,7))
    # fig = plt.figure(figsize=(7,5.5))
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    communication_rates, multi_agent_withthre_results = GaussianSmooth_communication_rates, GaussianSmooth_Where2comm
    multi_agent_withthre_results = [x * 100 for x in multi_agent_withthre_results[i]]
    communication_volume_bytes = [math.log2(communication_volume * 32 // 8 * r) for r in communication_rates[:-1]] + [0]
    plt.plot(communication_volume_bytes[:-4], multi_agent_withthre_results[:-4], label='Gaussian', linewidth=3)
    communication_rates = [1/x for x in communication_rates[:-1]] + [0]

    
    communication_rates, multi_agent_withthre_results = init_communication_rates, Where2comm
    multi_agent_withthre_results = [x * 100 for x in multi_agent_withthre_results[i]]
    communication_volume_bytes = [math.log2(communication_volume * 32 // 8 * r) for r in communication_rates[:-1]] + [0]
    plt.plot(communication_volume_bytes[:-5], multi_agent_withthre_results[:-5], label='NoGaussian', linewidth=3, c='seagreen')
    communication_rates = [1/x for x in communication_rates[:-1]] + [0]

    
    # plt.title('{}'.format(metric))
    # plt.xlabel('AP')
    plt.ylabel(metric, size=label_size)
    plt.xlabel('CommunicationVolume(log2)', size=label_size)
    plt.legend(loc=4,prop={'size': legend_size})
    # plt.ylim(0, 0.4)
    ratio = 0.4
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    plt.savefig('GaussianSmooth_{}_vs_Commcost.png'.format(metric))


# Figure2: MultiRound Communication
metrics = ['AP', 'AP@0.50', 'AP@0.70']
plt.tick_params(labelsize=labelsize)
for i, metric in enumerate(metrics[:3]):
    # fig = plt.figure(figsize=(12,10))
    fig = plt.figure(figsize=figsize)
    
    communication_rates, multi_agent_withthre_results = GaussianSmooth_communication_rates_r3, GaussianSmooth_Where2comm_r3
    multi_agent_withthre_results = [x * 100 for x in multi_agent_withthre_results[i]]
    communication_volume_bytes = [math.log2(communication_volume * 32 // 8 * r) for r in communication_rates[:-1]] + [0]
    plt.plot(communication_volume_bytes[:-3], multi_agent_withthre_results[:-3], label='Round3', linewidth=3)
    communication_rates = [1/x for x in communication_rates[:-1]] + [0]

    # plt.scatter(communication_volume_bytes[0], multi_agent_withthre_results[0], label='Round3(NoBandLimit)', linewidth=3, s=point_size)
    # plt.text(communication_volume_bytes[0]-1, multi_agent_withthre_results[0], 'Round3'.format(communication_volume_bytes[0],multi_agent_withthre_results[0]), ha='center', va='bottom', fontsize=fontsize)
    

    communication_rates, multi_agent_withthre_results = GaussianSmooth_communication_rates_r2, GaussianSmooth_Where2comm_r2
    multi_agent_withthre_results = [x * 100 for x in multi_agent_withthre_results[i]]
    communication_volume_bytes = [math.log2(communication_volume * 32 // 8 * r) for r in communication_rates[:-1]] + [0]
    plt.plot(communication_volume_bytes[:-2], multi_agent_withthre_results[:-2], label='Round2', linewidth=3)
    communication_rates = [1/x for x in communication_rates[:-1]] + [0]

    # plt.scatter(communication_volume_bytes[0], multi_agent_withthre_results[0], label='Round2(NoBandLimit)', linewidth=3, s=point_size)
    # plt.text(communication_volume_bytes[0]-1, multi_agent_withthre_results[0]-0.6, 'Round2'.format(communication_volume_bytes[0],multi_agent_withthre_results[0]), ha='center', va='bottom', fontsize=fontsize)
    

    communication_rates, multi_agent_withthre_results = GaussianSmooth_communication_rates, GaussianSmooth_Where2comm
    multi_agent_withthre_results = [x * 100 for x in multi_agent_withthre_results[i]]
    communication_volume_bytes = [math.log2(communication_volume * 32 // 8 * r) for r in communication_rates[:-1]] + [0]
    plt.plot(communication_volume_bytes[:-2], multi_agent_withthre_results[:-2], label='Round1', linewidth=3, c='seagreen')
    communication_rates = [1/x for x in communication_rates[:-1]] + [0]
    
    # plt.scatter(communication_volume_bytes[0], multi_agent_withthre_results[0], label='Round1(NoBandLimit)', linewidth=3, c='seagreen', s=point_size)
    # plt.text(communication_volume_bytes[0]-1, multi_agent_withthre_results[0]-0.7, 'Round1'.format(communication_volume_bytes[0],multi_agent_withthre_results[0]), ha='center', va='bottom', fontsize=fontsize)

    # plt.title('{}'.format(metric))
    # plt.xlabel('AP')
    plt.ylabel(metric, size=label_size)
    plt.xlabel('CommunicationVolume(log2)', size=label_size)
    plt.legend(loc=4,prop={'size': legend_size})
    # plt.ylim(0, 0.4)
    plt.savefig('Multiround_{}_vs_Commcost.png'.format(metric))

