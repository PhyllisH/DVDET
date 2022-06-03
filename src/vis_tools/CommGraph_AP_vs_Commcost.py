from matplotlib import markers
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import math

communication_thres = [0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.9, 1.0]
communication_thres = [0.0, 0.001, 0.01, 0.03, 0.06, 0.08, 0.1, 0.13, 0.16, 0.20, 0.24, 0.28, 1.0]
# Model Local Transformer (Train thre 0.03)
Thre003_communication_rates = [1, 0.081173, 0.007459, 0.002180, 0.000855, 0.000477, 0.000002, 0]
Thre003_multi_agent_withthre_results = [
    [0.3591, 0.3616, 0.3621, 0.3608, 0.3545, 0.3453, 0.3371, 0.3371],
    [0.6447, 0.6464, 0.6440, 0.6436, 0.6384, 0.6254, 0.6046, 0.6046],
    [0.3701, 0.3726, 0.3732, 0.3709, 0.3617, 0.3500, 0.3454, 0.3454]
]

# Model Local Transformer (Train varing thre 0.0-1.0)
VaringThre_communication_rates = [1, 0.113694, 0.014132, 0.003331, 0.001488, 0.001155, 0.000981, 0.000830, 0.000735, 0.000646, 0.000579, 0.000523, 0.000499, 0.000397, 0.000312, 0.000232, 0.000002, 0]
VaringThre_multi_agent_withthre_results = [
    [0.3765, 0.3743, 0.3651, 0.3615, 0.3588, 0.3573, 0.3560, 0.3546, 0.3529, 0.3517, 0.3500, 0.3492, 0.3483, 0.3460, 0.3443, 0.3430, 0.3410, 0.3410],
    [0.6581, 0.6540, 0.6446, 0.6407, 0.6369, 0.6345, 0.6324, 0.6309, 0.6289, 0.6257, 0.6231, 0.6210, 0.6199, 0.6163, 0.6118, 0.6087, 0.6053, 0.6053],
    [0.3929, 0.3900, 0.3771, 0.3722, 0.3687, 0.3667, 0.3646, 0.3632, 0.3615, 0.3604, 0.3579, 0.3565, 0.3561, 0.3536, 0.3522, 0.3511, 0.3485, 0.3485]
]

BandwidthAware_VaringThre_communication_rates = [1, 0.113694, 0.014132, 0.003331, 0.000981, 0.000499, 0.000002, 0]
BandwidthAware_VaringThre_multi_agent_withthre_results = [
    [0.3763, 0.3756, 0.3738, 0.3643, 0.3506, 0.3458, 0.3410, 0.3410],
    [0.6579, 0.6557, 0.6520, 0.6419, 0.6237, 0.6151, 0.6053, 0.6053],
    [0.3926, 0.3928, 0.3904, 0.3772, 0.3584, 0.3541, 0.3485, 0.3485]
]

# K = 5; sigma=1
GaussianSmooth_VaringThre_communication_rates = [1, 0.118105, 0.016523, 0.004736, 0.002442, 0.001904,  0.001552, 0.001138, 0.000793, 0.000517, 0.000322, 0.000002, 0]
GaussianSmooth_VaringThre_multi_agent_withthre_results = [
    [0.3766, 0.3754, 0.3730, 0.3689, 0.3649, 0.3625, 0.3597, 0.3560, 0.3521, 0.3474, 0.3442, 0.3410, 0.3410],
    [0.6583, 0.6553, 0.6512, 0.6472, 0.6435, 0.6402, 0.6374, 0.6324, 0.6268, 0.6186, 0.6118, 0.6053, 0.6053],
    [0.3928, 0.3924, 0.3888, 0.3830, 0.3768, 0.3739, 0.3691, 0.3653, 0.3604, 0.3550, 0.3524, 0.3485, 0.3485]
]

# K = 5; sigma=0.5
GaussianSmooth_VaringThre_communication_rates = [1, 0.118105, 0.016523, 0.004736, 0.002442, 0.001904,  0.001552, 0.001138, 0.000793, 0.000344, 0.000251, 0.000002, 0]
GaussianSmooth_VaringThre_multi_agent_withthre_results = [
    [0.3766, 0.3754, 0.3730, 0.3689, 0.3649, 0.3625, 0.3597, 0.3560, 0.3521, 0.3449, 0.3432, 0.3410, 0.3410],
    [0.6583, 0.6553, 0.6512, 0.6472, 0.6435, 0.6402, 0.6374, 0.6324, 0.6268, 0.6132, 0.6095, 0.6053, 0.6053],
    [0.3928, 0.3924, 0.3888, 0.3830, 0.3768, 0.3739, 0.3691, 0.3653, 0.3604, 0.3530, 0.3514, 0.3485, 0.3485]
]
# K = 3; sigma=1
# GaussianSmooth_VaringThre_communication_rates = [1, 0.118105, 0.016523, 0.004736, 0.002442, 0.001904,  0.001552, 0.001138, 0.000793, 0.000450, 0.000282, 0.000002, 0]
# GaussianSmooth_VaringThre_multi_agent_withthre_results = [
#     [0.3766, 0.3754, 0.3730, 0.3689, 0.3649, 0.3625, 0.3597, 0.3560, 0.3521, 0.3463, 0.3435, 0.3410, 0.3410],
#     [0.6583, 0.6553, 0.6512, 0.6472, 0.6435, 0.6402, 0.6374, 0.6324, 0.6268, 0.6162, 0.6103, 0.6053, 0.6053],
#     [0.3928, 0.3924, 0.3888, 0.3830, 0.3768, 0.3739, 0.3691, 0.3653, 0.3604, 0.3544, 0.3519, 0.3485, 0.3485]
# ]

single_agent_results = [0.3057, 0.5767, 0.2952]
multi_agent_results = [0.3738, 0.6539, 0.3853]
V2V_results = [0.3292, 0.5982, 0.3314]
When2com_results = [0.3364, 0.6163, 0.3355]
DiscoNet_results = [0.3113, 0.5974, 0.2971]

communication_volume = 192*352*64*5*4
max_volume = math.log2(communication_volume * 32 // 8)

params = {
    # 'legend.fontsize': 'x-large',
        #   'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)



metrics = ['AP', 'AP@0.50', 'AP@0.75']
plt.tick_params(labelsize=20)
# for i, metric in enumerate(metrics[:3]):
#     # fig = plt.figure(figsize=(12,10))
#     fig = plt.figure()
#     plt.scatter(max_volume, multi_agent_results[i]* 100, label='Where2com', linewidth=3, c='steelblue')
#     plt.text(max_volume, multi_agent_results[i]* 100, '({:.02f},{:.02f})'.format(max_volume,multi_agent_results[i]* 100), ha='center', va='bottom', fontsize=10)
#     plt.scatter(0, single_agent_results[i]* 100, label='No Collaboration', linewidth=3, c='mediumpurple')
#     plt.text(0, single_agent_results[i]* 100, 'No Collaboration({:.02f},{:.02f})'.format(0,single_agent_results[i]* 100), ha='center', va='bottom', fontsize=10)
#     plt.scatter(max_volume+math.log2(3), V2V_results[i]* 100, label='V2V-3Round', linewidth=3, c='orange')
#     plt.text(max_volume+math.log2(3), V2V_results[i]* 100, 'V2V ({:.02f},{:.02f})'.format(max_volume+math.log2(3),V2V_results[i]* 100), ha='center', va='bottom', fontsize=10)
#     plt.scatter(max_volume, DiscoNet_results[i]* 100, label='DiscoNet', linewidth=3, c='olivedrab')
#     plt.text(max_volume, DiscoNet_results[i]* 100, 'DiscoNet ({:.02f},{:.02f})'.format(max_volume,DiscoNet_results[i]* 100), ha='center', va='bottom', fontsize=10)
#     plt.scatter(max_volume, When2com_results[i]* 100, label='When2com', linewidth=3, c='violet')
#     plt.text(max_volume+math.log2(0.7), When2com_results[i]* 100, 'When2com ({:.02f},{:.02f})'.format(max_volume+math.log2(0.7),When2com_results[i]* 100), ha='center', va='bottom', fontsize=10)
    
#     # communication_rates, multi_agent_withthre_results = Thre003_communication_rates, Thre003_multi_agent_withthre_results
#     communication_rates, multi_agent_withthre_results = VaringThre_communication_rates, VaringThre_multi_agent_withthre_results
#     multi_agent_withthre_results = [x * 100 for x in multi_agent_withthre_results[i]]
#     communication_volume_bytes = [math.log2(communication_volume * 32 // 8 * r) for r in communication_rates[:-1]] + [0]
#     plt.plot(communication_volume_bytes, multi_agent_withthre_results, label='Where2com(Threshold)', linewidth=3, c='seagreen')
#     communication_rates = [1/x for x in communication_rates[:-1]] + [0]
#     # for x, y, r, t in zip(communication_volume_bytes, multi_agent_withthre_results, communication_rates, communication_thres):
#     #     # print(x,y,r)
#     #     # plt.text(x, y, '({},{})'.format(int(r),y), ha='center', va='bottom', fontsize=10)
#     #     plt.text(x, y, '({:.02f},{:.02f})'.format(x,y), ha='center', va='bottom', fontsize=10)
    
#     # communication_rates, multi_agent_withthre_results = Thre003_communication_rates, Thre003_multi_agent_withthre_results
#     # communication_rates, multi_agent_withthre_results = BandwidthAware_VaringThre_communication_rates, BandwidthAware_VaringThre_multi_agent_withthre_results
#     # multi_agent_withthre_results = [x * 100 for x in multi_agent_withthre_results[i]]
#     # communication_volume_bytes = [math.log2(communication_volume * 32 // 8 * r) for r in communication_rates[:-1]] + [0]
#     # plt.plot(communication_volume_bytes, multi_agent_withthre_results, label='Where2com(BandwidthAware)', linewidth=3)
#     # communication_rates = [1/x for x in communication_rates[:-1]] + [0]
#     # for x, y, r, t in zip(communication_volume_bytes, multi_agent_withthre_results, communication_rates, communication_thres):
#     #     # print(x,y,r)
#     #     # plt.text(x, y, '({},{})'.format(int(r),y), ha='center', va='bottom', fontsize=10)
#     #     plt.text(x, y, '({:.02f},{:.02f})'.format(x,y), ha='center', va='bottom', fontsize=10)

#     communication_rates, multi_agent_withthre_results = GaussianSmooth_VaringThre_communication_rates, GaussianSmooth_VaringThre_multi_agent_withthre_results
#     multi_agent_withthre_results = [x * 100 for x in multi_agent_withthre_results[i]]
#     communication_volume_bytes = [math.log2(communication_volume * 32 // 8 * r) for r in communication_rates[:-1]] + [0]
#     plt.plot(communication_volume_bytes, multi_agent_withthre_results, label='Where2com(Gaussian)', linewidth=3)
#     communication_rates = [1/x for x in communication_rates[:-1]] + [0]
    
#     plt.title('{}'.format(metric))
#     # plt.xlabel('AP')
#     plt.ylabel(metric, size=14)
#     plt.xlabel('CommunicationVolume(log2)', size=14)
#     plt.legend()
#     # plt.ylim(0, 0.4)
#     plt.savefig('CommGraph_{}_vs_Commcost.png'.format(metric))



# rate = 0.012963
# context_results = [0.3740, 0.6523, 0.3906]
# for i, metric in enumerate(metrics[:3]):
#     # fig = plt.figure(figsize=(12,10))
#     fig = plt.figure()
    
#     communication_rates, multi_agent_withthre_results = VaringThre_communication_rates, VaringThre_multi_agent_withthre_results
#     multi_agent_withthre_results = [x * 100 for x in multi_agent_withthre_results[i]]
#     communication_volume_bytes = [math.log2(communication_volume * 32 // 8 * r) for r in communication_rates[:-1]] + [0]
#     plt.plot(communication_volume_bytes, multi_agent_withthre_results, marker='o', label='BandwidthLimited-Collaboration (threshold, AP)', linewidth=3)
#     communication_rates = [1/x for x in communication_rates[:-1]] + [0]
#     for x, y, r, t in zip(communication_volume_bytes, multi_agent_withthre_results, communication_rates, communication_thres):
#         # print(x,y,r)
#         # plt.text(x, y, '({},{})'.format(int(r),y), ha='center', va='bottom', fontsize=10)
#         plt.text(x, y, '({},{})'.format(t,y), ha='center', va='bottom', fontsize=10)
    
#     plt.scatter(math.log2(communication_volume * 32 // 8 * rate), context_results[i]* 100, label='Context', linewidth=3, c='steelblue')
    
#     plt.title('{}'.format(metric))
#     # plt.xlabel('AP')
#     plt.ylabel(metric, size=14)
#     plt.xlabel('CommunicationVolume(log2)', size=14)
#     plt.legend()
#     # plt.ylim(0, 0.4)
#     plt.savefig('CommGraph_{}_vs_Commcost.png'.format(metric))


################################ Multi-Round Performance ##################################

# Model Local Transformer (Train varing thre 0.0-1.0)
# Baseline_VaringThre_multiround_r1_rates = [1, 0.102008, 0.010344, 0.002625, 0.000936, 0.000497, 0.000002, 0]
# Baseline_VaringThre_multiround_r1_results = [
#     [0.3694, 0.3669, 0.3589, 0.3555, 0.3507, 0.3442, 0.3376, 0.3376],
#     [0.6486, 0.6438, 0.6355, 0.6316, 0.6250, 0.6150, 0.6023, 0.6023],
#     [0.3848, 0.3826, 0.3711, 0.3665, 0.3587, 0.3517, 0.3464, 0.3464]
# ]

Baseline_VaringThre_multiround_r1_rates = [1, 0.102008, 0.010344, 0.002625, 0.000936, 0.000497, 0.000319, 0.000125, 0.000002, 0]
Baseline_VaringThre_multiround_r1_results = [
    [0.3692, 0.3669, 0.3589, 0.3555, 0.3507, 0.3442, 0.3399, 0.3381, 0.3376, 0.3376],
    [0.6486, 0.6438, 0.6355, 0.6316, 0.6250, 0.6150, 0.6077, 0.6037, 0.6023, 0.6023],
    [0.3848, 0.3826, 0.3711, 0.3665, 0.3587, 0.3517, 0.3477, 0.3464, 0.3464, 0.3464]
]

# VaringThre_multiround_r2_rates = [1+0.086622, 0.102008+0.022217, 0.010344+0.015011, 0.002625+0.007526, 0.000936+0.003472, 0.000497+0.002145, 0.000002, 0]
# VaringThre_multiround_r2_results = [
#     [0.3766, 0.3736, 0.3703, 0.3688, 0.3662, 0.3614, 0.3344, 0.3344],
#     [0.6666, 0.6624, 0.6595, 0.6576, 0.6541, 0.6487, 0.5998, 0.5998],
#     [0.3910, 0.3892, 0.3824, 0.3800, 0.3776, 0.3689, 0.3412, 0.3412]
# ]

# GaussianSmooth
# communication_thres = [0.0, 0.001, 0.01, 0.03, 0.06, 0.08, 0.1, 0.13, 0.16, 0.20, 0.24, 0.28, 1.0]
# VaringThre_multiround_r1_rates = [1, 0.105872, 0.012726, 0.004085, 0.002319, 0.001844, 0.001515, 0.001113, 0.000776, 0.000509, 0.000319, 0.000125, 0]
# VaringThre_multiround_r1_results = [
#     [0.3692, 0.3679, 0.3653, 0.3620, 0.3587, 0.3564, 0.3537, 0.3507, 0.3472, 0.3433, 0.3399, 0.3381, 0.3376],
#     [0.6483, 0.6448, 0.6411, 0.6374, 0.6342, 0.6312, 0.6287, 0.6243, 0.6200, 0.6139, 0.6077, 0.6037, 0.6023],
#     [0.3849, 0.3835, 0.3815, 0.3764, 0.3713, 0.3674, 0.3628, 0.3587, 0.3548, 0.3504, 0.3477, 0.3464, 0.3464]
# ]

communication_thres = [0.0, 0.001, 0.01, 0.03, 0.06, 0.08, 0.1, 0.13, 0.16, 0.20, 0.24, 0.28, 1.0]
VaringThre_multiround_r1_rates = [1, 0.105872, 0.012726, 0.004085, 0.002319, 0.001844, 0.001515, 0.000936,  0.000497, 0.000319, 0.000125, 0]
VaringThre_multiround_r1_results = [
    [0.3692, 0.3679, 0.3653, 0.3620, 0.3587, 0.3564, 0.3537, 0.3507, 0.3442, 0.3399, 0.3381, 0.3376],
    [0.6483, 0.6448, 0.6411, 0.6374, 0.6342, 0.6312, 0.6287, 0.6250, 0.6150, 0.6077, 0.6037, 0.6023],
    [0.3849, 0.3835, 0.3815, 0.3764, 0.3713, 0.3674, 0.3628, 0.3587, 0.3517, 0.3477, 0.3464, 0.3464]
]

# round2: thre > 0.001
VaringThre_multiround_r2_rates = [1+0.090380, 0.105872+0.023455, 0.012726+0.014729, 0.004085+0.007200, 0.002319+0.004601, 0.001844+0.003967, 0.001515+0.003555, 0.001113+0.002999, 0.000776+0.002604, 0.000509+0.002184, 0.000319+0.001526, 0.000125+0.000508, 0]
VaringThre_multiround_r2_results = [
    [0.3763, 0.3757, 0.3742, 0.3731, 0.3719, 0.3705, 0.3691, 0.3659, 0.3617, 0.3571, 0.3491, 0.3382, 0.3344],
    [0.6661, 0.6647, 0.6621, 0.6599, 0.6588, 0.6566, 0.6562, 0.6540, 0.6497, 0.6458, 0.6338, 0.6095, 0.5998],
    [0.3913, 0.3911, 0.3895, 0.3876, 0.3853, 0.3843, 0.3813, 0.3765, 0.3682, 0.3623, 0.3520, 0.3439, 0.3412]
]

# round2: thre > thre at previous 2 step
VaringThre_multiround_r2_rates = [1+0.090380, 0.105872+0.023455, 0.012726+0.014729, 0.004085+0.007200, 0.002319+0.002324, 0.001844+0.001191, 0.001515+0.000692, 0.001113+0.000500, 0.000776+0.000247, 0.000509+0.000247, 0.000319+0.000140, 0.000125+0.000508, 0]
VaringThre_multiround_r2_results = [
    [0.3763, 0.3757, 0.3742, 0.3731, 0.3699, 0.3649, 0.3590, 0.3544, 0.3505, 0.3451, 0.3398, 0.3382, 0.3344],
    [0.6661, 0.6647, 0.6621, 0.6599, 0.6570, 0.6524, 0.6475, 0.6419, 0.6365, 0.6266, 0.6153, 0.6095, 0.5998],
    [0.3913, 0.3911, 0.3895, 0.3876, 0.3832, 0.3750, 0.3646, 0.3597, 0.3546, 0.3494, 0.3444, 0.3439, 0.3412]
]

VaringThre_multiround_r2_rates = [1+0.090380, 0.105872+0.023455, 0.012726+0.014729, 0.004085+0.007200, 0.002319+0.002324, 0.001844+0.001191, 0.001515+0.000692, 0.001113+0.000500, 0.000776+0.000247, 0.000509+0.000074, 0.000319+0.000140, 0.000125+0.000508, 0]
VaringThre_multiround_r2_results = [
    [0.3763, 0.3757, 0.3742, 0.3731, 0.3699, 0.3649, 0.3590, 0.3544, 0.3505, 0.3465, 0.3398, 0.3382, 0.3344],
    [0.6661, 0.6647, 0.6621, 0.6599, 0.6570, 0.6524, 0.6475, 0.6419, 0.6365, 0.6238, 0.6153, 0.6095, 0.5998],
    [0.3913, 0.3911, 0.3895, 0.3876, 0.3832, 0.3750, 0.3646, 0.3597, 0.3546, 0.3534, 0.3444, 0.3439, 0.3412]
]

VaringThre_multiround_r2_rates = [1+0.090380, 0.105872+0.023455, 0.012726+0.014729, 0.004085+0.007200, 0.002319+0.002324, 0.001844+0.001191, 0.001515+0.000692, 0.001113+0.000500, 0.000776+0.000247, 0.000497, 0.000319, 0.000125, 0]
VaringThre_multiround_r2_results = [
    [0.3763, 0.3757, 0.3742, 0.3731, 0.3699, 0.3649, 0.3590, 0.3544, 0.3505, 0.3442, 0.3399, 0.3381, 0.3376],
    [0.6661, 0.6647, 0.6621, 0.6599, 0.6570, 0.6524, 0.6475, 0.6419, 0.6365, 0.6150, 0.6077, 0.6037, 0.6023],
    [0.3913, 0.3911, 0.3895, 0.3876, 0.3832, 0.3750, 0.3646, 0.3597, 0.3546, 0.3517, 0.3477, 0.3464, 0.3464]
]

# Round3
# Gaussian
# GaussianSmooth_VaringThre_multiround_r3_rates = [0.092037, 0.021727, 0.019465, 0.006333, 0.002112, 0.001116, 0.000659, 0.000482, 0.000361, 0.000240, 0.000136, 0.000032, 0.000003, 0]
# GaussianSmooth_VaringThre_multiround_r3_results = [
#     [0.3758, 0.3746, 0.3748, 0.3734, 0.3706, 0.3656, 0.3592, 0.3550, 0.3515, 0.3465, 0.3411, 0.3367, 0.3360, 0.3360],
#     [0.6626, 0.6591, 0.6593, 0.6567, 0.6535, 0.6485, 0.6434, 0.6392, 0.6344, 0.6253, 0.6132, 0.6020, 0.6001, 0.5998],
#     [0.3906, 0.3903, 0.3900, 0.3893, 0.3844, 0.3776, 0.3663, 0.3612, 0.3566, 0.3513, 0.3462, 0.3446, 0.3436, 0.3439]
# ]

# GaussianSmooth_VaringThre_multiround_r3_rates = [0.087227, 0.121255, 0.031158, 0.009548, 0.004244, 0.002883, 0.002139, 0.001566, 0.001112, 0.000736, 0.000449, 0.000157, 0.000057, 0]
# GaussianSmooth_VaringThre_multiround_r3_results = [
#     [0.3751, 0.3740, 0.3734, 0.3719, 0.3686, 0.3640, 0.3596, 0.3559, 0.3517, 0.3474, 0.3432, 0.3400, 0.3396, 0.3393],
#     [0.6560, 0.6532, 0.6521, 0.6496, 0.6462, 0.6420, 0.6380, 0.6326, 0.6271, 0.6195, 0.6121, 0.6035, 0.6022, 0.6017],
#     [0.3916, 0.3903, 0.3897, 0.3872, 0.3823, 0.3742, 0.3682, 0.3641, 0.3595, 0.3549, 0.3500, 0.3473, 0.3471, 0.3466]
# ]

GaussianSmooth_VaringThre_multiround_r3_rates = [1.092106, 0.124043, 0.032255, 0.010361, 0.004394, 0.002932, 0.002159, 0.001583, 0.001125, 0.000746, 0.000459, 0.000160, 0.000058, 0]
GaussianSmooth_VaringThre_multiround_r3_results = [
    [0.3759, 0.3748, 0.3739, 0.3722, 0.3692, 0.3645, 0.3604, 0.3569, 0.3532, 0.3487, 0.3440, 0.3408, 0.3402, 0.3401],
    [0.6573, 0.6543, 0.6532, 0.6462, 0.6473, 0.6424, 0.6388, 0.6346, 0.6307, 0.6221, 0.6126, 0.6040, 0.6023, 0.6019],
    [0.3936, 0.3919, 0.3906, 0.3842, 0.3823, 0.3768, 0.3711, 0.3654, 0.3609, 0.3569, 0.3526, 0.3501, 0.3493, 0.3494]
]

GaussianSmooth_VaringThre_multiround_r2_rates = [0.079944, 0.018155, 0.018185, 0.005372, 0.001937, 0.001057, 0.000638, 0.000464, 0.000350, 0.000232, 0.000129, 0.000031, 0.000003, 0]
GaussianSmooth_VaringThre_multiround_r2_results = [
    [0.3762, 0.3753, 0.3746, 0.3731, 0.3706, 0.3698, 0.3658, 0.3609, 0.3569, 0.3532, 0.3493, 0.3443, 0.3402, 0.3401],
    [0.6571, 0.6546, 0.6532, 0.6510, 0.6535, 0.6479, 0.6436, 0.6394, 0.6341, 0.6304, 0.6228, 0.6130, 0.6023, 0.6019],
    [0.3938, 0.3927, 0.3925, 0.3898, 0.3844, 0.3855, 0.3786, 0.3716, 0.3661, 0.3610, 0.3573, 0.3529, 0.3493, 0.3494]
]

GaussianSmooth_VaringThre_multiround_r1_rates = [1.0, 0.102075, 0.012303, 0.003941, 0.002268, 0.001812, 0.001495, 0.001098, 0.000761, 0.000503, 0.000320, 0.000126, 0.000054, 0]
# GaussianSmooth_VaringThre_multiround_r1_results = [
#     [0.3685, 0.3675, 0.3655, 0.3616, 0.3584, 0.3558, 0.3537, 0.3508, 0.3479, 0.3443, 0.3416, 0.3399, 0.3396, 0.3393],
#     [0.6445, 0.6422, 0.6388, 0.6346, 0.6311, 0.6277, 0.6251, 0.6213, 0.6166, 0.6110, 0.6064, 0.6026, 0.6023, 0.6017],
#     [0.3846, 0.3847, 0.3810, 0.3753, 0.3698, 0.3663, 0.3638, 0.3610, 0.3575, 0.3525, 0.3493, 0.3476, 0.3473, 0.3466]
# ]

# No Gaussian
# VaringThre_multiround_r3_rates = [0.087240, 0.020460, 0.020495, 0.006508, 0.003986, 0.003370, 0.003005, 0.002696, 0.002507, 0.002315, 0.002149, 0.001975, 0.001895, 0]
# VaringThre_multiround_r3_results = [
#     [0.3760, 0.3715, 0.3731, 0.3687, 0.3679, 0.3673, 0.3671, 0.3670, 0.3665, 0.3661, 0.3652, 0.3634, 0.3628, 0.3360],
#     [0.6620, 0.6565, 0.6580, 0.6535, 0.6520, 0.6515, 0.6513, 0.6509, 0.6504, 0.6494, 0.6486, 0.6464, 0.6457, 0.5998],
#     [0.3915, 0.3861, 0.3887, 0.3806, 0.3795, 0.3784, 0.3779, 0.3774, 0.3769, 0.3768, 0.3753, 0.3734, 0.3723, 0.3439]
# ]

# VaringThre_multiround_r2_rates = [0.076924, 0.016758, 0.019016, 0.005283, 0.001453, 0.000713, 0.000460, 0.000382, 0.000330, 0.000273, 0.000227, 0.000183, 0.000120, 0.000003, 0]
# VaringThre_multiround_r2_results = [
#     [0.3761, 0.3730, 0.3721, 0.3674, 0.3613, 0.3582, 0.3557, 0.3548, 0.3532, 0.3516, 0.3507, 0.3494, 0.3474, 0.3402, 0.3401],
#     [0.6568, 0.6523, 0.6522, 0.6471, 0.6397, 0.6357, 0.6325, 0.6310, 0.6296, 0.6271, 0.6247, 0.6219, 0.6182, 0.6023, 0.6019],
#     [0.3936, 0.3890, 0.3879, 0.3806, 0.3725, 0.3680, 0.3650, 0.3634, 0.3605, 0.3590, 0.3579, 0.3581, 0.3563, 0.3493, 0.3494]
# ]

# VaringThre_multiround_r1_rates = [1.0, 0.097991, 0.009897, 0.002471, 0.001259, 0.001032, 0.000903, 0.000781, 0.000701, 0.000624, 0.000564, 0.000515, 0.000494, 0]
# VaringThre_multiround_r1_results = [
#     [0.3683, 0.3658, 0.3551, 0.3525, 0.3514, 0.3504, 0.3496, 0.3486, 0.3479, 0.3474, 0.3463, 0.3455, 0.3449, 0.3393],
#     [0.6444, 0.6405, 0.6270, 0.6346, 0.6224, 0.6210, 0.6251, 0.6180, 0.6194, 0.6167, 0.6145, 0.6127, 0.6118, 0.6017],
#     [0.3844, 0.3812, 0.3662, 0.3753, 0.3612, 0.3599, 0.3593, 0.3575, 0.3575, 0.3555, 0.3550, 0.3547, 0.3535, 0.3466]
# ]

for i, metric in enumerate(metrics[:3]):
    # fig = plt.figure(figsize=(12,10))
    fig = plt.figure()

    r2_results = [x * 100 for x in GaussianSmooth_VaringThre_multiround_r2_results[i]]
    r2_bytes = [math.log2(communication_volume * 32 // 8 * (r+r0)) for r, r0 in zip(GaussianSmooth_VaringThre_multiround_r2_rates[:-1], GaussianSmooth_VaringThre_multiround_r1_rates[:-1])] + [0]
    plt.plot(r2_bytes, r2_results, label='Round3 (Gaussian)', linewidth=3)

    # r3_results = [x * 100 for x in GaussianSmooth_VaringThre_multiround_r3_results[i]]
    # r3_bytes = [math.log2(communication_volume * 32 // 8 * r) for r in GaussianSmooth_VaringThre_multiround_r3_rates[:-1]] + [0]
    # plt.plot(r3_bytes, r3_results, label='Round3 (Gaussian)', linewidth=3)

    # r3_results = [x * 100 for x in VaringThre_multiround_r3_results[i]]
    # r3_bytes = [math.log2(communication_volume * 32 // 8 * (r+r0)) for r, r0 in zip(VaringThre_multiround_r3_rates[:-1], VaringThre_multiround_r1_rates[:-1])] + [0]
    # plt.plot(r3_bytes, r3_results, label='Round3', linewidth=3)

    # r2_results = [x * 100 for x in VaringThre_multiround_r2_results[i]]
    # r2_bytes = [math.log2(communication_volume * 32 // 8 * (r+r0)) for r, r0 in zip(VaringThre_multiround_r2_rates[:-1], VaringThre_multiround_r1_rates[:-1])] + [0]
    # plt.plot(r2_bytes, r2_results, label='Round2', linewidth=3)

    r2_results = [x * 100 for x in VaringThre_multiround_r2_results[i]]
    r2_bytes = [math.log2(communication_volume * 32 // 8 * r) for r in VaringThre_multiround_r2_rates[:-1]] + [0]
    plt.plot(r2_bytes, r2_results, label='Round2 (Gaussian)', linewidth=3)

    r1_results = [x * 100 for x in VaringThre_multiround_r1_results[i]]
    r1_bytes = [math.log2(communication_volume * 32 // 8 * r) for r in VaringThre_multiround_r1_rates[:-1]] + [0]
    plt.plot(r1_bytes, r1_results, label='Round1 (Gaussian)', linewidth=3)

    r1_results = [x * 100 for x in  Baseline_VaringThre_multiround_r1_results[i]]
    r1_bytes = [math.log2(communication_volume * 32 // 8 * r) for r in Baseline_VaringThre_multiround_r1_rates[:-1]] + [0]
    plt.plot(r1_bytes, r1_results, label='Round1 (Thre)', linewidth=3)

    # r3_results = [x * 100 for x in GaussianSmooth_VaringThre_multiround_r3_results[i]]
    # r3_bytes = [math.log2(communication_volume * 32 // 8 * r) for r in GaussianSmooth_VaringThre_multiround_r3_rates[:-1]] + [0]
    # plt.plot(r3_bytes, r3_results, label='Round3 (Gaussian)', linewidth=3)

    # r2_results = [x * 100 for x in GaussianSmooth_VaringThre_multiround_r2_results[i]]
    # r2_bytes = [math.log2(communication_volume * 32 // 8 * (r+r0)) for r, r0 in zip(GaussianSmooth_VaringThre_multiround_r2_rates[:-1], GaussianSmooth_VaringThre_multiround_r1_rates[:-1])] + [0]
    # plt.plot(r2_bytes, r2_results, label='Round2 (Gaussian)', linewidth=3)

    # r1_results = [x * 100 for x in GaussianSmooth_VaringThre_multiround_r1_results[i]]
    # r1_bytes = [math.log2(communication_volume * 32 // 8 * r) for r in GaussianSmooth_VaringThre_multiround_r1_rates[:-1]] + [0]
    # plt.plot(r1_bytes, r1_results, label='Round1 (Gaussian)', linewidth=3)

    plt.title('{}'.format(metric))
    # plt.xlabel('AP')
    plt.ylabel(metric, size=14)
    plt.xlabel('CommunicationVolume(log2)', size=14)
    plt.legend()
    # plt.ylim(0, 0.4)
    plt.savefig('CommGraph_{}_vs_Commcost.png'.format(metric))


# # Partial graph
# VaringThre_multiround_r1_rates = [x*0.8 for x in [1, 0.102008, 0.010344, 0.002625, 0.000936, 0.000497, 0.000002, 0]]
# VaringThre_multiround_r1_results = [
#     [0.3632, 0.3618, 0.3545, 0.3520, 0.3485, 0.3433, 0.3376, 0.3376],
#     [0.6393, 0.6373, 0.6290, 0.6259, 0.6216, 0.6135, 0.6023, 0.6023],
#     [0.3779, 0.3758, 0.3655, 0.3631, 0.3559, 0.3511, 0.3464, 0.3464]
# ]

# VaringThre_multiround_r2_rates = [x*0.8 for x in [1+0.086622, 0.102008+0.022217, 0.010344+0.015011, 0.002625+0.007526, 0.000936+0.003472, 0.000497+0.002145, 0.000002, 0]]
# VaringThre_multiround_r2_results = [
#     [0.3709, 0.3687, 0.3657, 0.3644, 0.3625, 0.3576, 0.3344, 0.3344],
#     [0.6582, 0.6547, 0.6512, 0.6502, 0.6477, 0.6415, 0.5998, 0.5998],
#     [0.3854, 0.3832, 0.3775, 0.3748, 0.3727, 0.3651, 0.3412, 0.5998]
# ]

# # R1 0.1; \beta=0; K=400, 270, 200
# top_K = [600, 400, 270, 200, 0]
# VaringThre_multiround_r2_topk_rates = [0.000936+0.008878, 0.000936+0.005919, 0.000936+0.003995, 0.000936+0.002959, 0.000936]
# VaringThre_multiround_r2_topk_results = [
#     [0.3646, 0.3626, 0.3595, 0.3571, 0.3507],
#     [0.6517, 0.6551, 0.6469, 0.6443, 0.6250],
#     [0.3736, 0.3793, 0.3667, 0.3639, 0.3587]
# ]

# for i, metric in enumerate(metrics[:3]):
#     # fig = plt.figure(figsize=(12,10))
#     fig = plt.figure()

#     r2_results = [x * 100 for x in VaringThre_multiround_r2_results[i]]
#     r2_bytes = [math.log2(communication_volume * 32 // 8 * r) for r in VaringThre_multiround_r2_rates[:-1]] + [0]
#     plt.plot(r2_bytes[:-3], r2_results[:-3], marker='o', label='Round2 (Gaussian)', linewidth=3)

#     r1_results = [x * 100 for x in VaringThre_multiround_r1_results[i]]
#     r1_bytes = [math.log2(communication_volume * 32 // 8 * r) for r in VaringThre_multiround_r1_rates[:-1]] + [0]
#     plt.plot(r1_bytes, r1_results, label='Round1 (Gaussian)', linewidth=3)

#     r1_results = [x * 100 for x in  Baseline_VaringThre_multiround_r1_results[i]]
#     r1_bytes = [math.log2(communication_volume * 32 // 8 * r) for r in Baseline_VaringThre_multiround_r1_rates[:-1]] + [0]
#     plt.plot(r1_bytes, r1_results, label='Round1 (Thre)', linewidth=3)

    
#     # for j in range(len(r1_results)-2):
#     #     plt.plot([r1_bytes[j], r2_bytes[j]], [r1_results[j], r2_results[j]], marker='o', linewidth=3, label='Round2(t={})'.format(communication_thres[j]))

#     r1_rates = [1/x for x in VaringThre_multiround_r1_rates[:-1]] + [0]
#     # for x, y, r, t in zip(r1_bytes, r1_results, r1_rates, communication_thres):
#     #     # print(x,y,r)
#     #     plt.text(x, y, '({:.01f},{:.02f})'.format(x,y), ha='center', va='bottom', fontsize=10)
#         # plt.text(x, y, '({},{})'.format(t,y), ha='center', va='bottom', fontsize=10)
    
#     r2_rates = [1/x for x in VaringThre_multiround_r2_rates[:-1]] + [0]
#     # for x, y, r, t in zip(r2_bytes[:-2], r2_results[:-2], r2_rates[:-2], communication_thres[:-2]):
#     #     # print(x,y,r)
#     #     plt.text(x, y, '({:.01f},{:.02f})'.format(x,y), ha='center', va='bottom', fontsize=10)
#         # plt.text(x, y, '({},{})'.format(t,y), ha='center', va='bottom', fontsize=10)

#     # r2_topk_results = [x * 100 for x in VaringThre_multiround_r2_topk_results[i]]
#     # r2_topk_bytes = [math.log2(communication_volume * 32 // 8 * r) for r in VaringThre_multiround_r2_topk_rates]
#     # plt.plot(r2_topk_bytes, r2_topk_results, marker='o', label='Round2(Topk)', linewidth=3, c='olivedrab')
    
#     # plt.scatter(math.log2(communication_volume * 32 // 8 * rate), context_results[i]* 100, label='Context', linewidth=3, c='steelblue')
    
#     plt.title('{}'.format(metric))
#     # plt.xlabel('AP')
#     plt.ylabel(metric, size=14)
#     plt.xlabel('CommunicationVolume(log2)', size=14)
#     plt.legend()
#     # plt.ylim(0, 0.4)
#     plt.savefig('CommGraph_{}_vs_Commcost.png'.format(metric))