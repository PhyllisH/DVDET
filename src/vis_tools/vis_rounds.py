from certifi import where
from matplotlib import markers
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import math

communication_thres = [0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.9, 1.0]
communication_thres = [0.0, 0.001, 0.01, 0.03, 0.06, 0.08, 0.1, 0.13, 0.16, 0.20, 0.24, 0.28, 1.0]
# Model Local Transformer (Train thre 0.03)
init_communication_rates = [1, 0.108138, 0.030407, 0.015512, 0.008961, 0.006608, 0.003117, 0.001986, 0.001181, 0.000825, 0.000649, 0.000593, 0]
Where2comm = [
    [0.3968, 0.3947, 0.3786, 0.3701, 0.3647, 0.3623, 0.3568, 0.3537, 0.3501, 0.3467, 0.3439, 0.3426, 0.3256],
    [0.7454, 0.7427, 0.7324, 0.7240, 0.7176, 0.7144, 0.7073, 0.7025, 0.6965, 0.6906, 0.6853, 0.6828, 0.6342],
    [0.5029, 0.5003, 0.4733, 0.4584, 0.4496, 0.4459, 0.4378, 0.4332, 0.4285, 0.4234, 0.4201, 0.4187, 0.4028]
]

GaussianSmooth_communication_rates = [1, 0.108138, 0.009913, 0.005275, 0.003212, 0.001854, 0.000921, 0.000505, 0.000236, 0.000136, 0]
GaussianSmooth_Where2comm = [
    [0.3968, 0.3947, 0.3796, 0.3690, 0.3613, 0.3533, 0.3422, 0.3339, 0.3282, 0.3266, 0.3256],
    [0.7454, 0.7427, 0.7300, 0.7213, 0.7127, 0.7010, 0.6808, 0.6605, 0.6440, 0.6386, 0.6342],
    [0.5029, 0.5003, 0.4762, 0.4575, 0.4452, 0.4332, 0.4186, 0.4099, 0.4045, 0.4035, 0.4028]
]

GaussianSmooth_communication_rates_r2 = [1, 0.081173, 0.007459, 0.002180, 0.000855, 0.000477, 0.000002, 0]
GaussianSmooth_Where2comm_r2 = [
    [0.3591, 0.3616, 0.3621, 0.3608, 0.3545, 0.3453, 0.3371, 0.3371],
    [0.6447, 0.6464, 0.6440, 0.6436, 0.6384, 0.6254, 0.6046, 0.6046],
    [0.3701, 0.3726, 0.3732, 0.3709, 0.3617, 0.3500, 0.3454, 0.3454]
]

GaussianSmooth_communication_rates_r3 = [1, 0.081173, 0.007459, 0.002180, 0.000855, 0.000477, 0.000002, 0]
GaussianSmooth_Where2comm_r3 = [
    [0.3591, 0.3616, 0.3621, 0.3608, 0.3545, 0.3453, 0.3371, 0.3371],
    [0.6447, 0.6464, 0.6440, 0.6436, 0.6384, 0.6254, 0.6046, 0.6046],
    [0.3701, 0.3726, 0.3732, 0.3709, 0.3617, 0.3500, 0.3454, 0.3454]
]

No_Message = [0.2843, 0.5756, 0.3492]
# Late_Fusion = [0.2843, 0.5756, 0.3492]
V2V = [0.3226, 0.6346, 0.3968]
When2com = [0.3230, 0.6372, 0.3980]
DiscoNet = [0.3097, 0.6227, 0.3788]
where2comm = [0.3968, 0.7454, 0.5029]


communication_volume = 192*352*64*5*4
max_volume = math.log2(communication_volume * 32 // 8)

fontsize = 13
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
plt.tick_params(labelsize=20)
for i, metric in enumerate(metrics[:3]):
    # fig = plt.figure(figsize=(12,10))
    fig = plt.figure()
    plt.scatter(max_volume, where2comm[i]* 100, label='Where2comm', linewidth=3, c='steelblue')
    plt.text(max_volume, where2comm[i]* 100, '({:.02f},{:.02f})'.format(max_volume,where2comm[i]* 100), ha='center', va='bottom', fontsize=fontsize)
    plt.scatter(0, No_Message[i]* 100, label='NoCollaboration', linewidth=3, c='mediumpurple')
    plt.text(0, No_Message[i]* 100, 'NoColla({:.02f},{:.02f})'.format(0,No_Message[i]* 100), ha='center', va='bottom', fontsize=fontsize)
    plt.scatter(max_volume+math.log2(3), V2V[i]* 100, label='V2V-3Round', linewidth=3, c='orange')
    plt.text(max_volume+math.log2(3), V2V[i]* 100, 'V2V ({:.02f},{:.02f})'.format(max_volume+math.log2(3),V2V[i]* 100), ha='center', va='bottom', fontsize=fontsize)
    plt.scatter(max_volume, DiscoNet[i]* 100, label='DiscoNet', linewidth=3, c='olivedrab')
    plt.text(max_volume, DiscoNet[i]* 100, 'DiscoNet ({:.02f},{:.02f})'.format(max_volume,DiscoNet[i]* 100), ha='center', va='bottom', fontsize=fontsize)
    plt.scatter(max_volume, When2com[i]* 100, label='When2com', linewidth=3, c='violet')
    plt.text(max_volume+math.log2(0.7), When2com[i]* 100, 'When2com ({:.02f},{:.02f})'.format(max_volume,When2com[i]* 100), ha='center', va='bottom', fontsize=fontsize)
    plt.scatter(0, Where2comm[i][-1]* 100, label='Where2comm(NoBandLimit)', linewidth=3, c='steelblue')
    plt.text(0, Where2comm[i][-1]* 100, '({:.02f},{:.02f})'.format(0,Where2comm[i][-1]* 100), ha='center', va='bottom', fontsize=fontsize)
    
    # communication_rates, multi_agent_withthre_results = Thre003_communication_rates, Thre003_multi_agent_withthre_results
    communication_rates, multi_agent_withthre_results = init_communication_rates, Where2comm
    multi_agent_withthre_results = [x * 100 for x in multi_agent_withthre_results[i]]
    communication_volume_bytes = [math.log2(communication_volume * 32 // 8 * r) for r in communication_rates[:-1]] + [0]
    plt.plot(communication_volume_bytes, multi_agent_withthre_results, label='Where2comm', linewidth=3)
    communication_rates = [1/x for x in communication_rates[:-1]] + [0]

    plt.title('{}'.format(metric))
    # plt.xlabel('AP')
    plt.ylabel(metric, size=16)
    plt.xlabel('CommunicationVolume(log2)', size=16)
    plt.legend(loc=2,prop={'size': 11})
    # plt.ylim(0, 0.4)
    plt.savefig('SOTA_Performance_{}_vs_Commcost.png'.format(metric))

# Figure3: Gaussian Smooth
metrics = ['AP', 'AP@0.50', 'AP@0.70']
plt.tick_params(labelsize=20)
for i, metric in enumerate(metrics[:3]):
    # fig = plt.figure(figsize=(12,10))
    fig = plt.figure()
    
    communication_rates, multi_agent_withthre_results = init_communication_rates, Where2comm
    multi_agent_withthre_results = [x * 100 for x in multi_agent_withthre_results[i]]
    communication_volume_bytes = [math.log2(communication_volume * 32 // 8 * r) for r in communication_rates[:-1]] + [0]
    plt.plot(communication_volume_bytes, multi_agent_withthre_results, label='Where2comm', linewidth=3, c='seagreen')
    communication_rates = [1/x for x in communication_rates[:-1]] + [0]

    communication_rates, multi_agent_withthre_results = GaussianSmooth_communication_rates, GaussianSmooth_Where2comm
    multi_agent_withthre_results = [x * 100 for x in multi_agent_withthre_results[i]]
    communication_volume_bytes = [math.log2(communication_volume * 32 // 8 * r) for r in communication_rates[:-1]] + [0]
    plt.plot(communication_volume_bytes, multi_agent_withthre_results, label='Where2comm(Gaussian)', linewidth=3)
    communication_rates = [1/x for x in communication_rates[:-1]] + [0]

    plt.title('{}'.format(metric))
    # plt.xlabel('AP')
    plt.ylabel(metric, size=16)
    plt.xlabel('CommunicationVolume(log2)', size=16)
    plt.legend(loc=2,prop={'size': 11})
    # plt.ylim(0, 0.4)
    plt.savefig('GaussianSmooth_{}_vs_Commcost.png'.format(metric))


# Figure2: MultiRound Communication
metrics = ['AP', 'AP@0.50', 'AP@0.70']
plt.tick_params(labelsize=20)
for i, metric in enumerate(metrics[:3]):
    # fig = plt.figure(figsize=(12,10))
    fig = plt.figure()
    
    communication_rates, multi_agent_withthre_results = GaussianSmooth_communication_rates_r3, GaussianSmooth_Where2comm_r3
    multi_agent_withthre_results = [x * 100 for x in multi_agent_withthre_results[i]]
    communication_volume_bytes = [math.log2(communication_volume * 32 // 8 * r) for r in communication_rates[:-1]] + [0]
    plt.plot(communication_volume_bytes, multi_agent_withthre_results, label='Where2comm(Round3)', linewidth=3)
    communication_rates = [1/x for x in communication_rates[:-1]] + [0]

    plt.scatter(communication_volume_bytes[0], multi_agent_withthre_results[0], label='Where2comm(Round3_NoBandLimit)', linewidth=3)
    plt.text(communication_volume_bytes[0], multi_agent_withthre_results[0], '({:.02f},{:.02f})'.format(communication_volume_bytes[0],multi_agent_withthre_results[0]), ha='center', va='bottom', fontsize=fontsize)
    

    communication_rates, multi_agent_withthre_results = GaussianSmooth_communication_rates_r2, GaussianSmooth_Where2comm_r2
    multi_agent_withthre_results = [x * 100 for x in multi_agent_withthre_results[i]]
    communication_volume_bytes = [math.log2(communication_volume * 32 // 8 * r) for r in communication_rates[:-1]] + [0]
    plt.plot(communication_volume_bytes, multi_agent_withthre_results, label='Where2comm(Round2)', linewidth=3)
    communication_rates = [1/x for x in communication_rates[:-1]] + [0]

    communication_rates, multi_agent_withthre_results = GaussianSmooth_communication_rates, GaussianSmooth_Where2comm
    multi_agent_withthre_results = [x * 100 for x in multi_agent_withthre_results[i]]
    communication_volume_bytes = [math.log2(communication_volume * 32 // 8 * r) for r in communication_rates[:-1]] + [0]
    plt.plot(communication_volume_bytes, multi_agent_withthre_results, label='Where2comm(Round1)', linewidth=3, c='seagreen')
    communication_rates = [1/x for x in communication_rates[:-1]] + [0]
    
    plt.scatter(communication_volume_bytes[0], multi_agent_withthre_results[0], label='Where2comm(Round1_NoBandLimit)', linewidth=3, c='seagreen')
    plt.text(communication_volume_bytes[0], multi_agent_withthre_results[0], '({:.02f},{:.02f})'.format(communication_volume_bytes[0],multi_agent_withthre_results[0]), ha='center', va='bottom', fontsize=fontsize)
    

    plt.title('{}'.format(metric))
    # plt.xlabel('AP')
    plt.ylabel(metric, size=16)
    plt.xlabel('CommunicationVolume(log2)', size=16)
    plt.legend(loc=2,prop={'size': 11})
    # plt.ylim(0, 0.4)
    plt.savefig('Multiround_{}_vs_Commcost.png'.format(metric))