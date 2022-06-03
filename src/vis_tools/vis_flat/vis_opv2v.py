from ctypes import pointer
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

GaussianSmooth_communication_rates = [1, 0.4697544, 0.21919, 0.110502, 0.0485766, 0.0245, 0.01774, 0.01387295, 0.011219, 0.008926, 0.006466, 0.0045, 0.0031076, 0]

GaussianSmooth_Where2comm = [
    [0.6600056061308296, 0.6584374085392466, 0.6468196726630717, 0.646258816447803, 0.6416723288943581, 0.6356335578794745, 0.6353987772869524, 0.6325465691553946, 0.6333008559406559, 0.6301998428773053, 0.6234860289854047, 0.6179743684129745, 0.610059709980962, 0.5986610493835538],
    [0.4397077925251002, 0.4392955219428226, 0.4315567023914016, 0.43139742957985094, 0.42757439958828014, 0.4260165321842948, 0.4245278913246697, 0.4233269637008502, 0.42024323137484537, 0.4163711406573533, 0.4103993439749715, 0.4045009291908875, 0.4009773867209733, 0.38813528058426366],
    [0.17234750434479895, 0.17276072616069812, 0.17179075626273213, 0.17427852448250475, 0.17228222845949479, 0.16913210014095767, 0.16854170486470668, 0.1672377720644106, 0.16607293164649037, 0.165616058748732, 0.16357703468910126, 0.16151463861552107, 0.16036979309609353, 0.15648334396426142],
]

# multi-6, base_thre trained, two-round
GaussianSmooth_communication_rates_r2 = [1.207677, 0.52445, 0.435066, 0.31347, 0.195917, 0.113394, 0.0675, 0.047419, 0.0325612, 0.024507, 0.01890994, 0.01392, 0.00998, 0.00650067, 0]

GaussianSmooth_Where2comm_r2 = [
    [0.6677932597354528, 0.6683717097465753, 0.6692604317113963, 0.6699106835819177, 0.6691110784423315, 0.6623291756192445, 0.6555754049427269, 0.6496837666863565, 0.6447053752793974, 0.6387165472377423, 0.636425867797983, 0.6334568985003514, 0.6269655825376627, 0.617282775126239, 0.5981215582401571],
    [0.46197173391328217, 0.46083132989576664, 0.4595008445032923, 0.4594377128606246, 0.4540234579756745, 0.4502621616933667, 0.44561150835441166, 0.44189139946331835, 0.43924786378125324, 0.43534784902910584, 0.4326750270300926, 0.429926855823889, 0.42356686080239475, 0.4116859146271755, 0.39003303136859613],
    [0.18333632338603695, 0.18240517365672726, 0.18230685728555837, 0.18317958699662362, 0.1817187520805222, 0.1811901689835662, 0.17835262000858865, 0.17742177508507861, 0.1737809200774934, 0.17045198712543524, 0.16821984529552125, 0.16726583639477544, 0.16615893513663496, 0.16135291621125145, 0.15207636305632644]
]


# multi3_1 & 2
# GaussianSmooth_communication_rates_r3 = [1.339044, 0.533025, 0.3649054, 0.24076, 0.150497034, 0.0969433, 0.0692113, 0.046438, 0.034365, 0.02628, 0.0195102, 0.0140386, 0.0089689, 0]

# GaussianSmooth_Where2comm_r3 = [
#     [0.6728657702736681, 0.6694307034414217, 0.670680922547378, 0.6691477620148685, 0.6656237607928538, 0.6565800220121405, 0.6493068051646531,  0.6406852700159512, 0.6358229447200525, 0.636003145448309, 0.6354575265301962, 0.6332479868963521, 0.6191554409265326, 0.5981931055077964],
#     [0.4714416296371197, 0.4731016449291139, 0.47320470614676396, 0.473566424379574, 0.4708731959590588, 0.4603811595313383, 0.45230290815005886,  0.4467870801831065, 0.4407085489798917, 0.4406921826770181, 0.43684378054409245, 0.43374600494238374, 0.42152438286383237, 0.4011335836871555],
#     [0.19073043099286743, 0.19069989109897673, 0.18836752232059845, 0.18626414759459173, 0.1842017285585523, 0.18227099165915386, 0.1802208449553001, 0.1777258421844148, 0.1735037315853901, 0.17146586873462594, 0.17117463577456757, 0.16836500147003142, 0.16091460671807545, 0.15361522854572895]
# ]

GaussianSmooth_communication_rates_r3 = [1.339044, 0.150497034, 0.0969433, 0.0692113, 0.046438, 0.02628, 0.0195102, 0.0140386, 0.0089689, 1e-5, 0]

GaussianSmooth_Where2comm_r3 = [
    [0.6728657702736681,  0.6656237607928538, 0.6565800220121405, 0.6493068051646531,  0.6406852700159512,  0.636003145448309, 0.6354575265301962, 0.6332479868963521, 0.6191554409265326, 0.5981931055077964, 0.5981931055077964],
    [0.4714416296371197,  0.4708731959590588, 0.4603811595313383, 0.45230290815005886,  0.4467870801831065,  0.4406921826770181, 0.43684378054409245, 0.43374600494238374, 0.42152438286383237, 0.4011335836871555, 0.4011335836871555],
    [0.19073043099286743,  0.1842017285585523, 0.18227099165915386, 0.1802208449553001, 0.1777258421844148, 0.17146586873462594, 0.17117463577456757, 0.16836500147003142, 0.16091460671807545, 0.15361522854572895, 0.15361522854572895]
]

# GaussianSmooth_communication_rates_r3 = [1.339044, 0.533025, 0.3649054, 0.24076, 0.150497034, 0.0969433, 0.0692113, 0.046438, 0.034365, 0.02628, 0.0195102, 0.015891368, 0.0110106, 0.005306, 0.0042626, 0.002924765, 1e-5, 0]

# GaussianSmooth_Where2comm_r3 = [
#     [0.6728657702736681, 0.6694307034414217, 0.670680922547378, 0.6691477620148685, 0.6656237607928538, 0.6565800220121405, 0.6493068051646531,  0.6406852700159512, 0.6358229447200525, 0.636003145448309, 0.6354575265301962, 0.6332158028935654, 0.6228655450780333, 0.6166205345666335, 0.6152794927692005, 0.6104981109374231, 0.5981931055077964, 0.5981931055077964],
#     [0.4714416296371197, 0.4731016449291139, 0.47320470614676396, 0.473566424379574, 0.4708731959590588, 0.4603811595313383, 0.45230290815005886,  0.4467870801831065, 0.4407085489798917, 0.4406921826770181, 0.43684378054409245, 0.43323844268460865, 0.42635412295930647, 0.4177834375246076, 0.41613275576817493, 0.41283684036096324,  0.4011335836871555, 0.4011335836871555],
#     [0.19073043099286743, 0.19069989109897673, 0.18836752232059845, 0.18626414759459173, 0.1842017285585523, 0.18227099165915386, 0.1802208449553001, 0.1777258421844148, 0.1735037315853901, 0.17146586873462594, 0.17117463577456757, 0.17029353172299602, 0.16930917593075603, 0.1644437192707898, 0.16298003918823412, 0.16054966773482635, 0.15361522854572895, 0.15361522854572895]
# ]

# cam result in (40, 40)
No_Message = [0.3819674673801962, 0.2265724821878737, 0.09093133350867384]
LateFusion = [0.12204330698885894, 0.08236031374857278, 0.03843385007150976]
V2V = [0.5883738916258715, 0.37466369268190924, 0.14674561249806656]
When2com = [0.3226467986803283, 0.19694880142112364, 0.08288704255003766]
DiscoNet = [0.5601581029433428, 0.3600396408009624, 0.125080494472925]
# Transformer_results = [0.5757791310508683, 0.3951667164450668, 0.17070904122286915]
# QualityMap_results = [0.606922534114849, 0.4348083362491921, 0.17125444778467122]
# where2comm = [0.6600056061308296, 0.4397077925251002, 0.17234750434479895]
where2comm = [GaussianSmooth_Where2comm_r3[0][0], GaussianSmooth_Where2comm_r3[1][0], GaussianSmooth_Where2comm_r3[2][0]]

communication_volume = 200*200*32
max_volume = math.log2(communication_volume * 32 // 8)

box_num = 6.7*5*4*7
Late_cost = math.log2(box_num * 32 // 8)


fontsize = 20
label_size = 18
legend_size = 17
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


# Figure1: Compare with SOTAs
metrics = ['AP', 'AP@0.50', 'AP@0.70']
plt.tick_params(labelsize=labelsize)
for i, metric in enumerate(metrics[:3]):
    # fig = plt.figure(figsize=(12,10))
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    plt.scatter(max_volume, where2comm[i]* 100, label='Where2comm', linewidth=3, c='red', s=point_size)
    plt.text(max_volume-3, where2comm[i]* 100-4, 'Where2comm'.format(max_volume,where2comm[i]* 100), ha='center', va='bottom', fontsize=fontsize)
    plt.scatter(0, No_Message[i]* 100, label='NoCollaboration', linewidth=3, c='mediumpurple', s=point_size)
    plt.text(7.5, No_Message[i]* 100, 'NoCollaboration'.format(0,No_Message[i]* 100), ha='center', va='bottom', fontsize=fontsize)
    plt.scatter(max_volume+math.log2(3), V2V[i]* 100, label='V2V', linewidth=3, c='orange', s=point_size)
    plt.text(max_volume+math.log2(3)-1, V2V[i]* 100+0.7, 'V2V'.format(max_volume+math.log2(3),V2V[i]* 100), ha='center', va='bottom', fontsize=fontsize)
    plt.scatter(max_volume, DiscoNet[i]* 100, label='DiscoNet', linewidth=3, c='olivedrab', s=point_size)
    plt.text(max_volume-2, DiscoNet[i]* 100-4.5, 'DiscoNet'.format(max_volume,DiscoNet[i]* 100), ha='center', va='bottom', fontsize=fontsize)
    plt.scatter(max_volume, When2com[i]* 100, label='When2com', linewidth=3, c='violet', s=point_size)
    plt.text(max_volume+math.log2(0.7)-2, When2com[i]* 100+0.6, 'When2com'.format(max_volume,When2com[i]* 100), ha='center', va='bottom', fontsize=fontsize)
    # plt.scatter(0, Where2comm[i][-1]* 100, label='Where2comm(NoBandLimit)', linewidth=3, c='steelblue')
    # plt.text(0, Where2comm[i][-1]* 100, '({:.02f},{:.02f})'.format(0,Where2comm[i][-1]* 100), ha='center', va='bottom', fontsize=fontsize)
    plt.scatter(Late_cost, LateFusion[i]* 100, label='LateFusion', linewidth=3, c='steelblue', s=point_size)
    plt.text(Late_cost, LateFusion[i]* 100+0.5, 'LateFusion'.format(max_volume,LateFusion[i]* 100), ha='center', va='bottom', fontsize=fontsize)
    
    # communication_rates, multi_agent_withthre_results = Thre003_communication_rates, Thre003_multi_agent_withthre_results
    communication_rates, multi_agent_withthre_results = GaussianSmooth_communication_rates_r3, GaussianSmooth_Where2comm_r3
    multi_agent_withthre_results = [x * 100 for x in multi_agent_withthre_results[i]]
    communication_volume_bytes = [math.log2(communication_volume * 32 // 8 * r) for r in communication_rates[:-1]] + [0]
    plt.plot(communication_volume_bytes[:-1], multi_agent_withthre_results[:-1], label='Where2comm', linewidth=3)
    
    communication_rates = [1/x for x in communication_rates[:-1]] + [0]

    ratio = 0.82
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)

    # plt.title('{}'.format(metric), size=label_size)
    # plt.xlabel('AP')
    plt.ylabel(metric, size=label_size)
    plt.xlabel('CommunicationVolume(log2)', size=label_size)
    # plt.legend(prop={'size': legend_size})
    # plt.ylim(0, 0.4)
    plt.savefig('SOTA_Performance_{}_vs_Commcost.png'.format(metric))


# Figure2: MultiRound Communication
metrics = ['AP', 'AP@0.50', 'AP@0.70']
plt.tick_params(labelsize=labelsize)
for i, metric in enumerate(metrics[:3]):
    # fig = plt.figure(figsize=(12,10))
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    communication_rates, multi_agent_withthre_results = GaussianSmooth_communication_rates_r3, GaussianSmooth_Where2comm_r3
    multi_agent_withthre_results = [x * 100 for x in multi_agent_withthre_results[i]]
    communication_volume_bytes = [math.log2(communication_volume * 32 // 8 * r) for r in communication_rates[:-1]] + [0]
    plt.plot(communication_volume_bytes[:-2], multi_agent_withthre_results[:-2], label='3 Rounds', linewidth=3)
    communication_rates = [1/x for x in communication_rates[:-1]] + [0]

    # plt.scatter(communication_volume_bytes[0], multi_agent_withthre_results[0], label='Round3', linewidth=3, s=point_size)
    # plt.text(communication_volume_bytes[0]-0.7, multi_agent_withthre_results[0]-0.7, 'Round3'.format(communication_volume_bytes[0],multi_agent_withthre_results[0]), ha='center', va='bottom', fontsize=fontsize)
    

    communication_rates, multi_agent_withthre_results = GaussianSmooth_communication_rates_r2, GaussianSmooth_Where2comm_r2
    multi_agent_withthre_results = [x * 100 for x in multi_agent_withthre_results[i]]
    communication_volume_bytes = [math.log2(communication_volume * 32 // 8 * r) for r in communication_rates[:-1]] + [0]
    plt.plot(communication_volume_bytes[:-1], multi_agent_withthre_results[:-1], label='2 Rounds', linewidth=3)
    communication_rates = [1/x for x in communication_rates[:-1]] + [0]

    # plt.scatter(communication_volume_bytes[0], multi_agent_withthre_results[0], label='Round2', linewidth=3, s=point_size)
    # plt.text(communication_volume_bytes[0]-0.5, multi_agent_withthre_results[0]-0.8, 'Round2'.format(communication_volume_bytes[0],multi_agent_withthre_results[0]), ha='center', va='bottom', fontsize=fontsize)
    

    communication_rates, multi_agent_withthre_results = GaussianSmooth_communication_rates, GaussianSmooth_Where2comm
    multi_agent_withthre_results = [x * 100 for x in multi_agent_withthre_results[i]]
    communication_volume_bytes = [math.log2(communication_volume * 32 // 8 * r) for r in communication_rates[:-1]] + [0]
    plt.plot(communication_volume_bytes[:-1], multi_agent_withthre_results[:-1], label='1 Round', linewidth=3, c='seagreen')
    communication_rates = [1/x for x in communication_rates[:-1]] + [0]
    
    # plt.scatter(communication_volume_bytes[0], multi_agent_withthre_results[0], label='Round1', linewidth=3, c='seagreen', s=point_size)
    # plt.text(communication_volume_bytes[0]-0.3, multi_agent_withthre_results[0]-0.8, 'Round1'.format(communication_volume_bytes[0],multi_agent_withthre_results[0]), ha='center', va='bottom', fontsize=fontsize)
    
    ratio = 0.82
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)

    # plt.title('{}'.format(metric), size=label_size)
    # plt.xlabel('AP')
    plt.ylabel(metric, size=label_size)
    plt.xlabel('CommunicationVolume(log2)', size=label_size)
    plt.legend(loc=4,prop={'size': legend_size})
    # plt.ylim(0, 0.4)
    plt.savefig('Multiround_{}_vs_Commcost.png'.format(metric))


# Figure3: Gaussian Smooth
metrics = ['AP', 'AP@0.50', 'AP@0.70']
plt.tick_params(labelsize=labelsize)
for i, metric in enumerate(metrics[:3]):
    # fig = plt.figure(figsize=(7,7))
    # fig = plt.figure(figsize=(7,5.5))
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    
    communication_rates, multi_agent_withthre_results = init_communication_rates, Where2comm
    multi_agent_withthre_results = [x * 100 for x in multi_agent_withthre_results[i]]
    communication_volume_bytes = [math.log2(communication_volume * 32 // 8 * r) for r in communication_rates[:-1]] + [0]
    plt.plot(communication_volume_bytes[:-5], multi_agent_withthre_results[:-5], label='NOGaussianSmooth', linewidth=3, c='seagreen')
    communication_rates = [1/x for x in communication_rates[:-1]] + [0]

    communication_rates, multi_agent_withthre_results = GaussianSmooth_communication_rates, GaussianSmooth_Where2comm
    multi_agent_withthre_results = [x * 100 for x in multi_agent_withthre_results[i]]
    communication_volume_bytes = [math.log2(communication_volume * 32 // 8 * r) for r in communication_rates[:-1]] + [0]
    plt.plot(communication_volume_bytes[:-4], multi_agent_withthre_results[:-4], label='GaussianSmooth', linewidth=3)
    communication_rates = [1/x for x in communication_rates[:-1]] + [0]

    plt.title('{}'.format(metric))
    # plt.xlabel('AP')
    plt.ylabel(metric, size=label_size)
    plt.xlabel('CommunicationVolume(log2)', size=label_size)
    plt.legend(loc=2,prop={'size': legend_size})
    # plt.ylim(0, 0.4)
    ratio = 1.1
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    plt.savefig('GaussianSmooth_{}_vs_Commcost.png'.format(metric))
