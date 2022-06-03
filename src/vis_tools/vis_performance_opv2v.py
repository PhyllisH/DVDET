      
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import math
import os

# qua_3, one_round, 1e-7 finetuned

VaringThre_communication_rates_6 = [1, 0.4697544, 0.21919, 0.110502, 0.0485766, 0.0245, 0.01774, 0.01387295, 0.011219, 0.008926, 0.006466, 0.0045, 0.0031076, 0]

VaringThre_multi_agent_withthre_results_6 = [
    [0.6600056061308296, 0.6584374085392466, 0.6468196726630717, 0.646258816447803, 0.6416723288943581, 0.6356335578794745, 0.6353987772869524, 0.6325465691553946, 0.6333008559406559, 0.6301998428773053, 0.6234860289854047, 0.6179743684129745, 0.610059709980962, 0.5986610493835538],
    [0.4397077925251002, 0.4392955219428226, 0.4315567023914016, 0.43139742957985094, 0.42757439958828014, 0.4260165321842948, 0.4245278913246697, 0.4233269637008502, 0.42024323137484537, 0.4163711406573533, 0.4103993439749715, 0.4045009291908875, 0.4009773867209733, 0.38813528058426366],
    [0.17234750434479895, 0.17276072616069812, 0.17179075626273213, 0.17427852448250475, 0.17228222845949479, 0.16913210014095767, 0.16854170486470668, 0.1672377720644106, 0.16607293164649037, 0.165616058748732, 0.16357703468910126, 0.16151463861552107, 0.16036979309609353, 0.15648334396426142],
]

# multi-6, base_thre trained, two-round
VaringThre_communication_rates_10 = [1.207677, 0.52445, 0.435066, 0.31347, 0.195917, 0.113394, 0.0675, 0.047419, 0.0325612, 0.024507, 0.01890994, 0.01392, 0.00998, 0.00650067, 0]

VaringThre_multi_agent_withthre_results_10 = [
    [0.6677932597354528, 0.6683717097465753, 0.6692604317113963, 0.6699106835819177, 0.6691110784423315, 0.6623291756192445, 0.6555754049427269, 0.6496837666863565, 0.6447053752793974, 0.6387165472377423, 0.636425867797983, 0.6334568985003514, 0.6269655825376627, 0.617282775126239, 0.5981215582401571],
    [0.46197173391328217, 0.46083132989576664, 0.4595008445032923, 0.4594377128606246, 0.4540234579756745, 0.4502621616933667, 0.44561150835441166, 0.44189139946331835, 0.43924786378125324, 0.43534784902910584, 0.4326750270300926, 0.429926855823889, 0.42356686080239475, 0.4116859146271755, 0.39003303136859613],
    [0.18333632338603695, 0.18240517365672726, 0.18230685728555837, 0.18317958699662362, 0.1817187520805222, 0.1811901689835662, 0.17835262000858865, 0.17742177508507861, 0.1737809200774934, 0.17045198712543524, 0.16821984529552125, 0.16726583639477544, 0.16615893513663496, 0.16135291621125145, 0.15207636305632644]
]


# multi3_1 & 2
VaringThre_communication_rates_11 = [1.339044, 0.533025, 0.3649054, 0.24076, 0.150497034, 0.0969433, 0.0692113, 0.046438, 0.034365, 0.02628, 0.0195102, 0.0140386, 0.0089689, 0]

VaringThre_multi_agent_withthre_results_11 = [
    [0.6728657702736681, 0.6694307034414217, 0.670680922547378, 0.6691477620148685, 0.6656237607928538, 0.6565800220121405, 0.6493068051646531,  0.6406852700159512, 0.6358229447200525, 0.636003145448309, 0.6354575265301962, 0.6332479868963521, 0.6191554409265326, 0.5981931055077964],
    [0.4714416296371197, 0.4731016449291139, 0.47320470614676396, 0.473566424379574, 0.4708731959590588, 0.4603811595313383, 0.45230290815005886,  0.4467870801831065, 0.4407085489798917, 0.4406921826770181, 0.43684378054409245, 0.43374600494238374, 0.42152438286383237, 0.4011335836871555],
    [0.19073043099286743, 0.19069989109897673, 0.18836752232059845, 0.18626414759459173, 0.1842017285585523, 0.18227099165915386, 0.1802208449553001, 0.1777258421844148, 0.1735037315853901, 0.17146586873462594, 0.17117463577456757, 0.16836500147003142, 0.16091460671807545, 0.15361522854572895]
]


# cam result in (40, 40)
single_agent_results = [0.3819674673801962, 0.2265724821878737, 0.09093133350867384]
V2V_results = [0.5883738916258715, 0.37466369268190924, 0.14674561249806656]
When2com_results = [0.3226467986803283, 0.19694880142112364, 0.08288704255003766]
DiscoNet_results = [0.5601581029433428, 0.3600396408009624, 0.125080494472925]
Transformer_results = [0.5757791310508683, 0.3951667164450668, 0.17070904122286915]
QualityMap_results = [0.606922534114849, 0.4348083362491921, 0.17125444778467122]

multi_agent_results = [0.3738, 0.6539, 0.3853]

communication_volume = 200*200*32
# max_volume = math.log2(communication_volume)
max_volume = math.log2(communication_volume * 32 // 8)


fontsize = 14
label_size = 18
legend_size = 14
tick_size = 20
labelsize = 20
figsize=(5.5,5.5)
params = {
    # 'legend.fontsize': 'x-large',
        # 'figure.figsize': (9, 7),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)



metrics = ['AP@0.30', 'AP@0.50', 'AP@0.70']
plt.tick_params(labelsize=20)
for i, metric in enumerate(metrics[:3]):
    # fig = plt.figure(figsize=(12,10))
    fig = plt.figure()
    # plt.scatter(max_volume, multi_agent_results[i]* 100, label='BandwidthNoLimit-Collaboration', linewidth=3, c='steelblue')
    # plt.text(max_volume, multi_agent_results[i]* 100, '({},{:.02f})'.format(1,multi_agent_results[i]* 100), ha='center', va='bottom', fontsize=10)
    plt.scatter(0, single_agent_results[i]* 100, label='No Collaboration', linewidth=3, c='mediumpurple')
    plt.text(0, single_agent_results[i]* 100, 'No Collaboration({},{:.02f})'.format(0,single_agent_results[i]* 100), ha='center', va='bottom', fontsize=10)
    plt.scatter(max_volume+math.log2(3), V2V_results[i]* 100, label='V2V-2Round', linewidth=3, c='orange')
    plt.text(max_volume+math.log2(3), V2V_results[i]* 100, 'V2V ({},{:.02f})'.format(0.33,V2V_results[i]* 100), ha='center', va='bottom', fontsize=10)
    plt.scatter(max_volume, DiscoNet_results[i]* 100, label='DiscoNet', linewidth=3, c='olivedrab')
    plt.text(max_volume, DiscoNet_results[i]* 100, 'DiscoNet ({},{:.02f})'.format(1,DiscoNet_results[i]* 100), ha='center', va='bottom', fontsize=10)
    plt.scatter(max_volume, When2com_results[i]* 100, label='When2com', linewidth=3, c='violet')
    plt.text(max_volume, When2com_results[i]* 100, 'When2com ({:.02f},{:.02f})'.format(1.43,When2com_results[i]* 100), ha='center', va='bottom', fontsize=10)

    plt.scatter(max_volume, Transformer_results[i]* 100, label='Transformer', linewidth=3, c='steelblue')
    plt.text(max_volume, Transformer_results[i]* 100, 'Transformer ({},{:.02f})'.format(1,Transformer_results[i]* 100), ha='center', va='bottom', fontsize=10)
    # plt.scatter(max_volume+math.log2(0.033), QualityMap_results[i]* 100, label='QualityMap', linewidth=3, c='lightcoral')
    # plt.text(max_volume+math.log2(0.033), QualityMap_results[i]* 100, 'QualityMap ({},{:.02f})'.format(1,QualityMap_results[i]* 100), ha='center', va='bottom', fontsize=10)
    
    communication_rates = [VaringThre_communication_rates_6, 
                           VaringThre_communication_rates_10,
                           VaringThre_communication_rates_11]
    multi_agent_withthre_results = [VaringThre_multi_agent_withthre_results_6,
                                    VaringThre_multi_agent_withthre_results_10,
                                    VaringThre_multi_agent_withthre_results_11]
    labels = ['BandwidthLimited', 'Two round', 'Three round']
    for j in range(len(labels)):
        communication_rate = communication_rates[j]
        multi_agent_withthre_result = multi_agent_withthre_results[j]
        label = labels[j]
        print(communication_rate)
        multi_agent_withthre_result = [x * 100 for x in multi_agent_withthre_result[i]]
        communication_volume_bytes = [math.log2(communication_volume * r) for r in communication_rate[:-1]] + [0]
        # communication_volume_bytes = [math.log2(communication_volume * r) for r in communication_rate[:]]
        plt.plot(communication_volume_bytes, multi_agent_withthre_result, marker='o', label=label, linewidth=3)
        # communication_rate = [1/x for x in communication_rate[:-1]] + [0]
        # for x, y, r, t in zip(communication_volume_bytes, multi_agent_withthre_results, communication_rates, communication_thres):
        #     # print(x,y,r)
        #     # plt.text(x, y, '({},{})'.format(int(r),y), ha='center', va='bottom', fontsize=10)
        #     plt.text(x, y, '({},{})'.format(t,y), ha='center', va='bottom', fontsize=10)
    
    plt.title('{}'.format(metric))
    # plt.xlabel('AP')
    plt.ylabel(metric, size=14)
    plt.xlabel('CommunicationVolume(log2)', size=14)
    plt.legend()
    # plt.ylim(0, 0.4)
    result_path = '/GPFS/data/shfang/repository/co-monocular-3d/performance_visualization'
    save_file = os.path.join(result_path, 'CommGraph_{}_vs_Commcost.png'.format(metric))
    plt.savefig(save_file)

    