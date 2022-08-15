
# Official Version
gpu_id=$1

feat_mode='inter'
warp_mode='HW'
coord='Global'
depth_mode='Unique'
round=1
comm_thre=0
trans_layer='0'

# HW_NO_MESSAGE
# trans_layer='-2'
# epoch='80'
# exp_id='HW_NO_MESSAGE'
# message_mode='Pointwise'

# HW_Pointwise
# epoch='last'
# exp_id='HW_Pointwise'
# message_mode='Pointwise'

# HW_When2com
# epoch='last'
# exp_id='HW_When2com'
# message_mode='When2com'

# HW_V2V
# trans_layer='2'
# epoch='100'
# exp_id='HW_V2V'
# message_mode='V2V'

# HW_Max
# epoch='last'
# epoch='80'
# exp_id='HW_Max'
# message_mode='Max'

# HW_Mean
# epoch='20'
# exp_id='HW_Mean'
# message_mode='Mean'

# Ablation
# HW_ATTEN
# epoch='last'
# exp_id='HW_ATTEN'
# message_mode='ATTEN'

# MHA
# epoch='90'
# exp_id='HW_QualityMapMessage_Translayer0_CommMask_Transformer_WOPE_Repeat'
# message_mode='QualityMapTransformerWOPE'

# Weight
# epoch='120'
# exp_id='HW_QualityMapMessage_Translayer0_CommMask_Transformer_Weight'
# message_mode='QualityMapTransformerWeight'

# BEST
# epoch='best'
# exp_id='HW_QualityMapMessage_Translayer0_CommMask_Transformer_MultiRound_VaringThre_Repeat3'
# message_mode='QualityMapTransformer'

# HW_Max
# trans_layer='2'
# epoch='last'
# exp_id='HW_Max_Rebuttal'
# message_mode='Max'

# HW_Concat
# trans_layer='2'
# epoch='last'
# exp_id='HW_Concat_Rebuttal'
# message_mode='Concat'

for i in ${epoch[*]}
do
    echo $i
    CUDA_VISIBLE_DEVICES=$gpu_id python multiagent_test.py multiagent_det \
        --exp_id $exp_id \
        --load_model '../exp/multiagent_det/'$exp_id'/model_'$i'.pth' \
        --gpus $gpu_id --coord=$coord --uav_height=40 \
        --feat_mode=$feat_mode --polygon \
        --map_scale=1.0 --warp_mode=$warp_mode --depth_mode=$depth_mode \
        --message_mode=$message_mode --trans_layer=$trans_layer --round=$round --with_occluded \
        --comm_thre=$comm_thre
    
    # CUDA_VISIBLE_DEVICES=$gpu_id python multiagent_test.py multiagent_det \
    #     --exp_id $exp_id \
    #     --load_model '../exp/multiagent_det/'$exp_id'/model_'$i'.pth' \
    #     --gpus $gpu_id --coord=$coord --uav_height=40 \
    #     --feat_mode=$feat_mode --polygon \
    #     --map_scale=1.0 --warp_mode=$warp_mode --depth_mode=$depth_mode \
    #     --message_mode=$message_mode --trans_layer=$trans_layer --round=$round \
    #     --comm_thre=$comm_thre
done