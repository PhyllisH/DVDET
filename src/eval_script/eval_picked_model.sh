# Official Version
# gpu_id=5

# epoch=('20' '40' '60' '80' '100' '120' '140' '160' '180' '200' 'last' 'best')
# epoch=('last')
# epoch=('10' '20' '30' '40' '50' '60' '70' '80' '90' '100' '110' '120' '130' '140' '150' '160' '170' '180' '190' '200' 'last' 'best')
# epoch=('110' '120' '130' '140' '150' '160' '170' '180' '190' '200' 'last' 'best')
# epoch=('50' '60' '70' '80' '90' '100' '110' '120' '130' '140' '150' '160' '170' '180' '190' '200' 'last' 'best')
# epoch=('140' 'last')

# exp_id='LocalCoord_repeat'
# exp_id='CornerLoss_GlobalCoord_Inter'
# exp_id='CornerLoss_GlobalCoord_Inter_Revised'
# exp_id='CornerLoss_GlobalCoord_Early'
# exp_id='GlobalCoord_Early'
# exp_id='GlobalCoord_Inter'
# exp_id='GlobalCoord_Inter_Repeat'
# exp_id='GlobalCoord_Inter_LW'
# exp_id='GlobalCoord_Inter_RLW_Repeat'
# exp_id='GlobalCoord_Inter_RLW'
# exp_id='GlobalCoord_Inter_DW'
# exp_id='GlobalCoord_Inter_DW_Repeat'
# exp_id='GlobalCoord_Inter_DADW'
# exp_id='GlobalCoord_Inter_DADW_Repeat'
# exp_id='GlobalCoord_Inter_DADW_Revised'
# exp_id='GlobalCoord_Inter_HW_WeightedDepth_RevisedZ'
# exp_id='GlobalCoord_Inter_DADW_WeightedDepth'
# exp_id='GlobalCoord_Inter_DADW_WeightedDepth_Repeat'
# exp_id='GlobalCoord_Inter_DADW_WeightedDepth_Revised'
# exp_id='GlobalCoord_Inter_DADW_WeightedDepth_RevisedZ'
# exp_id='GlobalCoord_Inter_DADW_WeightedDepth_Revised_NoZsupervision'
# exp_id='JointCoord_Inter_DADW_WeightedDepth'
# exp_id='JointCoord_Inter_DADW_WeightedDepth_Repeat'
# exp_id='GlobalCoord_Inter_DADW_Revised_Repeat'

# feat_mode='inter'
# feat_mode='early'

# warp_mode='DW'
# warp_mode='DADW'
# warp_mode='HW'
# warp_mode='RLW'
# warp_mode='LW'

# coord='Joint'
# coord='Global'
# coord='Local'

# depth_mode='Unique'
# depth_mode='Weighted'


# Early
# gpu_id=1
# exp_id='GlobalCoord_Early'
# epoch=('60')
# feat_mode='early'
# warp_mode='HW'
# coord='Global'
# depth_mode='Unique'

# Inter
# gpu_id=1
# exp_id='GlobalCoord_Inter_Repeat'
# epoch=('100')
# feat_mode='inter'
# warp_mode='HW'
# coord='Global'
# depth_mode='Unique'

# LW
# gpu_id=1
# exp_id='GlobalCoord_Inter_LW'
# epoch=('50')
# feat_mode='inter'
# warp_mode='LW'
# coord='Global'
# depth_mode='Unique'

# RLW
# gpu_id=1
# exp_id='GlobalCoord_Inter_RLW_Repeat'
# epoch=('best')
# feat_mode='inter'
# warp_mode='RLW'
# coord='Global'
# depth_mode='Unique'

# DW
# gpu_id=5
# exp_id='GlobalCoord_Inter_DW'
# # epoch=('110')
# epoch=('best')
# feat_mode='inter'
# warp_mode='DW'
# coord='Global'
# depth_mode='Unique'

# DADW
# gpu_id=1
# exp_id='GlobalCoord_Inter_DADW_Revised_Finetune'
# epoch=('44')
# # epoch=('best')
# feat_mode='inter'
# warp_mode='DADW'
# coord='Global'
# depth_mode='Unique'


# HW + Weighted
# gpu_id=7
# exp_id='GlobalCoord_Inter_HW_WeightedDepth_RevisedZ'
# epoch=('120')
# feat_mode='inter'
# warp_mode='HW'
# coord='Global'
# depth_mode='Weighted'

# DADW + Weighted + NoZsupervision
# gpu_id=4
# exp_id='GlobalCoord_Inter_DADW_WeightedDepth_Revised_NoZsupervision'
# # epoch=('70')
# epoch=('120')
# feat_mode='inter'
# warp_mode='DADW'
# coord='Global'
# depth_mode='Weighted'

# DADW + Weighted
# gpu_id=5
# # exp_id='GlobalCoord_Inter_DADW_WeightedDepth_Revised'
# # epoch=('30')
# exp_id='GlobalCoord_Inter_DADW_WeightedDepth_RevisedZ_Repeat'
# epoch=('70')
# feat_mode='inter'
# warp_mode='DADW'
# coord='Global'
# depth_mode='Weighted'


# Joint
gpu_id=4
exp_id='JointCoord_Inter_DADW_WeightedDepth'
epoch=('60')
feat_mode='inter'
warp_mode='DADW'
coord='Joint'
depth_mode='Weighted'

# Joint
# gpu_id=7
# exp_id='JointCoord_Inter_DADW_WeightedDepth_Repeat'
# epoch=('70')
# feat_mode='inter'
# warp_mode='DADW'
# coord='Joint'
# depth_mode='Weighted'

# Local
# gpu_id=7
# exp_id='LocalCoord_repeat'
# # epoch=('40')
# epoch=('60')
# coord='Local'



for i in ${epoch[*]}
do
    echo $i

    CUDA_VISIBLE_DEVICES=$gpu_id python multiagent_test.py multiagent_det \
        --exp_id $exp_id \
        --load_model '../exp/multiagent_det/'$exp_id'/model_'$i'.pth' \
        --gpus $gpu_id --coord=$coord  --trans_layer -2 \
        --message_mode=NO_MESSAGE --uav_height=40 \
        --feat_mode=$feat_mode --polygon \
        --map_scale=1.0 --warp_mode=$warp_mode --depth_mode=$depth_mode \
        --input_dir='/GPFS/data/yhu/Dataset/airsim_camera/airsim_camera_seg_15'
    
    # CUDA_VISIBLE_DEVICES=$gpu_id python multiagent_test.py multiagent_det \
    #     --exp_id $exp_id \
    #     --load_model '../exp/multiagent_det/'$exp_id'/model_'$i'.pth' \
    #     --gpus $gpu_id --coord=$coord  --trans_layer -2 \
    #     --message_mode=NO_MESSAGE --uav_height=40 \
    #     --feat_mode=$feat_mode --polygon \
    #     --map_scale=1.0 --warp_mode=$warp_mode --depth_mode=$depth_mode \
    #     --input_dir='/GPFS/data/shfang/dataset/airsim_camera/airsim_camera_seg_town6_v2'
    
    # CUDA_VISIBLE_DEVICES=$gpu_id python multiagent_test.py multiagent_det \
    #     --exp_id $exp_id \
    #     --load_model '../exp/multiagent_det/'$exp_id'/model_'$i'.pth' \
    #     --gpus $gpu_id --coord=$coord  --trans_layer -2 \
    #     --message_mode=NO_MESSAGE --uav_height=40 \
    #     --feat_mode=$feat_mode --polygon \
    #     --map_scale=1.0 --warp_mode=$warp_mode --depth_mode=$depth_mode \
    #     --input_dir='/GPFS/data/shfang/dataset/airsim_camera/airsim_camera_seg_town4_v2_40m/'
    
    CUDA_VISIBLE_DEVICES=$gpu_id python multiagent_test.py multiagent_det \
        --exp_id $exp_id \
        --load_model '../exp/multiagent_det/'$exp_id'/model_'$i'.pth' \
        --gpus $gpu_id --coord=$coord  --trans_layer -2 \
        --message_mode=NO_MESSAGE --uav_height=40 \
        --feat_mode=$feat_mode --polygon \
        --map_scale=1.0 --warp_mode=$warp_mode --depth_mode=$depth_mode
    
    # CUDA_VISIBLE_DEVICES=$gpu_id python multiagent_test.py multiagent_det \
    #     --exp_id $exp_id \
    #     --load_model '../exp/multiagent_det/'$exp_id'/model_'$i'.pth' \
    #     --gpus $gpu_id --coord=$coord  --trans_layer -2 \
    #     --message_mode=NO_MESSAGE --uav_height=40 \
    #     --input_dir='/GPFS/data/shfang/dataset/airsim_camera/airsim_camera_seg_town6_v2'
    #     # --input_dir='/GPFS/data/shfang/dataset/airsim_camera/airsim_camera_seg_town4_v2_40m/'
    #     # --input_dir='/GPFS/data/yhu/Dataset/airsim_camera/airsim_camera_seg_15'
done
