# Official Version
# gpu_id=5

# epoch=('20' '40' '60' '80' '100' '120' '140' '160' '180' '200' 'last' 'best')
# epoch=('last')
epoch=('10' '20' '30' '40' '50' '60' '70' '80' '90' '100' '110' '120' '130' '140' '150' '160' '170' '180' '190' '200' 'last' 'best')
# epoch=('110' '120' '130' '140' '150' '160' '170' '180' '190' '200' 'last' 'best')
# epoch=('50' '60' '70' '80' '90' '100' '110' '120' '130' '140' '150' '160' '170' '180' '190' '200' 'last' 'best')
# epoch=('140' 'last')


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
# exp_id='RealData_GlobalCoord_Early_CleanedData'
# # epoch=('140')
# feat_mode='early'
# warp_mode='HW'
# coord='Global'
# depth_mode='Unique'

# Inter
# gpu_id=0
# exp_id='RealData_GlobalCoord_Inter_CleanedData'
# # epoch=('110')
# feat_mode='inter'
# warp_mode='HW'
# coord='Global'
# depth_mode='Unique'


# DADW
# gpu_id=4
# exp_id='RealData_GlobalCoord_Inter_DADW_CleanedData'
# # epoch=('110')
# feat_mode='inter'
# warp_mode='HW'
# coord='Global'
# depth_mode='Unique'

# # # Joint 
# gpu_id=5
# exp_id='RealData_JointCoord_Inter_CleanedData'
# # epoch=('110')
# feat_mode='inter'
# warp_mode='HW'
# coord='Joint'
# depth_mode='Unique'

# # Joint 
# gpu_id=5
# exp_id='RealData_JointCoord_Inter_DADW_WeightedDepth_CleanedData_Revised'
# # epoch=('110')
# feat_mode='inter'
# warp_mode='DADW'
# coord='Joint'
# depth_mode='Weighted'


# # DADW + Weighted
gpu_id=4
exp_id='RealData_GlobalCoord_Inter_DADW_WeightedDepth_CleanedData_Revised'
# epoch=('110')
feat_mode='inter'
warp_mode='DADW'
coord='Global'
depth_mode='Weighted'

# # Local
# gpu_id=0
# exp_id='RealData_LocalCoord_CleanedData'
# # epoch=('110')
# feat_mode='inter'
# warp_mode='HW'
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
        --input_h 480 --input_w 736

    # CUDA_VISIBLE_DEVICES=$gpu_id python multiagent_test.py multiagent_det \
    #     --exp_id $exp_id \
    #     --load_model '../exp/multiagent_det/'$exp_id'/model_'$i'.pth' \
    #     --gpus $gpu_id --coord=$coord  --trans_layer -2 \
    #     --message_mode=NO_MESSAGE --uav_height=40 \
    #     --feat_mode=$feat_mode \
    #     --map_scale=1.0 --warp_mode=$warp_mode --depth_mode=$depth_mode \
    #     --input_h 480 --input_w 736

done

