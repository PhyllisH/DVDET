
# Official Version
gpu_id=$1
message_mode=$2
exp_id=$3
trans_layer=$4
round=$5
epoch=$6
# feat_mode=$4


# comm_thre=(0.0 0.001 0.01 0.03 0.1 0.3 0.9 1.0)
comm_thre=(0.0 0.001 0.01 0.03 0.06 0.08 0.1 0.13 0.16 0.20 0.24 0.28 0.3 0.9 1.0)
# comm_thre=(0.06 0.08 0.1 0.13 0.16 0.20 0.24 0.28)
# comm_thre=(0.13 0.16 0.2 0.24 0.28)
# comm_thre=(0.4 0.5 0.6 0.7 0.8)
# comm_thre=(1.0 0.113694 0.014132 0.003331 0.000981  0.000499 0.000002 0)
# comm_thre=(1.0 0.113694 0.014132 0.003331 0.001488 0.001155 0.000981 0.000830 0.000735 0.000646 0.000579 0.000523 0.000499 0.000397 0.000312 0.000232 0.000002 0)

feat_mode='inter'
# feat_mode='early'

# warp_mode='DW'
# warp_mode='DADW'
# warp_mode='HW'
# warp_mode='RLW'
# warp_mode='LW'

# coord='Joint'
coord='Global'
# coord='Local'

# depth_mode='Unique'
# depth_mode='Weighted'

# message_mode='Pointwise'

# trans_layer='-2'
# trans_layer='0'
# trans_layer='2'


for i in ${comm_thre[*]}
do
    echo $i
    CUDA_VISIBLE_DEVICES=$gpu_id python multiagent_test.py multiagent_det \
        --exp_id $exp_id \
        --load_model '../exp/multiagent_det/'$exp_id'/model_'$epoch'.pth' \
        --gpus $gpu_id --coord=$coord --uav_height=40 \
        --feat_mode=$feat_mode --polygon \
        --map_scale=1.0 --warp_mode=$warp_mode --depth_mode=$depth_mode \
        --message_mode=$message_mode --trans_layer=$trans_layer --round=$round --with_occluded \
        --comm_thre=$i
    
    # CUDA_VISIBLE_DEVICES=$gpu_id python multiagent_test.py multiagent_det \
    #     --exp_id $exp_id \
    #     --load_model '../exp/multiagent_det/'$exp_id'/model_'$epoch'.pth' \
    #     --gpus $gpu_id --coord=$coord --uav_height=40 \
    #     --feat_mode=$feat_mode --polygon \
    #     --map_scale=1.0 --warp_mode=$warp_mode --depth_mode=$depth_mode \
    #     --message_mode=$message_mode --trans_layer=$trans_layer --round=$round \
    #     --comm_thre=$i
done
