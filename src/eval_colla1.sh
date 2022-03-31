
# Official Version
gpu_id=$1
message_mode=$2
exp_id=$3
trans_layer=$4
round=$5
# feat_mode=$4

epoch=('20' '40' '60' '80' '90' 'last' 'best')
# epoch=('20' '40' '60' '80' '90' '100' '120' '140' '160' '180' '200' 'last' 'best')
# epoch=('10' '20' '30' '40' '50' '60' '70' '80' '90' '100' '110' '120' '130' '140' 'last' 'best')
# epoch=('best' 'last')


# exp_id='NO_MESSAGE'


feat_mode='inter'
# feat_mode='early'

# warp_mode='DW'
# warp_mode='DADW'
warp_mode='HW'
# warp_mode='RLW'
# warp_mode='LW'

# coord='Joint'
coord='Global'
# coord='Local'

depth_mode='Unique'
# depth_mode='Weighted'

# message_mode='Pointwise'

# trans_layer='-2'
# trans_layer='0'
# trans_layer='2'


for i in ${epoch[*]}
do
    echo $i
    CUDA_VISIBLE_DEVICES=$gpu_id python multiagent_test.py multiagent_det \
        --exp_id $exp_id \
        --load_model '../exp/multiagent_det/'$exp_id'/model_'$i'.pth' \
        --gpus $gpu_id --coord=$coord --uav_height=40 \
        --feat_mode=$feat_mode --polygon \
        --map_scale=1.0 --warp_mode=$warp_mode --depth_mode=$depth_mode \
        --message_mode=$message_mode --trans_layer=$trans_layer --round=$round --with_occluded
    
    CUDA_VISIBLE_DEVICES=$gpu_id python multiagent_test.py multiagent_det \
        --exp_id $exp_id \
        --load_model '../exp/multiagent_det/'$exp_id'/model_'$i'.pth' \
        --gpus $gpu_id --coord=$coord --uav_height=40 \
        --feat_mode=$feat_mode --polygon \
        --map_scale=1.0 --warp_mode=$warp_mode --depth_mode=$depth_mode \
        --message_mode=$message_mode --trans_layer=$trans_layer --round=$round
done