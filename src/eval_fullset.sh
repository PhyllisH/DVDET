# Official Version
gpu_id=6
epoch=('20' '40' '60' '80' '100' '120' '140' '160' '180' '200' 'last' 'best')
# epoch=('10' '20' '30' '40' '50' '60' '70' '80' '90' '100' '110' '120' 'last' 'best')
# epoch=('140' 'last')
# exp_id='LocalCoord_repeat'
# exp_id='GlobalCoord_Early'
# feat_mode='early'
exp_id='GlobalCoord_Inter'
# exp_id='GlobalCoord_Inter_LW'
# exp_id='GlobalCoord_Inter_RLW'
# exp_id='GlobalCoord_Inter_DW'
# exp_id='GlobalCoord_Inter_DADW'
feat_mode='inter'
# warp_mode='DADW'
warp_mode='HW'

for i in ${epoch[*]}
do
    echo $i
    CUDA_VISIBLE_DEVICES=$gpu_id python multiagent_test.py multiagent_det \
        --exp_id $exp_id \
        --load_model '../exp/multiagent_det/'$exp_id'/model_'$i'.pth' \
        --gpus $gpu_id --coord=Global  --trans_layer -2 \
        --message_mode=NO_MESSAGE --uav_height=40 \
        --feat_mode=$feat_mode --polygon \
        --map_scale=1.0 --warp_mode=$warp_mode
    
    CUDA_VISIBLE_DEVICES=$gpu_id python multiagent_test.py multiagent_det \
        --exp_id $exp_id \
        --load_model '../exp/multiagent_det/'$exp_id'/model_'$i'.pth' \
        --gpus $gpu_id --coord=Global  --trans_layer -2 \
        --message_mode=NO_MESSAGE --uav_height=40 \
        --feat_mode=$feat_mode \
        --map_scale=1.0 --warp_mode=$warp_mode
done