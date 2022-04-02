gpu_id=$1

input_dir=('/DATA7_DB7/data/shfang/airsim_camera_seg_15')
# \
#              '/GPFS/data/shfang/dataset/airsim_camera/airsim_camera_seg_town6_v2'\
#               '/GPFS/data/shfang/dataset/airsim_camera/airsim_camera_seg_town4_v2_80m' \
            #   '/GPFS/data/shfang/dataset/airsim_camera/airsim_camera_seg_town4_v2_40m')
exp_id='dla_multiagent_withwarp_GlobalCoord_GTFeatMap_800_450_Down4_BEVGT_40m_Max_Featmap4_RandomPickView_Town5'

for i in ${input_dir[*]}
do
    echo $i
    CUDA_VISIBLE_DEVICES=$gpu_id python multiagent_test.py multiagent_det \
        --exp_id $exp_id \
        --load_model '../exp/multiagent_det/'$exp_id'/model_best.pth' \
        --gpus $gpu_id --coord=Global  --trans_layer 0 --down_ratio=1 \
        --message_mode=Max --uav_height=40 \
        --input_dir $i
done
