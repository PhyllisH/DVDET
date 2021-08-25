gpu_id=$1

CUDA_VISIBLE_DEVICES=$gpu_id python multiagent_test.py multiagent_det \
    --exp_id dla_multiagent_withwarp_GlobalCoord_GTFeatMap_800_450_Down4_BEVGT_40m_Baseline_Town5 \
    --load_model ../exp/multiagent_det/dla_multiagent_withwarp_GlobalCoord_GTFeatMap_800_450_Down4_BEVGT_40m_Baseline_Town5/model_last.pth \
    --gpus $gpu_id --coord=Global  --trans_layer -2 --down_ratio=1 \
    --message_mode=Pointwise --uav_height=40

CUDA_VISIBLE_DEVICES=$gpu_id python multiagent_test.py multiagent_det \
    --exp_id dla_multiagent_withwarp_GlobalCoord_GTFeatMap_800_450_Down4_BEVGT_40m_V2V_Featmap16_RandomPickView_Town5 \
    --load_model ../exp/multiagent_det/dla_multiagent_withwarp_GlobalCoord_GTFeatMap_800_450_Down4_BEVGT_40m_V2V_Featmap16_RandomPickView_Town5/model_last.pth \
    --gpus $gpu_id --coord=Global  --trans_layer 2 --down_ratio=1 \
    --message_mode=V2V --uav_height=40 

CUDA_VISIBLE_DEVICES=$gpu_id python multiagent_test.py multiagent_det \
    --exp_id dla_multiagent_withwarp_GlobalCoord_GTFeatMap_800_450_Down4_BEVGT_40m_Pointwise_Featmap16_RandomPickView_Town5 \
    --load_model ../exp/multiagent_det/dla_multiagent_withwarp_GlobalCoord_GTFeatMap_800_450_Down4_BEVGT_40m_Pointwise_Featmap16_RandomPickView_Town5/model_last.pth \
    --gpus $gpu_id --coord=Global  --trans_layer 2 --down_ratio=1 \
    --message_mode=Pointwise --uav_height=40 

# CUDA_VISIBLE_DEVICES=$gpu_id python multiagent_test.py multiagent_det \
#     --exp_id dla_multiagent_withwarp_GlobalCoord_GTFeatMap_800_450_Down4_BEVGT_40m_When2com_Featmap4_RandomPickView_Town5 \
#     --load_model ../exp/multiagent_det/dla_multiagent_withwarp_GlobalCoord_GTFeatMap_800_450_Down4_BEVGT_40m_When2com_Featmap4_RandomPickView_Town5/model_last.pth \
#     --gpus $gpu_id --coord=Global  --trans_layer 2 --down_ratio=1 \
#     --message_mode=When2com --uav_height=40 

# CUDA_VISIBLE_DEVICES=$gpu_id python multiagent_test.py multiagent_det \
#     --exp_id dla_multiagent_withwarp_GlobalCoord_GTFeatMap_800_450_Down4_BEVGT_40m_Mean_Featmap4_RandomPickView_Town5 \
#     --load_model ../exp/multiagent_det/dla_multiagent_withwarp_GlobalCoord_GTFeatMap_800_450_Down4_BEVGT_40m_Mean_Featmap4_RandomPickView_Town5/model_last.pth \
#     --gpus $gpu_id --coord=Global  --trans_layer 2 --down_ratio=1 \
#     --message_mode=Mean --uav_height=40 

# CUDA_VISIBLE_DEVICES=$gpu_id python multiagent_test.py multiagent_det \
#     --exp_id dla_multiagent_withwarp_GlobalCoord_GTFeatMap_800_450_Down4_BEVGT_40m_Max_Featmap4_RandomPickView_Town5 \
#     --load_model ../exp/multiagent_det/dla_multiagent_withwarp_GlobalCoord_GTFeatMap_800_450_Down4_BEVGT_40m_Max_Featmap4_RandomPickView_Town5/model_last.pth \
#     --gpus $gpu_id --coord=Global  --trans_layer 2 --down_ratio=1 \
#     --message_mode=Max --uav_height=40 