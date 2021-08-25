# CUDA_VISIBLE_DEVICES=7,6,3 python main.py multiagent_det \
#     --exp_id dla_multiagent_withwarp_GlobalCoord_GTFeatMap_800_450_Down4_BEVGT_40m_Baseline  \
#     --batch_size=3 --master_batch=1 --num_agents=5 --lr=5e-4 \
#     --gpus 7,6,3 --trans_layer -2 --num_epochs 100 --coord=Global \
#     --message_mode=Pointwise --uav_height=40 --down_ratio=1 

# CUDA_VISIBLE_DEVICES=7,6,3 python main.py multiagent_det \
#     --exp_id dla_multiagent_withwarp_GlobalCoord_GTFeatMap_800_450_Down4_BEVGT_40m_Mean_Featmap4_RandomPickView  \
#     --batch_size=3 --master_batch=1 --num_agents=5 --lr=5e-4 \
#     --gpus 7,6,3 --trans_layer 0 --num_epochs 100 --coord=Global \
#     --message_mode=Mean --uav_height=40 --down_ratio=1 

# CUDA_VISIBLE_DEVICES=7,6,3 python main.py multiagent_det \
#     --exp_id dla_multiagent_withwarp_GlobalCoord_GTFeatMap_800_450_Down4_BEVGT_40m_Max_Featmap4_RandomPickView  \
#     --batch_size=3 --master_batch=1 --num_agents=5 --lr=5e-4 \
#     --gpus 7,6,3 --trans_layer 0 --num_epochs 100 --coord=Global \
#     --message_mode=Max --uav_height=40 --down_ratio=1 

# CUDA_VISIBLE_DEVICES=7,6,3,2 python main.py multiagent_det \
#     --exp_id dla_multiagent_withwarp_GlobalCoord_GTFeatMap_800_450_Down4_BEVGT_40m_When2com_Featmap4_RandomPickView  \
#     --batch_size=4 --master_batch=1 --num_agents=5 --lr=5e-4 \
#     --gpus 7,6,3,2 --trans_layer 0 --num_epochs 100 --coord=Global \
#     --message_mode=When2com --uav_height=40 --down_ratio=1 

# CUDA_VISIBLE_DEVICES=7,6,3,2 python main.py multiagent_det \
#     --exp_id dla_multiagent_withwarp_GlobalCoord_GTFeatMap_800_450_Down4_BEVGT_40m_V2V_Featmap4_RandomPickView  \
#     --batch_size=4 --master_batch=1 --num_agents=5 --lr=5e-4 \
#     --gpus 7,6,3,2 --trans_layer 0 --num_epochs 100 --coord=Global \
#     --message_mode=V2V --uav_height=40 --down_ratio=1 

# CUDA_VISIBLE_DEVICES=7,6,3,2 python main.py multiagent_det \
#     --exp_id dla_multiagent_withwarp_GlobalCoord_GTFeatMap_800_450_Down4_BEVGT_40m_Pointwise_Featmap4_RandomPickView  \
#     --batch_size=4 --master_batch=1 --num_agents=5 --lr=5e-4 \
#     --gpus 7,6,3,2 --trans_layer 0 --num_epochs 100 --coord=Global \
#     --message_mode=Pointwise --uav_height=40 --down_ratio=1 

# CUDA_VISIBLE_DEVICES=7,6,3,2 python main.py multiagent_det \
#     --exp_id dla_multiagent_withwarp_GlobalCoord_GTFeatMap_800_450_Down4_BEVGT_40m_Baseline  \
#     --batch_size=4 --master_batch=1 --num_agents=5 --lr=5e-4 \
#     --gpus 7,6,3,2 --trans_layer -2 --num_epochs 100 --coord=Global \
#     --message_mode=Pointwise --uav_height=40 --down_ratio=1 

CUDA_VISIBLE_DEVICES=7,6,3 python main.py multiagent_det \
    --exp_id dla_multiagent_withwarp_GlobalCoord_GTFeatMap_800_450_Down4_BEVGT_40m_Mean_Featmap4_RandomPickView  \
    --batch_size=3 --master_batch=1 --num_agents=5 --lr=5e-4 \
    --gpus 7,6,3 --trans_layer 0 --num_epochs 50 --coord=Global \
    --message_mode=Mean --uav_height=40 --down_ratio=1 \
    --resume \
    --load_model ../exp/multiagent_det/dla_multiagent_withwarp_GlobalCoord_GTFeatMap_800_450_Down4_BEVGT_40m_Mean_Featmap4_RandomPickView/model_last.pth 

CUDA_VISIBLE_DEVICES=7,6,3 python main.py multiagent_det \
    --exp_id dla_multiagent_withwarp_GlobalCoord_GTFeatMap_800_450_Down4_BEVGT_40m_Max_Featmap4_RandomPickView  \
    --batch_size=3 --master_batch=1 --num_agents=5 --lr=5e-4 \
    --gpus 7,6,3 --trans_layer 0 --num_epochs 50 --coord=Global \
    --message_mode=Max --uav_height=40 --down_ratio=1 \
    --resume \
    --load_model ../exp/multiagent_det/dla_multiagent_withwarp_GlobalCoord_GTFeatMap_800_450_Down4_BEVGT_40m_Max_Featmap4_RandomPickView/model_last.pth
