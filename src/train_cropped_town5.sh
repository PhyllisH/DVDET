# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py multiagent_det \
#     --exp_id dla_multiagent_withwarp_GlobalCoord_GTFeatMap_352_192_JointUAVBEV_40m_Town5_Baseline_MapScale1_B23  \
#     --batch_size=23 --master_batch=2 --num_agents=5 --lr=5e-4 \
#     --gpus 0,1,2,3,4,5,6,7 --trans_layer -2 --num_epochs 100 --coord=Joint \
#     --message_mode=Pointwise --uav_height=40 --map_scale=1.0

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py multiagent_det \
    --exp_id dla_multiagent_withwarp_GlobalCoord_GTFeatMap_352_192_BEVGT_40m_Town5_Baseline_MapScale1_B12  \
    --batch_size=60 --master_batch=15 --num_agents=5 --lr=5e-4 \
    --gpus 0,1,2,3 --trans_layer -2 --num_epochs 100 --coord=Global \
    --message_mode=NO_MESSAGE --uav_height=40 --map_scale=1.0

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py multiagent_det \
    --exp_id dla_multiagent_withwarp_GlobalCoord_GTFeatMap_352_192_BEVGT_40m_Town5_Baseline_MapScale1_B12  \
    --batch_size=60 --master_batch=15 --num_agents=5 --lr=5e-4 \
    --gpus 0,1,2,3 --trans_layer -2 --num_epochs 100 --coord=Global \
    --message_mode=NO_MESSAGE --uav_height=40 --map_scale=1.0

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py multiagent_det \
    --exp_id dla_multiagent_withwarp_GlobalCoord_GTFeatMap_352_192_BEVGT_40m_Town5_Baseline_MapScale1_B12  \
    --batch_size=60 --master_batch=15 --num_agents=5 --lr=5e-4 \
    --gpus 0,1,2,3 --trans_layer -2 --num_epochs 100 --coord=Global \
    --message_mode=NO_MESSAGE --uav_height=40 --map_scale=1.0


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py multiagent_det \
#     --exp_id dla_multiagent_withwarp_GlobalCoord_GTFeatMap_352_192_UAVGT_40m_Town5_Baseline_MapScale1_B23  \
#     --batch_size=23 --master_batch=2 --num_agents=5 --lr=5e-4 \
#     --gpus 0,1,2,3,4,5,6,7 --trans_layer -2 --num_epochs 100 --coord=Local \
#     --message_mode=Pointwise --uav_height=40 --map_scale=1.0

# CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 python main.py multiagent_det \
#     --exp_id dla_multiagent_withwarp_GlobalCoord_GTFeatMap_352_192_Down4_BEVGT_40m_Town5_Baseline_MapScale1_B20  \
#     --batch_size=20 --master_batch=2 --num_agents=5 --lr=5e-4 \
#     --gpus 0,2,3,4,5,6,7 --trans_layer -2 --num_epochs 200 --coord=Global \
#     --message_mode=Pointwise --uav_height=40 --down_ratio=1 --map_scale=1.0

# CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py multiagent_det \
#     --exp_id dla_multiagent_withwarp_GlobalCoord_GTFeatMap_352_192_Down4_BEVGT_40m_Town5_Baseline_MapScale1  \
#     --batch_size=12 --master_batch=3 --num_agents=5 --lr=5e-4 \
#     --gpus 4,5,6,7 --trans_layer -2 --num_epochs 200 --coord=Global \
#     --message_mode=Pointwise --uav_height=40 --down_ratio=1 --map_scale=1.0

# CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py multiagent_det \
#     --exp_id dla_multiagent_withwarp_GlobalCoord_GTFeatMap_352_192_Down4_BEVGT_40m_Town5_Baseline_MapScale2  \
#     --batch_size=4 --master_batch=1 --num_agents=5 --lr=5e-4 \
#     --gpus 4,5,6,7 --trans_layer -2 --num_epochs 200 --coord=Global \
#     --message_mode=Pointwise --uav_height=40 --down_ratio=1 --map_scale=0.5

# CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py multiagent_det \
#     --exp_id dla_multiagent_withwarp_GlobalCoord_GTFeatMap_352_192_Down4_BEVGT_40m_Town5_Pointwise_MapScale2  \
#     --batch_size=4 --master_batch=1 --num_agents=5 --lr=5e-4 \
#     --gpus 4,5,6,7 --trans_layer 2 --num_epochs 200 --coord=Global \
#     --message_mode=Pointwise --uav_height=40 --down_ratio=1 --map_scale=0.5

# CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py multiagent_det \
#     --exp_id dla_multiagent_withwarp_GlobalCoord_GTFeatMap_352_192_Down4_BEVGT_40m_Town5_Max_MapScale2  \
#     --batch_size=4 --master_batch=1 --num_agents=5 --lr=5e-4 \
#     --gpus 4,5,6,7 --trans_layer 2 --num_epochs 200 --coord=Global \
#     --message_mode=Max --uav_height=40 --down_ratio=1 --map_scale=0.5

# CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py multiagent_det \
#     --exp_id dla_multiagent_withwarp_GlobalCoord_GTFeatMap_352_192_Down4_BEVGT_40m_Town5_When2com_MapScale2  \
#     --batch_size=4 --master_batch=1 --num_agents=5 --lr=5e-4 \
#     --gpus 4,5,6,7 --trans_layer 2 --num_epochs 200 --coord=Global \
#     --message_mode=When2com --uav_height=40 --down_ratio=1 --map_scale=0.5

# CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py multiagent_det \
#     --exp_id dla_multiagent_withwarp_GlobalCoord_GTFeatMap_352_192_Down4_BEVGT_40m_Town5_V2V_MapScale2  \
#     --batch_size=4 --master_batch=1 --num_agents=5 --lr=5e-4 \
#     --gpus 4,5,6,7 --trans_layer 2 --num_epochs 200 --coord=Global \
#     --message_mode=V2V --uav_height=40 --down_ratio=1 --map_scale=0.5