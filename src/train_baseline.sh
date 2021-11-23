# CUDA_VISIBLE_DEVICES=1,2,5,6,7 python main.py multiagent_det \
#     --exp_id LocalCoord_repeat  \
#     --batch_size=96 --master_batch=16 --num_agents=1 --lr=5e-4 \
#     --gpus 1,2,5,6,7 --gpu_chunk_size 16,24,12,24,20 --trans_layer -2 --num_epochs 200 --coord=Local \
#     --message_mode=NO_MESSAGE --uav_height=40 \
#     --resume --load_model ../exp/multiagent_det/LocalCoord_repeat/model_last.pth

# CUDA_VISIBLE_DEVICES=1,0,2,3 python main.py multiagent_det \
#     --exp_id GlobalCoord_Early  \
#     --batch_size=38 --master_batch=8 --num_agents=1 --lr=5e-4 \
#     --gpus 1,0,2,3 --trans_layer -2 --num_epochs 200 --coord=Global \
#     --message_mode=NO_MESSAGE --uav_height=40 \
#     --feat_mode=early --polygon --warp_mode=HW \
#     --resume --load_model ../exp/multiagent_det/GlobalCoord_Early/model_last.pth

# CUDA_VISIBLE_DEVICES=1,5,6,7 python main.py multiagent_det \
#     --exp_id GlobalCoord_Inter  \
#     --batch_size=56 --master_batch=8 --num_agents=1 --lr=5e-4 \
#     --gpus 1,5,6,7 --gpu_chunk_size 8,16,16,16 --trans_layer -2 --num_epochs 200 --coord=Global \
#     --message_mode=NO_MESSAGE --uav_height=40 \
#     --feat_mode=inter --polygon --warp_mode=HW \
#     --resume --load_model ../exp/multiagent_det/GlobalCoord_Inter/model_last.pth

# CUDA_VISIBLE_DEVICES=0,2,3,4 python main.py multiagent_det \
#     --exp_id GlobalCoord_Inter_LW  \
#     --batch_size=44 --master_batch=8 --num_agents=1 --lr=5e-4 \
#     --gpus 0,2,3,4 --gpu_chunk_size 8,12,12,12 --trans_layer -2 --num_epochs 200 --coord=Global \
#     --message_mode=NO_MESSAGE --uav_height=40 \
#     --feat_mode=inter --polygon --warp_mode=LW \
#     --resume --load_model ../exp/multiagent_det/GlobalCoord_Inter_LW/model_last.pth

# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py multiagent_det \
#     --exp_id GlobalCoord_Inter_RLW  \
#     --batch_size=54 --master_batch=12 --num_agents=1 --lr=5e-4 \
#     --gpus 0,1,2,3 --trans_layer -2 --num_epochs 200 --coord=Global \
#     --message_mode=NO_MESSAGE --uav_height=40 \
#     --feat_mode=inter --polygon --warp_mode=RLW 

# CUDA_VISIBLE_DEVICES=1,5,6,7 python main.py multiagent_det \
#     --exp_id GlobalCoord_Inter_DW  \
#     --batch_size=44 --master_batch=8 --num_agents=1 --lr=5e-4 \
#     --gpus 1,5,6,7 --trans_layer -2 --num_epochs 200 --coord=Global \
#     --message_mode=NO_MESSAGE --uav_height=40 \
#     --feat_mode=inter --polygon --warp_mode=DW \
#     --resume --load_model ../exp/multiagent_det/GlobalCoord_Inter_DW/model_last.pth


# CUDA_VISIBLE_DEVICES=0,2,3,4 python main.py multiagent_det \
#     --exp_id GlobalCoord_Inter_DADW  \
#     --batch_size=44 --master_batch=8 --num_agents=1 --lr=5e-4 \
#     --gpus 0,2,3,4 --gpu_chunk_size 8,12,12,12  --trans_layer -2 --num_epochs 200 --coord=Global \
#     --message_mode=NO_MESSAGE --uav_height=40 \
#     --feat_mode=inter --polygon --warp_mode=DADW \
#     --resume --load_model ../exp/multiagent_det/GlobalCoord_Inter_DADW/model_last.pth

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py multiagent_det \
#     --exp_id GlobalCoord_Inter_DADW_WeightedDepth  \
#     --batch_size=64 --master_batch=8 --num_agents=1 --lr=5e-4 \
#     --gpus 0,1,2,3,4,5,6,7  --trans_layer -2 --num_epochs 200 --coord=Global \
#     --message_mode=NO_MESSAGE --uav_height=40 \
#     --feat_mode=inter --polygon --warp_mode=DADW \
#     --depth_mode=Weighted \
#     --resume --load_model ../exp/multiagent_det/GlobalCoord_Inter_DADW_WeightedDepth/model_last.pth

# CUDA_VISIBLE_DEVICES=6,1,4,5,0,7 python main.py multiagent_det \
#     --exp_id JointCoord_Inter_DADW_WeightedDepth  \
#     --batch_size=70 --master_batch=10 --num_agents=1 --lr=5e-4 \
#     --gpus 6,1,4,5,0,7  --trans_layer -2 --num_epochs 200 --coord=Joint \
#     --message_mode=NO_MESSAGE --uav_height=40 \
#     --feat_mode=inter --polygon --warp_mode=DADW \
#     --depth_mode=Weighted 

# CUDA_VISIBLE_DEVICES=4,0,1,2,3,5,6,7 python main.py multiagent_det \
#     --exp_id GlobalCoord_Fused  \
#     --batch_size=43 --master_batch=1 --num_agents=1 --lr=5e-4 \
#     --gpus 4,0,1,2,3,5,6,7 --trans_layer -2 --num_epochs 200 --coord=Global \
#     --message_mode=NO_MESSAGE --uav_height=40 \
#     --feat_mode=fused --polygon --warp_mode=HW \
#     --resume --load_model ../exp/multiagent_det/GlobalCoord_Fused/model_last.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py multiagent_det \
    --exp_id GlobalCoord_Fused_DADW  \
    --batch_size=8 --master_batch=1 --num_agents=1 --lr=5e-4 \
    --gpus 0,1,2,3,4,5,6,7 --trans_layer -2 --num_epochs 200 --coord=Global \
    --message_mode=NO_MESSAGE --uav_height=40 \
    --feat_mode=fused --polygon --warp_mode=DADW \
    --resume --load_model ../exp/multiagent_det/GlobalCoord_Fused_DADW/model_last.pth

# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python main.py multiagent_det \
#     --exp_id GlobalCoord_Fused_LW  \
#     --batch_size=12 --master_batch=1 --num_agents=1 --lr=5e-4 \
#     --gpus 1,2,3,4,5,6,7 --gpu_chunk_size 1,2,2,2,1,2,2 --trans_layer -2 --num_epochs 200 --coord=Global \
#     --message_mode=NO_MESSAGE --uav_height=40 \
#     --feat_mode=fused --polygon --warp_mode=LW 

# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python main.py multiagent_det \
#     --exp_id GlobalCoord_Fused_RLW  \
#     --batch_size=13 --master_batch=1 --num_agents=1 --lr=5e-4 \
#     --gpus 1,2,3,4,5,6,7 --gpu_chunk_size 1,2,2,2,2,2,2 --trans_layer -2 --num_epochs 200 --coord=Global \
#     --message_mode=NO_MESSAGE --uav_height=40 \
#     --feat_mode=fused --polygon --warp_mode=RLW 

# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python main.py multiagent_det \
#     --exp_id GlobalCoord_Fused_DW  \
#     --batch_size=7 --master_batch=1 --num_agents=1 --lr=5e-4 \
#     --gpus 1,2,3,4,5,6,7 --trans_layer -2 --num_epochs 200 --coord=Global \
#     --message_mode=NO_MESSAGE --uav_height=40 \
#     --feat_mode=fused --polygon --warp_mode=DW

# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python main.py multiagent_det \
#     --exp_id GlobalCoord_Fused_DADW  \
#     --batch_size=7 --master_batch=1 --num_agents=1 --lr=5e-4 \
#     --gpus 1,2,3,4,5,6,7 --gpu_chunk_size 1,1,1,1,1,1,1 --trans_layer -2 --num_epochs 200 --coord=Global \
#     --message_mode=NO_MESSAGE --uav_height=40 \
#     --feat_mode=fused --polygon --warp_mode=DADW

CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py multiagent_det \
    --exp_id GlobalCoord_Fused_DADW_WeightedDepth  \
    --batch_size=4 --master_batch=1 --num_agents=1 --lr=5e-4 \
    --gpus 4,5,6,7  --trans_layer -2 --num_epochs 200 --coord=Global \
    --message_mode=NO_MESSAGE --uav_height=40 \
    --feat_mode=fused --polygon --warp_mode=DADW \
    --depth_mode=Weighted 