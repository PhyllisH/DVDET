CUDA_VISIBLE_DEVICES=0,2,3,4 python main.py multiagent_det \
    --exp_id CornerLoss_GlobalCoord_Early  \
    --batch_size=40 --master_batch=10 --num_agents=1 --lr=5e-4 \
    --gpus 0,2,3,4 --trans_layer -2 --num_epochs 200 --coord=Global \
    --message_mode=NO_MESSAGE --uav_height=40 \
    --feat_mode=early  --polygon --warp_mode=HW

CUDA_VISIBLE_DEVICES=1,5,6,7 python main.py multiagent_det \
    --exp_id CornerLoss_GlobalCoord_Inter  \
    --batch_size=60 --master_batch=12 --num_agents=1 --lr=5e-4 \
    --gpus 1,5,6,7 --trans_layer -2 --num_epochs 200 --coord=Global \
    --message_mode=NO_MESSAGE --uav_height=40 \
    --feat_mode=inter --polygon --warp_mode=HW

CUDA_VISIBLE_DEVICES=0,1,2,7 python main.py multiagent_det \
    --exp_id CornerLoss_GlobalCoord_Inter_DADW  \
    --batch_size=36 --master_batch=12 --num_agents=1 --lr=5e-4 \
    --gpus 0,1,2,6 --gpu_chunk_size 12,12,8,4  --trans_layer -2 --num_epochs 200 --coord=Global \
    --message_mode=NO_MESSAGE --uav_height=40 \
    --feat_mode=inter --polygon --warp_mode=DADW 

CUDA_VISIBLE_DEVICES=3,4,5,6 python main.py multiagent_det \
    --exp_id CornerLoss_GlobalCoord_Inter_DADW_WeightedDepth  \
    --batch_size=38 --master_batch=10 --num_agents=1 --lr=5e-4 \
    --gpus 3,4,5,6 --gpu_chunk_size 10,12,12,4 --trans_layer -2 --num_epochs 200 --coord=Global \
    --message_mode=NO_MESSAGE --uav_height=40 \
    --feat_mode=inter --polygon --warp_mode=DADW \
    --depth_mode=Weighted 