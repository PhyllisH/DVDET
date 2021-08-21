CUDA_VISIBLE_DEVICES=4 python main.py multiagent_det \
    --exp_id dla_multiagent_withwarp_GlobalCoord_Polygon_FeatMap_800_450_Down4_BEVGT_40m_Baseline  \
    --batch_size=1 --master_batch=1 --num_agents=5 --lr=5e-4 \
    --gpus 4 --trans_layer -2 --num_epochs 100 --coord=Global \
    --message_mode=Pointwise --uav_height=40 --down_ratio=1 \
