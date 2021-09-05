CUDA_VISIBLE_DEVICES=3,4,5,6,7 python main.py multiagent_det \
    --exp_id dla_multiagent_withwarp_GlobalCoord_GTFeatMap_512_512_Down4_BEVGT_40m_V2V_Featmap16_RandomPickView_Town5  \
    --batch_size=5 --master_batch=1 --num_agents=5 --lr=5e-4 \
    --gpus 3,4,5,6,7 --trans_layer 2 --num_epochs 100 --coord=Global \
    --message_mode=V2V --uav_height=40 --down_ratio=1

CUDA_VISIBLE_DEVICES=3,4,5,6,7 python main.py multiagent_det \
    --exp_id dla_multiagent_withwarp_GlobalCoord_GTFeatMap_512_512_Down4_BEVGT_40m_Mean_Featmap4_RandomPickView_Town5  \
    --batch_size=5 --master_batch=1 --num_agents=5 --lr=5e-4 \
    --gpus 3,4,5,6,7 --trans_layer 0 --num_epochs 100 --coord=Global \
    --message_mode=Mean --uav_height=40 --down_ratio=1 

CUDA_VISIBLE_DEVICES=3,4,5,6,7 python main.py multiagent_det \
    --exp_id dla_multiagent_withwarp_GlobalCoord_GTFeatMap_512_512_Down4_BEVGT_40m_Max_Featmap4_RandomPickView_Town5  \
    --batch_size=5 --master_batch=1 --num_agents=5 --lr=5e-4 \
    --gpus 3,4,5,6,7 --trans_layer 0 --num_epochs 100 --coord=Global \
    --message_mode=Max --uav_height=40 --down_ratio=1