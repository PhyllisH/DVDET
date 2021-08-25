CUDA_VISIBLE_DEVICES=6,7 python main.py multiagent_det \
    --exp_id dla_multiagent_withwarp_GlobalCoord_GTFeatMap_800_450_Down4_BEVGT_40m_Baseline_RandomPickView_Updated  \
    --batch_size=2 --master_batch=1 --num_agents=5 --lr=5e-4 \
    --gpus 6,7 --trans_layer -2 --num_epochs 50 --coord=Global \
    --message_mode=Pointwise --uav_height=40 --down_ratio=1 \
    --resume \
    --load_model ../exp/multiagent_det/dla_multiagent_withwarp_GlobalCoord_GTFeatMap_800_450_Down4_BEVGT_40m_Baseline_RandomPickView_Updated/model_last.pth

CUDA_VISIBLE_DEVICES=6,7 python main.py multiagent_det \
    --exp_id dla_multiagent_withwarp_GlobalCoord_GTFeatMap_800_450_Down4_BEVGT_40m_When2com_Featmap4_RandomPickView_Updated  \
    --batch_size=2 --master_batch=1 --num_agents=5 --lr=5e-4 \
    --gpus 6,7 --trans_layer 0 --num_epochs 50 --coord=Global \
    --message_mode=When2com --uav_height=40 --down_ratio=1 