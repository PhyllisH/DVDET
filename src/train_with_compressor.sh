gpu_id=$1
exp_id=$2
message_mode=$3
trans_layer=$4
batch_size=$5
master_batch=$6
detector_warmup=$7
compressor_warmup=$8

# CUDA_VISIBLE_DEVICES=$gpu_id python main.py multiagent_det \
#    --exp_id=$exp_id --batch_size=$batch_size --master_batch=$master_batch \
#    --lr=5e-4 --gpus $gpu_id --trans_layer=$trans_layer\
#    --coord=Global  --map_scale=1.0 --num_agents=5 \
#    --warp_mode=HW --depth_mode=Unique --message_mode=$message_mode \
#    --polygon --feat_mode=inter --with_occluded --train_mode detector \
#    --num_epochs=$detector_warmup

CUDA_VISIBLE_DEVICES=$gpu_id python main.py multiagent_det \
    --exp_id=$exp_id --batch_size=$batch_size --master_batch=$master_batch \
    --lr=5e-4 --gpus $gpu_id --trans_layer=$trans_layer\
    --coord=Global  --map_scale=1.0 --num_agents=5 \
    --warp_mode=HW --depth_mode=Unique --message_mode=$message_mode \
    --polygon --feat_mode=inter --with_occluded --train_mode compressor \
    --num_epochs=$compressor_warmup --load_model '../exp/multiagent_det/'$exp_id'_Detector/model_100.pth'

CUDA_VISIBLE_DEVICES=$gpu_id python main.py multiagent_det \
    --exp_id=$exp_id --batch_size=$batch_size --master_batch=$master_batch \
    --lr=5e-4 --gpus $gpu_id --trans_layer=$trans_layer\
    --coord=Global  --map_scale=1.0 --num_agents=5 \
    --warp_mode=HW --depth_mode=Unique --message_mode=$message_mode \
    --polygon --feat_mode=inter --with_occluded --train_mode full \
    --load_model '../exp/multiagent_det/'$exp_id'/model_last.pth'
