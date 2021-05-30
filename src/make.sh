echo -e "\033[?25h"
# CUDA_VISIBLE_DEVICES=0 python main.py multiagent_det --exp_id dla_multiagent_withwarp \
# 	--batch_size 1 --master_batch 1 --lr 5e-4 --gpus 0 --num_workers 16 --num_epochs=200
CUDA_VISIBLE_DEVICES=0,1,3,4,5,6 python main.py multiagent_det --exp_id dla_multiagent_withwarp \
	--batch_size 6 --master_batch 1 --lr 5e-4 --gpus 0,1,3,4,5,6 --num_workers 16 --num_epochs=200