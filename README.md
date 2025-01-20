# DVDET

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2208.03974)

This repository contains the official PyTorch implementation of

[**Aerial Monocular 3D Object Detection</a>**](https://arxiv.org/abs/2208.03974)
<br>
<a href="https://scholar.google.com/citations?user=XBbwb78AAAAJ&hl=zh-CN"> Yue Hu, <a href="https://github.com/dongfeng12"> Shaoheng Fang, <a href="https://weidixie.github.io/"> Weidi Xie, <a href="https://mediabrain.sjtu.edu.cn/sihengc/">Siheng Chen</a> 
<br>
Presented at [RAL 2022]

[**Dataset**] can be downloaded at [https://pan.baidu.com/s/1ZT9z4B5hvwJVFqwdEftkPQ?pwd=pdh3#list/path=%2F](https://pan.baidu.com/s/1ZT9z4B5hvwJVFqwdEftkPQ?pwd=pdh3#list/path=%2F)

# Details

### Args
> exp_id: the path to save the models and logs

> batch_size: the overall batch size

> master_batch: the batch size of the master gpu (which maybe slightly smaller than the average batch size)

> num_agents: the agent amount of a single sample

> lr: the learning rate

> gpus: the visible gpus; 0,1,2,3

> num_epochs: the overall epoches

> message_mode: NO_MESSAGE; this arg may be used in the collaborative setting

> uav_height: the altitude of the drone used to colllect dataset, used to chose dataset; could be 40/60/80

> map_scale: the default value is 1.0, which means the resolution of the BEV feature map is 0.25m/pixel

> trans_layer: the layer of the feature map where collaboration operated; -2 means no collaboration

> coord: the coordinate of object detection; Global means the BEV coordinate, Local means the image/camera coordinate; Joint means both BEV and Image coordinate

> warp_mode: the method to transform feature/image to the BEV coordinate; HW means hard warping which transforms based on the projection matrix; DW means deformable warping which tranforms based on the residual of the projection matrix and the learnable deformable offsets; DADW means a distance-aware deformable warping which considers the geometric prior: the coordinates of the pixels; since the offset of the near pixels should be smaller than the far-away ones

> depth_mode: the BEV feature map could be generated considering all the possible altitudes or only the ground plane whose altitude equals to zerop; the corresponding modes are Weighted and Unique

> polygon: the object representation could be rotated rectangle or axis-aligned rectangle

> real: if set true, the model would be trained on the real dataset, otherwise the virtual dataset

### Train
~~~
CUDA_VISIBLE_DEVICES=GPU_ID python main.py multiagent_det --exp_id EXP_DIR --batch_size=BATCH_SIZE --master_batch=MASTER_BATCH_SIZE --num_agents=NUM_AGENTS --lr=LR --gpus GPU_ID --num_epochs EPOCHS --message_mode=NO_MESSAGE --uav_height=40 --map_scale=1.0 --trans_layer -2 --coord=Global/Local/Joint --feat_mode=FEAT_MODE --warp_mode=DW/HW/DADW --depth_mode=Weighted/Unique --polygon
~~~

### Inference
~~~
CUDA_VISIBLE_DEVICES=GPU_ID python multiagent_test.py multiagent_det --exp_id EXP_DIR --load_model MODEL_DIR --gpus GPU_ID  --message_mode=NO_MESSAGE --uav_height=40 --map_scale=1.0 --trans_layer -2 --coord=Global/Local/Joint --feat_mode=FEAT_MODE  --warp_mode=DW/HW/DADW --depth_mode=Weighted/Unique --polygon
~~~


