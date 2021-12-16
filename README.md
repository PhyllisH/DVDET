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

### Train
~~~
CUDA_VISIBLE_DEVICES=GPU_ID python main.py multiagent_det --exp_id EXP_DIR --batch_size=BATCH_SIZE --master_batch=MASTER_BATCH_SIZE --num_agents=NUM_AGENTS --lr=LR --gpus GPU_ID --num_epochs EPOCHS --message_mode=NO_MESSAGE --uav_height=40 --map_scale=1.0 --trans_layer -2 --coord=Global/Local/Joint --feat_mode=FEAT_MODE --warp_mode=DW/HW/DADW --depth_mode=Weighted/Unique --polygon
~~~

### Inference
~~~
CUDA_VISIBLE_DEVICES=GPU_ID python multiagent_test.py multiagent_det --exp_id EXP_DIR --load_model MODEL_DIR --gpus GPU_ID  --message_mode=NO_MESSAGE --uav_height=40 --map_scale=1.0 --trans_layer -2 --coord=Global/Local/Joint --feat_mode=FEAT_MODE  --warp_mode=DW/HW/DADW --depth_mode=Weighted/Unique --polygon
~~~


## Installation

Please refer to [INSTALL.md](readme/INSTALL.md) for installation instructions.

## Use CenterNet

We support demo for image/ image folder, video, and webcam. 

First, download the models (By default, [ctdet_coco_dla_2x](https://drive.google.com/open?id=1pl_-ael8wERdUREEnaIfqOV_VF2bEVRT) for detection and 
[multi_pose_dla_3x](https://drive.google.com/open?id=1PO1Ax_GDtjiemEmDVD7oPWwqQkUu28PI) for human pose estimation) 
from the [Model zoo](readme/MODEL_ZOO.md) and put them in `CenterNet_ROOT/models/`.

For object detection on images/ video, run:
## Third-party resources

- CenterNet + embedding learning based tracking: [FairMOT](https://github.com/ifzhang/FairMOT) from [Yifu Zhang](https://github.com/ifzhang).
- Detectron2 based implementation: [CenterNet-better](https://github.com/FateScript/CenterNet-better) from [Feng Wang](https://github.com/FateScript).
- Keras Implementation: [keras-centernet](https://github.com/see--/keras-centernet) from [see--](https://github.com/see--) and [keras-CenterNet](https://github.com/xuannianz/keras-CenterNet) from [xuannianz](https://github.com/xuannianz).
- MXnet implementation: [mxnet-centernet](https://github.com/Guanghan/mxnet-centernet) from [Guanghan Ning](https://github.com/Guanghan).
- Stronger human open estimation models: [centerpose](https://github.com/tensorboy/centerpose) from [tensorboy](https://github.com/tensorboy).
- TensorRT extension with ONNX models: [TensorRT-CenterNet](https://github.com/CaoWGG/TensorRT-CenterNet) from [Wengang Cao](https://github.com/CaoWGG).
- CenterNet + DeepSORT tracking implementation: [centerNet-deep-sort](https://github.com/kimyoon-young/centerNet-deep-sort) from [kimyoon-young](https://github.com/kimyoon-young).
- Blogs on training CenterNet on custom datasets (in Chinese): [ships](https://blog.csdn.net/weixin_42634342/article/details/97756458) from [Rhett Chen](https://blog.csdn.net/weixin_42634342) and [faces](https://blog.csdn.net/weixin_41765699/article/details/100118353) from [linbior](https://me.csdn.net/weixin_41765699).


## License

CenterNet itself is released under the MIT License (refer to the LICENSE file for details).
Portions of the code are borrowed from [human-pose-estimation.pytorch](https://github.com/Microsoft/human-pose-estimation.pytorch) (image transform, resnet), [CornerNet](https://github.com/princeton-vl/CornerNet) (hourglassnet, loss functions), [dla](https://github.com/ucbdrive/dla) (DLA network), [DCNv2](https://github.com/CharlesShang/DCNv2)(deformable convolutions), [tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn)(Pascal VOC evaluation) and [kitti_eval](https://github.com/prclibo/kitti_eval) (KITTI dataset evaluation). Please refer to the original License of these projects (See [NOTICE](NOTICE)).

## Citation

If you find this project useful for your research, please use the following BibTeX entry.

    @inproceedings{zhou2019objects,
      title={Objects as Points},
      author={Zhou, Xingyi and Wang, Dequan and Kr{\"a}henb{\"u}hl, Philipp},
      booktitle={arXiv preprint arXiv:1904.07850},
      year={2019}
    }
