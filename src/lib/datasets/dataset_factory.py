from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ddd import DddDataset
from .sample.exdet import EXDetDataset
from .sample.ctdet import CTDetDataset
from .sample.multi_pose import MultiPoseDataset
from .sample.multiagent_det import MultiAgentDetDataset

from .dataset.coco import COCO
from .dataset.pascal import PascalVOC
from .dataset.kitti import KITTI
from .dataset.coco_hp import COCOHP
from .dataset.airsim_camera import AIRSIMCAM
from .dataset.multiagent_airsim_camera import MULTIAGENTAIRSIMCAM

dataset_factory = {
    'coco': COCO,
    'pascal': PascalVOC,
    'kitti': KITTI,
    'coco_hp': COCOHP,
    'airsim_camera': AIRSIMCAM,
    'multiagent_airsim_camera': MULTIAGENTAIRSIMCAM
}

_sample_factory = {
    'exdet': EXDetDataset,
    'ctdet': CTDetDataset,
    'ddd': DddDataset,
    'multi_pose': MultiPoseDataset,
    'multiagent_det': MultiAgentDetDataset
}


def get_dataset(dataset, task):
    class Dataset(dataset_factory[dataset], _sample_factory[task]):
        pass

    return Dataset
