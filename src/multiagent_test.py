'''
Author: yhu
Contact: phyllis1sjtu@outlook.com
LastEditTime: 2021-07-22 17:13:26
Description: 
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch._C import dtype

import _init_paths

import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch
import random

from external.nms import soft_nms
from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory
import copy

class PrefetchDataset(torch.utils.data.Dataset):
    def __init__(self, opt, dataset, pre_process_func):
        self.samples = dataset.samples
        self.img_dir = dataset.img_dir
        self.pre_process_func = pre_process_func
        self.opt = opt
    
    # def load_image_func(self, index):
    #     sample_id = index // 25
    #     cam_id = index % 25
    #     images_key = 'image'
    #     images = []
    #     trans_mat_list = []
    #     image_idx = []
    #     cams_list = [x for x in self.samples[sample_id].keys() if not x.startswith('vehicles')]
    #     cam_list = random.sample([x for x in cams_list if not x.startswith(cams_list[cam_id])], self.opt.num_agents-1) + [cams_list[cam_id]]
    #     for cam, info in self.samples[sample_id].items():
    #         if cam.startswith('vehicles'):
    #             continue
    #         # else:
    #         # elif cam.startswith('F'):
    #         # elif cam.endswith(str(cam_id)):
    #         if cam in cam_list:
    #             images.append(cv2.imread(os.path.join(self.img_dir, info[images_key])))
    #             image_idx.append(info['image_id'])
    #             trans_mat_list.append(np.array(info['trans_mat'], dtype=np.float32))
    #     trans_mats = np.concatenate([x[None,:,:] for x in trans_mat_list], axis=0)
    #     return images, image_idx, trans_mats

    def load_image_func(self, index):
        sample_id = 54
        cam_id = 0
        # sample_id = index // 5
        # cam_id = index % 5
        img_dir = self.img_dir[sample_id] if len(self.img_dir) == len(self.samples) else self.img_dir
        images_key = 'image'
        images = []
        trans_mat_list = []
        trans_mats_n010_list = []
        trans_mats_n005_list = []
        trans_mats_p005_list = []
        trans_mats_p007_list = []
        trans_mats_p010_list = []
        trans_mats_p015_list = []
        trans_mats_p020_list = []
        trans_mats_p080_list = []
        shift_mats_1_list = []
        shift_mats_2_list = []
        shift_mats_4_list = []
        shift_mats_8_list = []
        image_idx = []
        cams_list = [x for x in self.samples[sample_id].keys() if not x.startswith('vehicles')]
        # cam_list = random.sample([x for x in cams_list if not x.startswith(cams_list[cam_id])], self.opt.num_agents-1) + [cams_list[cam_id]]
        # sensor = cams_list[cam_id].split('_')[1]
        sensors = ['FRONT', 'BOTTOM', 'LEFT', 'RIGHT', "BACK"]
        sensor = sensors[cam_id]
        cam_list = [x for x in cams_list if sensor in x]
        # print(cam_list)
        
        for cam, info in self.samples[sample_id].items():
            if cam.startswith('vehicles'):
                continue
            # else:
            # elif cam.startswith('F'):
            # elif cam.endswith(str(cam_id)):
            if cam in cam_list:
                image = cv2.imread(os.path.join(img_dir, info[images_key]))
                if self.opt.real:
                    image = cv2.resize(image, (720, 480))
                images.append(image)
                image_idx.append(info['image_id'])
                trans_mat_list.append(np.array(info['trans_mat'], dtype=np.float32))
                trans_mats_n010_list.append(np.array(info['trans_mat_n010'], dtype=np.float32))
                trans_mats_n005_list.append(np.array(info['trans_mat_n005'], dtype=np.float32))
                trans_mats_p005_list.append(np.array(info['trans_mat_p005'], dtype=np.float32))
                trans_mats_p007_list.append(np.array(info['trans_mat_p007'], dtype=np.float32))
                trans_mats_p010_list.append(np.array(info['trans_mat_p010'], dtype=np.float32))
                trans_mats_p015_list.append(np.array(info['trans_mat_p015'], dtype=np.float32))
                trans_mats_p020_list.append(np.array(info['trans_mat_p020'], dtype=np.float32))
                trans_mats_p080_list.append(np.array(info['trans_mat_p080'], dtype=np.float32))
                shift_mats_1_list.append(np.array(info['shift_mats'][1*self.opt.map_scale], dtype=np.float32))
                shift_mats_2_list.append(np.array(info['shift_mats'][2*self.opt.map_scale], dtype=np.float32))
                shift_mats_4_list.append(np.array(info['shift_mats'][4*self.opt.map_scale], dtype=np.float32))
                shift_mats_8_list.append(np.array(info['shift_mats'][8*self.opt.map_scale], dtype=np.float32))
        trans_mats = np.concatenate([x[None,:,:] for x in trans_mat_list], axis=0)
        trans_mats_n010 = np.concatenate([x[None,:,:] for x in trans_mats_n010_list], axis=0)
        trans_mats_n005 = np.concatenate([x[None,:,:] for x in trans_mats_n005_list], axis=0)
        trans_mats_p005 = np.concatenate([x[None,:,:] for x in trans_mats_p005_list], axis=0)
        trans_mats_p007 = np.concatenate([x[None,:,:] for x in trans_mats_p007_list], axis=0)
        trans_mats_p010 = np.concatenate([x[None,:,:] for x in trans_mats_p010_list], axis=0)
        trans_mats_p015 = np.concatenate([x[None,:,:] for x in trans_mats_p015_list], axis=0)
        trans_mats_p020 = np.concatenate([x[None,:,:] for x in trans_mats_p020_list], axis=0)
        trans_mats_p080 = np.concatenate([x[None,:,:] for x in trans_mats_p080_list], axis=0)
        shift_mats_1 = np.concatenate([x[None,:,:] for x in shift_mats_1_list], axis=0)
        shift_mats_2 = np.concatenate([x[None,:,:] for x in shift_mats_2_list], axis=0)
        shift_mats_4 = np.concatenate([x[None,:,:] for x in shift_mats_4_list], axis=0)
        shift_mats_8 = np.concatenate([x[None,:,:] for x in shift_mats_8_list], axis=0)
        return images, image_idx, [trans_mats, trans_mats_n010, trans_mats_n005, trans_mats_p005, trans_mats_p007, trans_mats_p010, trans_mats_p015, trans_mats_p020, trans_mats_p080],\
                        [shift_mats_1, shift_mats_2, shift_mats_4, shift_mats_8]

    def load_sample_func(self, index):
        info = self.samples[index]
        img_dir = self.img_dir[index] if len(self.img_dir) == len(self.samples) else self.img_dir
        images_key = 'image'
        images = []
        image_idx = []
        image = cv2.imread(os.path.join(img_dir, info[images_key]))
        if self.opt.real:
            image = cv2.resize(image, (720, 480))
        images.append(image)
        image_idx.append(info['image_id'])
        trans_mats = np.array(info['trans_mat'], dtype=np.float32)[None,:,:]
        trans_mats_n010 = np.array(info['trans_mat_n010'], dtype=np.float32)[None,:,:]
        trans_mats_n005 = np.array(info['trans_mat_n005'], dtype=np.float32)[None,:,:]
        trans_mats_p005 = np.array(info['trans_mat_p005'], dtype=np.float32)[None,:,:]
        trans_mats_p007 = np.array(info['trans_mat_p007'], dtype=np.float32)[None,:,:]
        trans_mats_p010 = np.array(info['trans_mat_p010'], dtype=np.float32)[None,:,:]
        trans_mats_p015 = np.array(info['trans_mat_p015'], dtype=np.float32)[None,:,:]
        trans_mats_p020 = np.array(info['trans_mat_p020'], dtype=np.float32)[None,:,:]
        trans_mats_p080 = np.array(info['trans_mat_p080'], dtype=np.float32)[None,:,:]
        shift_mats_1 = np.array(info['shift_mats'][1*self.opt.map_scale], dtype=np.float32)[None,:,:]
        shift_mats_2 = np.array(info['shift_mats'][2*self.opt.map_scale], dtype=np.float32)[None,:,:]
        shift_mats_4 = np.array(info['shift_mats'][4*self.opt.map_scale], dtype=np.float32)[None,:,:]
        shift_mats_8 = np.array(info['shift_mats'][8*self.opt.map_scale], dtype=np.float32)[None,:,:]
        return images, image_idx, [trans_mats, trans_mats_n010, trans_mats_n005, trans_mats_p005, trans_mats_p007, trans_mats_p010, trans_mats_p015, trans_mats_p020, trans_mats_p080],\
                     [shift_mats_1, shift_mats_2, shift_mats_4, shift_mats_8]

    def __getitem__(self, index):
        if 'NO_MESSAGE' in self.opt.message_mode:
            images, image_idx, trans_mats, shift_mats = self.load_sample_func(index)
        else:
            images, image_idx, trans_mats, shift_mats = self.load_image_func(index)
        
        scaled_images, meta = {}, {}
        for scale in opt.test_scales:
            cur_images = []
            for image in images:
                cur_image, cur_meta = self.pre_process_func(image, scale)
                cur_images.append(cur_image)
            scaled_images[scale] = np.concatenate(cur_images, axis=0)
            meta[scale] = cur_meta
        return image_idx, {'images': scaled_images, 'image': images, 'meta': meta, \
                            'trans_mats': trans_mats[0], 'trans_mats_n010': trans_mats[1], 'trans_mats_n005': trans_mats[2], 'trans_mats_p005': trans_mats[3],\
                            'trans_mats_p007': trans_mats[4], 'trans_mats_p010': trans_mats[5], 'trans_mats_p015': trans_mats[6], 'trans_mats_p020': trans_mats[7],\
                            'trans_mats_p080': trans_mats[8], \
                            'shift_mats_1': shift_mats[0], 'shift_mats_2': shift_mats[1], 'shift_mats_4': shift_mats[2], 'shift_mats_8': shift_mats[3]}

    def __len__(self):
        if 'NO_MESSAGE' in self.opt.message_mode:
            return len(self.samples)
        else:
            return len(self.samples)*5


def prefetch_test(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)
    print('Message mode: {}'.format(opt.message_mode))
    Logger(opt)
    Detector = detector_factory[opt.task]

    split = 'val' if not opt.trainval else 'test'
    dataset = Dataset(opt, split)
    detector = Detector(opt)

    data_loader = torch.utils.data.DataLoader(
        PrefetchDataset(opt, dataset, detector.pre_process),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    results = {}
    if opt.coord == 'Joint':
        results_i = {}
    num_iters = len(dataset)
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'comm_rate']
    avg_time_stats = {t: AverageMeter() for t in time_stats}
    for ind, (img_idx, pre_processed_images) in enumerate(data_loader):
        ret = detector.run(pre_processed_images, img_idx)
        if opt.coord == 'Joint':
            rets, ret_i = ret['results']
        else:
            rets = ret['results']
        
        for i in range(len(rets)):
            img_id = img_idx[i]
            results[img_id.numpy().astype(np.int32)[0]] = rets[i]
        
        if opt.coord == 'Joint':
            for i in range(len(ret_i)):
                img_id = img_idx[i]
                results_i[img_id.numpy().astype(np.int32)[0]] = ret_i[i]

        Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
        for t in avg_time_stats:
            avg_time_stats[t].update(ret[t])
            Bar.suffix = Bar.suffix + '|{} {tm.val:.4f}s ({tm.avg:.4f}s) '.format(
                t, tm=avg_time_stats[t])
        bar.next()
    bar.finish()
    print('#################### comm_rate: {:.6f}s) #################### '.format(avg_time_stats['comm_rate'].avg))

    if opt.coord == 'Joint':
        print('######################################################')
        print('################# BEV Object Detection ###############')
        print('######################################################')
        if opt.polygon:
            dataset.run_polygon_eval(results, opt.save_dir, 'Global')
        else:
            dataset.run_eval(results, opt.save_dir, 'Global')
        print('######################################################')
        print('################# UAV Object Detection ###############')
        print('######################################################')
        dataset.run_eval(results_i, opt.save_dir, 'Local')
    elif opt.coord == 'Local':
        print('######################################################')
        print('################# UAV Object Detection ###############')
        print('######################################################')
        dataset.run_eval(results, opt.save_dir, 'Local')
    elif opt.coord == 'Global':
        print('######################################################')
        print('################# BEV Object Detection ###############')
        print('######################################################')
        if opt.polygon:
            dataset.run_polygon_eval(results, opt.save_dir, 'Global')
        else:    
            dataset.run_eval(results, opt.save_dir, 'Global')

if __name__ == '__main__':
    opt = opts().parse()
    prefetch_test(opt)
