'''
Author: yhu
Contact: phyllis1sjtu@outlook.com
LastEditTime: 2021-07-04 15:57:34
Description: 
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from numpy.lib.type_check import imag
from progress.bar import Bar
import time
import torch
import os
import matplotlib.pyplot as plt

try:
    from external.nms import soft_nms
except:
    print('NMS not imported! If you need it,'
          ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import ctdet_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import ctdet_post_process
from utils.debugger import Debugger

from .base_detector import BaseDetector


class MultiAgentDetector(BaseDetector):
    def __init__(self, opt):
        super(MultiAgentDetector, self).__init__(opt)

    def process(self, images, trans_mats, return_time=False):
        with torch.no_grad():
            output = self.model(images, trans_mats)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            reg = output['reg'] if self.opt.reg_offset else None
            if self.opt.flip_test:
                hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
                wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
                reg = reg[0:1] if reg is not None else None
            torch.cuda.synchronize()
            forward_time = time.time()
            dets = ctdet_decode(hm, wh, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)

        if return_time:
            return output, dets, forward_time
        else:
            return output, dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes)
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            dets[0][j][:, :4] /= scale
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)
            if len(self.scales) > 1 or self.opt.nms:
                soft_nms(results[j], Nt=0.5, method=2)
        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def debug(self, debugger, images, dets, output, scale=1):
        detection = dets.detach().cpu().numpy().copy()
        detection[:, :, :4] *= self.opt.down_ratio
        for i in range(images.size(1)):
            img = images[0, i].detach().cpu().numpy().transpose(1, 2, 0)
            img = ((img * self.std + self.mean) * 255).astype(np.uint8)
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
            debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
            for k in range(len(dets[i])):
                if detection[i, k, 4] > self.opt.center_thresh:
                    debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1],
                                           detection[i, k, 4],
                                           img_id='out_pred_{:.1f}'.format(scale))

    def show_results(self, debugger, image, results):
        for i in range(len(image)):
            debugger.add_img(image[i].cpu().numpy(), img_id='ctdet')
            for j in range(1, self.num_classes + 1):
                for bbox in results[i][j]:
                    if bbox[4] > self.opt.vis_thresh:
                        debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctdet')
        debugger.show_all_imgs(pause=self.pause)
    
    def save_attn_weights(self, img_idx, images, output):
        if 'weight_mats' not in output:
            return
        images = images.cpu().numpy().transpose(0,1,3,4,2)
        img_ids = [str(x.cpu().numpy()[0]) for x in img_idx]
        images = (((images * self.std) + self.mean) * 255.).astype('uint8')
        b, num_agents, c, h, w = images.shape
        weight_mats = output['weight_mats'][-1].cpu().numpy()  # (b, k_agents, q_agents, 1, h, w)
        val_mats = output['val_mats'][-1].max(dim=-3)[0].cpu().numpy()  # (b, k_agents, q_agents, c, h, w)
        root_dir = os.path.join(os.path.dirname(__file__), '../../../exp/multiagent_det', self.opt.exp_id, 'weight_mats_vis')
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        fig, axes = plt.subplots(num_agents*2+1, num_agents)
        save_path = os.path.join(root_dir, '{}.png'.format('_'.join(img_ids)))
        for j in range(num_agents):
            axes[0,j].imshow(images[0,j][:,:,::-1])
            axes[0,j].set_xticks([])
            axes[0,j].set_yticks([])
            
        for j in range(num_agents):
            for k in range(num_agents):
                axes[j*2+1,k].imshow((val_mats[0,k,j]*255.).astype('uint8'))
                axes[j*2+1,k].set_xticks([])
                axes[j*2+1,k].set_yticks([])

                axes[j*2+2, k].imshow((weight_mats[0,k,j,0]*255.).astype('uint8'))
                axes[j*2+2, k].set_xticks([])
                axes[j*2+2, k].set_yticks([])
        plt.savefig(save_path)
        plt.close()

    def run(self, image_or_path_or_tensor, img_idx=None, meta=None):
        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        merge_time, tot_time = 0, 0
        debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug == 3),
                            theme=self.opt.debugger_theme)
        start_time = time.time()
        pre_processed = False
        if isinstance(image_or_path_or_tensor, np.ndarray):
            image = image_or_path_or_tensor
        elif type(image_or_path_or_tensor) == type(''):
            image = cv2.imread(image_or_path_or_tensor)
        else:
            image = image_or_path_or_tensor['image']
            pre_processed_images = image_or_path_or_tensor
            pre_processed = True

        loaded_time = time.time()
        load_time += (loaded_time - start_time)

        detections = []
        results = []
        for scale in self.scales:
            scale_start_time = time.time()
            if not pre_processed:
                images, meta = self.pre_process(image, scale, meta)
            else:
                # import pdb; pdb.set_trace()
                images = pre_processed_images['images'][scale]
                meta = pre_processed_images['meta'][scale]
                meta = {k: v.numpy()[0] for k, v in meta.items()}
                trans_mats = pre_processed_images['trans_mats']
            images = images.to(self.opt.device)
            trans_mats = trans_mats.to(self.opt.device)
            torch.cuda.synchronize()
            pre_process_time = time.time()
            pre_time += pre_process_time - scale_start_time
            output, dets, forward_time = self.process(images, trans_mats, return_time=True)

            torch.cuda.synchronize()
            net_time += forward_time - pre_process_time
            decode_time = time.time()
            dec_time += decode_time - forward_time

            if self.opt.debug >= 2:
                self.debug(debugger, images, dets, output, scale)
            
            if self.opt.vis_weight_mats:
                self.save_attn_weights(img_idx, images, output)

            for i in range(len(dets)):
                detections.append(self.post_process(dets[i:i+1], meta, scale))
                results.append(self.merge_outputs([detections[-1]]))
            torch.cuda.synchronize()
            post_process_time = time.time()
            post_time += post_process_time - decode_time
            
        torch.cuda.synchronize()
        end_time = time.time()
        merge_time += end_time - post_process_time
        tot_time += end_time - start_time

        if self.opt.debug >= 1:
            self.show_results(debugger, image, results)

        return {'results': results, 'tot': tot_time, 'load': load_time,
                'pre': pre_time, 'net': net_time, 'dec': dec_time,
                'post': post_time, 'merge': merge_time}