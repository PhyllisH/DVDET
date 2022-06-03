from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from progress.bar import Bar
from lib.models.networks.Compressor import compressor
from models.data_parallel import DataParallel
from utils.utils import AverageMeter


class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss, map_scale):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss
        self.map_scale = map_scale

    def forward(self, batch):
        outputs = self.model(batch['input'], [batch['trans_mats'], batch['trans_mats_n005'], batch['trans_mats_n010'], batch['trans_mats_p005'], batch['trans_mats_p007'], batch['trans_mats_p010'], batch['trans_mats_p015'], batch['trans_mats_p020'], batch['trans_mats_p080']], \
                             [batch['shift_mats_1'], batch['shift_mats_2'], batch['shift_mats_4'], batch['shift_mats_8']], self.map_scale)
        loss, compressor_loss, compressor_aux_loss, loss_stats = self.loss(outputs, batch)
        return outputs[-1], loss, compressor_loss, compressor_aux_loss, loss_stats

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        torch.nn.utils.clip_grad_norm_(group["params"], grad_clip)
        # for param in group["params"]:
        #     if param.grad is not None:
        #         param.grad.data.clamp_(-grad_clip, grad_clip)
class BaseTrainer(object):
    def __init__(
            self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(opt)
        self.model_with_loss = ModelWithLoss(model, self.loss, self.opt.map_scale)

    def set_device(self, gpus, chunk_sizes, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus,
                chunk_sizes=chunk_sizes).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)
        
        for optimizer in self.optimizer.values():
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            if len(self.opt.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)

            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)
            output, loss, compressor_loss, compressor_aux_loss, loss_stats = model_with_loss(batch)
            if phase == 'train':
                if self.opt.train_mode == 'detector':
                    loss = loss.mean()
                    self.optimizer['detector_optimizer'].zero_grad()
                    loss.backward()
                    self.optimizer['detector_optimizer'].step()
                elif self.opt.train_mode == 'compressor':
                    self.optimizer['compressor_optimizer'].zero_grad()
                    self.optimizer['compressor_aux_optimizer'].zero_grad()

                    compressor_loss = compressor_loss.mean()
                    compressor_loss.backward()
                    clip_gradient(self.optimizer['compressor_optimizer'], 5)
                    self.optimizer['compressor_optimizer'].step()

                    compressor_aux_loss = compressor_aux_loss.mean()
                    compressor_aux_loss.backward()
                    self.optimizer['compressor_aux_optimizer'].step()
                else:
                    self.optimizer['optimizer'].zero_grad()
                    tot_loss = (loss + compressor_loss).mean()
                    tot_loss.backward()
                    clip_gradient(self.optimizer['optimizer'], 5)
                    self.optimizer['optimizer'].step()


            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)
            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['input'].size(0))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            if not opt.hide_data_time:
                Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                                          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            if opt.print_iter > 0:
                if iter_id % opt.print_iter == 0:
                    print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
            else:
                bar.next()

            if opt.debug > 0:
                self.debug(batch, output, iter_id)

            if opt.test:
                self.save_result(output, batch, results)
            del output, loss, compressor_loss, compressor_aux_loss, loss_stats

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret, results

    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        raise NotImplementedError

    def _get_losses(self, opt):
        raise NotImplementedError

    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)
