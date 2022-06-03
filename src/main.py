from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory


def configure_optimizers(net, lr, train_mode):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return three optimizers"""

    detector_parameters = {
        n
        for n, p in net.named_parameters()
        if not n.startswith("compressor") and p.requires_grad
        # if not n.startswith("compressor") and ('self_attn' in n or 'norm3' in n or 'norm4' in n) and p.requires_grad
    }
    # detector_parameters = {
    #     n
    #     for n, p in net.named_parameters()
    #     if n in ['alpha', 'beta'] and p.requires_grad
    #     # if not n.startswith("compressor") and ('self_attn' in n or 'norm3' in n or 'norm4' in n) and p.requires_grad
    # }
    # detector_parameters = {
    #     n
    #     for n, p in net.named_parameters()
    #     if 'weight_net' in n and p.requires_grad
    #     # if not n.startswith("compressor") and ('self_attn' in n or 'norm3' in n or 'norm4' in n) and p.requires_grad
    # }

    compressor_parameters = {
        n
        for n, p in net.named_parameters()
        if n.startswith("compressor") and not n.endswith(".quantiles") and p.requires_grad
    }
    compressor_aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.startswith("compressor") and n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = compressor_parameters & compressor_aux_parameters
    union_params = compressor_parameters | compressor_aux_parameters | detector_parameters

    assert len(inter_params) == 0
    # assert len(union_params) - len(params_dict.keys()) == 0

    optimizer_dict = {}
    if train_mode == 'detector':
        detector_optimizer = torch.optim.Adam(
            (params_dict[n] for n in sorted(detector_parameters)),
            lr=lr,
        )
        optimizer_dict['detector_optimizer'] = detector_optimizer
    elif train_mode == 'compressor':
        compressor_optimizer = torch.optim.Adam(
            (params_dict[n] for n in sorted(compressor_parameters)),
            lr=lr,
        )
        compressor_aux_optimizer = torch.optim.Adam(
            (params_dict[n] for n in sorted(compressor_aux_parameters)),
            lr=lr,
        )
        optimizer_dict['compressor_optimizer'] = compressor_optimizer
        optimizer_dict['compressor_aux_optimizer'] = compressor_aux_optimizer
    else:
        optimizer = torch.optim.Adam(
            (params_dict[n] for n in sorted((detector_parameters|compressor_parameters))),
            lr=lr,
        )
        compressor_aux_optimizer = torch.optim.Adam(
            (params_dict[n] for n in sorted(compressor_aux_parameters)),
            lr=lr,
        )
        optimizer_dict['optimizer'] = optimizer
        # optimizer_dict['compressor_aux_optimizer'] = compressor_aux_optimizer
    return optimizer_dict

def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    print('Message mode: {}'.format(opt.message_mode))
    compress_flag = False if opt.train_mode=='detector' else True
    model = create_model(opt.arch, opt.heads, opt.head_conv, opt.message_mode, opt.trans_layer, opt.coord, \
                        opt.warp_mode, opt.depth_mode, opt.feat_mode, opt.feat_shape, opt.round, compress_flag, opt.comm_thre, opt.sigma)
    # import ipdb; ipdb.set_trace()
    optimizer = configure_optimizers(model, opt.lr, opt.train_mode)
    # optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    print('Setting up data...')
    val_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'val'),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    if opt.test:
        _, preds = trainer.val(0, val_loader)
        val_loader.dataset.run_eval(preds, opt.save_dir)
        return

    train_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'train'),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

    print('Starting training...')
    best = 1e10
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
        # if opt.val_intervals > 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer)
            with torch.no_grad():
                log_dict_val, preds = trainer.val(epoch, val_loader)
            for k, v in log_dict_val.items():
                logger.scalar_summary('val_{}'.format(k), v, epoch)
                logger.write('{} {:8f} | '.format(k, v))
            if log_dict_val[opt.metric] < best:
                best = log_dict_val[opt.metric]
                save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                           epoch, model)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, optimizer)
        logger.write('\n')
        if (epoch in opt.lr_step) or (epoch%10==0):
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
            if epoch in opt.lr_step:
                lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
                print('Drop LR to', lr)
                for optim in optimizer.values():
                    for param_group in optim.param_groups:
                        param_group['lr'] = lr
    logger.close()


if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
