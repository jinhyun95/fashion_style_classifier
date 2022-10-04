import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from time import time
import argparse
import os

import utils.logger as logger
from utils.load_checkpoint import load_checkpoint
from utils.metrics import TopKAllAccuracy, TopKAnyAccuracy, TopKMainOnlyAccuracy
from utils.lossfns import FocalBCELoss, AnchorCELoss, AnchorBCELoss, RatioLoss
from utils.visualization import gradcam, visualize_gradcam, visualize_feature
from dataset import fs14, deepfashion, kfashion, inference_dataset, hipster_wars
from model import baselines, learned_pooling, attentions, baseline_learned_poolings
from functools import reduce


def train(args):
    base_dir = os.path.join(args.out_dir, args.exp_name)
    writer = SummaryWriter(os.path.join(base_dir, 'tensorboard'))
    dataset = {'fashionstyle14': fs14, 'kfashion': kfashion, 'hipsterwars': hipster_wars}[args.dataset]
    collate_fn_train = dataset.collate_fn_train
    collate_fn_test = dataset.collate_fn_test
    dataset = dataset.FashionDataset
    if args.dataset == 'fashionstyle14' and args.fssplit is not None:
        tr_ds = dataset(args.data_dir, phase='train', img_size=(args.image_size, args.image_size), fssplit=args.fssplit)
        va_ds = dataset(args.data_dir, phase='val', img_size=(args.image_size, args.image_size), fssplit=args.fssplit)
    else:
        tr_ds = dataset(args.data_dir, phase='train', img_size=(args.image_size, args.image_size))
        va_ds = dataset(args.data_dir, phase='val', img_size=(args.image_size, args.image_size))
    logger.info('Dataset Created')
    tr_dl = DataLoader(tr_ds, batch_size=args.batchsize, collate_fn=collate_fn_train, shuffle=True, drop_last=True, num_workers=args.num_workers)
    va_dl = DataLoader(va_ds, batch_size=args.batchsize, collate_fn=collate_fn_test, shuffle=False, drop_last=False, num_workers=args.num_workers)

    # initialize model, optimizer, scheduler and loss criterion
    net = {'baseline': baselines.BaselineClassifier,
           'learned_pooling': learned_pooling.LearnedPooling,
           'glpool': baseline_learned_poolings.GLPool,
           'alphamex': baseline_learned_poolings.alphamex,
           'bam': attentions.ResidualBAM,
           'cbam': attentions.ResidualCBAM}[args.model]
    net = net(args, tr_ds.num_labels)
    logger.info('NUMBER OF PARAMETERS: %d ' % sum(p.numel() for p in net.parameters() if p.requires_grad))
    net = nn.DataParallel(net.cuda())
    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                           patience=1 if args.dataset == 'kfashion' else 3,
                                                           cooldown=1,
                                                           verbose=True, min_lr=1e-8)
    criterion = {'ce': nn.CrossEntropyLoss(), 'bce': nn.BCEWithLogitsLoss()}[args.criterion]

    if args.dataset == 'kfashion':
        metrics = {'top_1_any_accuracy': TopKAnyAccuracy(1), 'top_3_all_accuracy': TopKAllAccuracy(3)}
    else:
        metrics = {'top_1_main_accuracy': TopKMainOnlyAccuracy(1), 'top_3_main_accuracy': TopKMainOnlyAccuracy(3)}
    main_metric = 'top_3_all_accuracy' if args.dataset == 'kfashion' else 'top_1_main_accuracy'
    start_ep, best_perf = load_checkpoint(args, net, optimizer, main_metric)

    # start training
    for ep in range(start_ep, args.epoch):
        logger.info('EPOCH %d TRAINING START WITH LEARNING RATE %.9f' % (ep + 1, optimizer.param_groups[0]['lr']))
        epoch_start = time()
        net.train()
        step = 0
        epoch_loss = 0.
        # collections for metric calculation
        full_labels = []
        full_logits = []
        full_main_labels = []
        batch_start = time()
        for bix, batch in enumerate(tr_dl):
            step += 1
            if args.dataset == 'kfashion':
                images, labels, main_labels = [b.cuda() for b in batch]
                full_main_labels.append(main_labels)
            else:
                images, labels = [b.cuda() for b in batch]
            out = net(images)
            if isinstance(criterion, RatioLoss):
                loss = criterion(out['logits'], out['baseline_logits'], labels)
            else:
                loss = criterion(out['logits'], labels)
            if args.dataset == 'kfashion':
                performance = dict()
                for key in metrics.keys():
                    if isinstance(metrics[key], TopKMainOnlyAccuracy):
                        performance[key] = metrics[key](out['logits'], main_labels)
                    else:
                        performance[key] = metrics[key](out['logits'], labels)
            else:
                performance = dict((key, metrics[key](out['logits'], labels)) for key in metrics.keys())
            full_labels.append(labels)
            full_logits.append(out['logits'])
            epoch_loss += loss.item()
            # LOGGING
            log = 'EPOCH %d STEP %3d / %d | LOSS: %2.5f' % (ep + 1, step, len(tr_dl), loss.item())
            for key in metrics.keys():
                log += ', %s: %.5f' % (key, performance[key])
            log += ', TIME: %6.3f' %(time() - batch_start)
            logger.info(log)
            writer.add_scalars('data/loss', {'loss (tr)': loss.item()}, step + len(tr_dl) * ep)
            writer.add_scalars('data/performance', dict((key + ' (tr)', performance[key]) for key in metrics.keys()), step + len(tr_dl) * ep)
            optimizer.zero_grad()
            if torch.isnan(loss):
                logger.info('EARLY STOPPING: LOSS RETURNED NAN')
                return None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            optimizer.step()
            batch_start = time()

        for key in metrics.keys():
            if isinstance(metrics[key], TopKMainOnlyAccuracy) and args.dataset == 'kfashion':
                performance[key] = metrics[key](torch.cat(full_logits, 0), torch.cat(full_main_labels, 0))
            else:
                performance[key] = metrics[key](torch.cat(full_logits, 0), torch.cat(full_labels, 0))
        writer.add_scalars('data/loss', {'epoch loss (tr)': epoch_loss / len(tr_dl)}, len(tr_dl) * (ep + 1))
        writer.add_scalars('data/performance',
                           dict(('epoch ' + key + ' (tr)', performance[key]) for key in metrics.keys()),
                           len(tr_dl) * (ep + 1))
        # LOGGING
        log = 'EPOCH %d LOSS: %2.5f' % (ep + 1, epoch_loss / len(tr_dl))
        for key in metrics.keys():
            log += ', %s: %.5f' % (key, performance[key])
        log += ', TIME: %6.3f' % (time() - epoch_start)
        logger.info(log)
        logger.info('\tPER CLASS STATISTICS')
        logger.info('\t          Style  #Images   Top1   Top2   Top3   Top4   Top5')
        for cls in range(tr_ds.num_labels):
            results = []
            if args.dataset == 'kfashion':
                is_answer = torch.cat(full_labels, 0)[:, cls]
            else:
                is_answer = torch.eye(tr_ds.num_labels)[torch.cat(full_labels, 0)][:, cls].to(full_labels[0].device)
            for k in range(5):
                in_topk = (torch.topk(torch.cat(full_logits, 0), k + 1, 1)[1] == cls).sum(1)
                results.append(
                    (torch.logical_and(is_answer, in_topk).sum().to(torch.float32) / (is_answer.sum() + 1e-8)).item() * 100.)
            logger.info('\t%15s  %7d %6.2f %6.2f %6.2f %6.2f %6.2f' % (tr_ds.inverse_dict[cls],
                                                                       int(is_answer.sum().item()), results[0],
                                                                       results[1], results[2], results[3], results[4]))

        # evaluation
        with torch.no_grad():
            net.eval()
            epoch_start = time()
            epoch_loss = 0.
            num_instances = 0
            full_labels = []
            full_logits = []
            full_main_labels = []
            for bix, batch in enumerate(va_dl):
                if args.dataset == 'kfashion':
                    images, labels, main_labels = [b.cuda() for b in batch]
                    full_main_labels.append(main_labels)
                else:
                    images, labels = [b.cuda() for b in batch]
                out = net(images)
                if isinstance(criterion, RatioLoss):
                    epoch_loss += criterion(out['logits'], out['baseline_logits'], labels).item() * images.size(0)
                else:
                    epoch_loss += criterion(out['logits'], labels).item() * images.size(0)
                full_labels.append(labels)
                full_logits.append(out['logits'])
                num_instances += images.size(0)

            performance = dict()
            for key in metrics.keys():
                if isinstance(metrics[key], TopKMainOnlyAccuracy) and args.dataset == 'kfashion':
                    performance[key] = metrics[key](torch.cat(full_logits, 0), torch.cat(full_main_labels, 0))
                else:
                    performance[key] = metrics[key](torch.cat(full_logits, 0), torch.cat(full_labels, 0))
            writer.add_scalars('data/loss', {'epoch loss (va)': epoch_loss / num_instances}, len(tr_dl) * (ep + 1))
            writer.add_scalars('data/performance', dict(('epoch ' + key + ' (va)', performance[key]) for key in metrics.keys()), len(tr_dl) * (ep + 1))
            # LOGGING
            log = 'EPOCH %d VALIDATION LOSS: %2.5f' % (ep + 1, epoch_loss / num_instances)
            for key in metrics.keys():
                log += ', %s: %.5f' % (key, performance[key])
            log += ', TIME: %6.3f' % (time() - epoch_start)
            logger.info(log)
            logger.info('\tPER CLASS STATISTICS')
            logger.info('\t          Style  #Images   Top1   Top2   Top3   Top4   Top5')
            for cls in range(tr_ds.num_labels):
                results = []
                if args.dataset == 'kfashion':
                    is_answer = torch.cat(full_labels, 0)[:, cls]
                else:
                    is_answer = torch.eye(tr_ds.num_labels)[torch.cat(full_labels, 0)][:, cls].to(full_labels[0].device)
                for k in range(5):
                    in_topk = (torch.topk(torch.cat(full_logits, 0), k + 1, 1)[1] == cls).sum(1)
                    results.append(
                        (torch.logical_and(is_answer, in_topk).sum().to(torch.float32) /
                         (is_answer.sum() + 1e-8)).item() * 100.)
                logger.info('\t%15s  %7d %6.2f %6.2f %6.2f %6.2f %6.2f' % (tr_ds.inverse_dict[cls],
                                                                           int(is_answer.sum().item()), results[0],
                                                                           results[1], results[2], results[3], results[4]))

            # ckpt update
            torch.save({'net_state_dict': net.state_dict(), 'optim_state_dict': optimizer.state_dict()},
                       os.path.join(base_dir, 'checkpoints', 'last.pth'))
            if performance[main_metric] > best_perf:
                torch.save({'net_state_dict': net.state_dict(), 'optim_state_dict': optimizer.state_dict()},
                           os.path.join(base_dir, 'checkpoints', 'best.pth'))
                logger.info('EPOCH %d CHECKPOINT SAVED' % (ep + 1))
                best_perf = performance[main_metric]
        scheduler.step(metrics=epoch_loss)
        if optimizer.param_groups[0]['lr'] < 1.5e-8:
            logger.info('EARLY STOPPING: LEARNING RATE TOO SMALL')
            break

def test(args):
    base_dir = os.path.join(args.out_dir, args.exp_name)
    assert os.path.exists(base_dir) and os.path.exists(os.path.join(base_dir, 'checkpoints'))
    dataset = {'fashionstyle14': fs14, 'kfashion': kfashion, 'hipsterwars': hipster_wars}[args.dataset]
    collate_fn_test = dataset.collate_fn_test
    dataset = dataset.FashionDataset
    if args.dataset == 'fashionstyle14' and args.fssplit is not None:
        te_ds = dataset(args.data_dir, phase='test', img_size=(args.image_size, args.image_size), fssplit=args.fssplit)
    else:
        te_ds = dataset(args.data_dir, phase='test', img_size=(args.image_size, args.image_size))
    te_dl = DataLoader(te_ds, batch_size=args.batchsize, collate_fn=collate_fn_test, shuffle=False, drop_last=False, num_workers=args.num_workers)

    # initialize model and loss criterion
    net = {'baseline': baselines.BaselineClassifier,
           'learned_pooling': learned_pooling.LearnedPooling,
           'glpool': baseline_learned_poolings.GLPool,
           'alphamex': baseline_learned_poolings.alphamex,
           'bam': attentions.ResidualBAM,
           'cbam': attentions.ResidualCBAM}[args.model]
    net = nn.DataParallel(net(args, te_ds.num_labels).cuda())
    if args.dataset == 'kfashion':
        metrics = {'top_1_any_accuracy': TopKAnyAccuracy(1), 'top_3_all_accuracy': TopKAllAccuracy(3)}
    else:
        metrics = {'top_1_main_accuracy': TopKMainOnlyAccuracy(1), 'top_3_main_accuracy': TopKMainOnlyAccuracy(3)}

    net.load_state_dict(torch.load(os.path.join(base_dir, 'checkpoints', 'best.pth'))['net_state_dict'])
    net.eval()
    logger.info('TESTING')
    num_instances = 0
    full_labels = []
    full_logits = []
    full_main_labels = []
    for bix, batch in enumerate(te_dl):
        if args.dataset == 'kfashion':
            images, labels, main_labels = [b.cuda() for b in batch]
            full_main_labels.append(main_labels)
        else:
            images, labels = [b.cuda() for b in batch]
        with torch.no_grad():
            if args.model == 'learned_pooling' and bix == 0:
                out = net(images, True)
            else:
                out = net(images)
            num_instances += images.size(0)
            full_labels.append(labels)
            full_logits.append(out['logits'])
    for key in metrics.keys():
        if isinstance(metrics[key], TopKMainOnlyAccuracy) and args.dataset == 'kfashion':
            performance = metrics[key](torch.cat(full_logits, 0), torch.cat(full_main_labels, 0))
        else:
            performance = metrics[key](torch.cat(full_logits, 0), torch.cat(full_labels, 0))
        logger.info('TEST RESULT %s %.5f' % (key, performance))
    logger.info('\tPER CLASS STATISTICS')
    logger.info('\t          Style  #Images   Top1   Top2   Top3   Top4   Top5')
    top1s = []
    for cls in range(te_ds.num_labels):
        results = []
        if args.dataset == 'kfashion':
            is_answer = torch.cat(full_labels, 0)[:, cls]
        else:
            is_answer = torch.eye(te_ds.num_labels)[torch.cat(full_labels, 0)][:, cls].to(full_labels[0].device)
        for k in range(5):
            in_topk = (torch.topk(torch.cat(full_logits, 0), k + 1, 1)[1] == cls).sum(1)
            results.append(
                (torch.logical_and(is_answer, in_topk).sum().to(torch.float32) /
                 (is_answer.sum() + 1e-8)).item() * 100.)
        logger.info('\t%15s  %7d %6.2f %6.2f %6.2f %6.2f %6.2f' % (te_ds.inverse_dict[cls],
                                                                   int(is_answer.sum().item()), results[0],
                                                                   results[1], results[2], results[3], results[4]))
        top1s.append(results[0])
    if args.dataset == 'hipsterwars':
        logger.info('\t%.1f %.1f' % (reduce((lambda x, y: x + y), top1s) / 5., reduce((lambda x, y: x * y), top1s) ** 0.2))
        # os.remove(os.path.join(base_dir, 'checkpoints', 'best.pth'))
    full_logits = torch.cat(full_logits, 0)
    torch.save(full_logits, os.path.join(base_dir, 'result', 'test_result.pth'))

def visualize(args):
    base_dir = os.path.join(args.out_dir, args.exp_name)
    assert os.path.exists(base_dir) and os.path.exists(os.path.join(base_dir, 'checkpoints'))
    assert args.batchsize == 1
    inf_ds = inference_dataset.VisDataset(args.data_dir, (args.image_size, args.image_size))
    inf_dl = DataLoader(inf_ds, batch_size=1, collate_fn=inference_dataset.collate_fn_vis, shuffle=False, drop_last=False, num_workers=1)

    # initialize model and loss criterion
    net = {'baseline': baselines.BaselineClassifier,
           'learned_pooling': learned_pooling.LearnedPooling,
           'glpool': baseline_learned_poolings.GLPool,
           'alphamex': baseline_learned_poolings.alphamex,
           'bam': attentions.ResidualBAM,
           'cbam': attentions.ResidualCBAM}[args.model]
    net = nn.DataParallel(net(args, inf_ds.num_labels).cuda())

    try:
        net.load_state_dict(torch.load(os.path.join(base_dir, 'checkpoints', 'best.pth'))['net_state_dict'], strict=False)
    except KeyError:
        net.load_state_dict(torch.load(os.path.join(base_dir, 'checkpoints', 'best.pth')), strict=False)

    net.eval()
    from tqdm import tqdm
    for bix, batch in tqdm(enumerate(inf_dl)):
        images, label = batch
        images = images.cuda()
        label = label[0]
        net.zero_grad()
        if args.model in ['baseline', 'bam', 'cbam']:
            out = net(images)
        else:
            out = net(images, True)
        image_key = 'Visualize_%s' % '.'.join(inf_ds.img_path[bix].split('.')[:-1])
        if args.model in ['learned_pooling', 'glpool']:
            visualize_feature(images[0], image_key, out['visualized_features'],
                              os.path.join(base_dir, 'result', 'images'), label, inf_ds.labels)
        gradcams = gradcam(net.module, out, label)
        visualize_gradcam(images[0], image_key, gradcams, os.path.join(base_dir, 'result', 'images'), inf_ds.labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='visualize', help='train / test / train_test')
    parser.add_argument('--dataset', type=str, choices=['fashionstyle14', 'hipsterwars', 'kfashion'], default='kfashion')
    parser.add_argument('--fssplit', type=int, default=None)
    parser.add_argument('--data_dir', type=str, default='/data5/fashion/K-fashion')
    parser.add_argument('--out_dir', type=str, default='/data5/assets/jinhyun95/FashionStyle')
    parser.add_argument('--exp_name', type=str, default='Vis_GLPool')
    parser.add_argument('--model', type=str, default='glpool')
    parser.add_argument('--baseline_pooltype', type=str, default='avg')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--criterion', type=str, default='bce')
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--load_checkpoint', default=False, action='store_true')
    parser.add_argument('--num_workers', type=int, default=5)
    # attention params, not required while training and/or testing the baselines.
    parser.add_argument('--attentions', type=int, default=8)
    parser.add_argument('--sigma_ratio', type=float, default=8.)
    # learned pooling params
    parser.add_argument('--stochastic', default=False, action='store_true')
    parser.add_argument('--layers', type=int, default=5)
    parser.add_argument('--trunk', type=str, choices=['t', 'e', 'p'], default='t') # t: trunk e: end p: pooling only
    # other adaptive global poolings
    parser.add_argument('--twophase', default=True, action='store_true')
    args = parser.parse_args()

    # experiment directory setting and logger initialization
    base_dir = os.path.join(args.out_dir, args.exp_name)
    if not os.path.exists(os.path.join(base_dir, 'checkpoints')):
        os.makedirs(os.path.join(base_dir, 'checkpoints'))
    if not os.path.exists(os.path.join(base_dir, 'result')):
        os.makedirs(os.path.join(base_dir, 'result'))
    if not os.path.exists(os.path.join(base_dir, 'result', 'images')):
        os.makedirs(os.path.join(base_dir, 'result', 'images'))
    if not os.path.exists(os.path.join(base_dir, 'tensorboard')):
        os.makedirs(os.path.join(base_dir, 'tensorboard'))
    if not os.path.exists(os.path.join(base_dir, 'logs')):
        os.makedirs(os.path.join(base_dir, 'logs'))
    if not os.path.exists(os.path.join(base_dir, 'images')):
        os.mkdir(os.path.join(base_dir, 'images'))
    log_dir = os.path.join(args.out_dir, args.exp_name, 'logs')
    if 'visualize' in args.mode:
        logger.info('VISUALIZATION PHASE')
        visualize(args)
    else:
        logger.add_filehandler(os.path.join(log_dir, 'run_sequence_%d_log.txt' % len(os.listdir(log_dir))))
        logger.logging_verbosity(1)
        logger.info('EXPERIMENT NAME: %s' % args.exp_name)
        logger.info('MODEL: %s' % args.model)
        logger.info('BACKBONE: %s' % args.backbone)
        logger.info('CRITERION: %s' % args.criterion)
        logger.info('INITIAL LEARNING RATE: %s' % args.learning_rate)
        logger.info('DATADIR: %s' % args.data_dir)
        logger.info('OUTDIR: %s' % args.out_dir)
        if 'train' in args.mode:
            logger.info('TRAINING PHASE')
            train(args)
        if 'test' in args.mode:
            logger.info('TESTING PHASE')
            test(args)
