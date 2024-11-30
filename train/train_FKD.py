import os
import sys
import math
import time
import shutil
import argparse
import numpy as np
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchattacks
from torchvision.transforms import InterpolationMode
from torch.optim.lr_scheduler import LambdaLR

from utils import AverageMeter, accuracy, get_parameters
sys.path.append('..')
from relabel.utils_fkd import ImageFolder_FKD_MIX, ComposeWithCoords, RandomResizedCropWithCoords, \
    RandomHorizontalFlipWithRes, mix_aug, DATASETS, get_num_class_map

# It is imported for you to access and modify the PyTorch source code (via Ctrl+Click), more details in README.md
from torch.utils.data._utils.fetch import _MapDatasetFetcher
from utils import save_arguments, NormalizeByChannelMeanStd
from models.factory import ModelFactory

all_attack_dicts = {
        'PGD - 100': {'type': 'PGD', 'eps': 1/255, 'alpha': 0.25/255, 'steps': 100},
        'Square': {'type': 'Square', 'eps': 1/255},
        'AutoAttack': {'type': 'AutoAttack', 'eps': 1/255},
        'CW': {'type': 'CW', 'c': 0.0001},
        'MIM': {'type': 'MIM', 'epsilon': 1/255, 'alpha': 1/255, 'iters': 20, 'decay': 1.0},
    }

quick_attack_dicts = {
        'PGD - 100': {'type': 'PGD', 'eps': 1/255, 'alpha': 0.25/255, 'steps': 100},
    }

def get_args():
    parser = argparse.ArgumentParser("FKD Training on ImageNet-1K")
    parser.add_argument("--dataset", default="imagenet", type=str,
                        choices=DATASETS, help="dataset name")
    parser.add_argument('--batch-size', type=int,
                        default=1024, help='batch size')
    parser.add_argument('--gradient-accumulation-steps', type=int,
                        default=1, help='gradient accumulation steps for small gpu memory')
    parser.add_argument('--start-epoch', type=int,
                        default=0, help='start epoch')
    parser.add_argument('--epochs', type=int, default=300, help='total epoch')
    parser.add_argument('-j', '--workers', default=16, type=int,
                        help='number of data loading workers')

    parser.add_argument('--train-dir', type=str, default=None,
                        help='path to training dataset')
    parser.add_argument('--val-dir', type=str,
                        default='/path/to/imagenet/val', help='path to validation dataset')
    parser.add_argument('--output-dir', type=str,
                        default='./save/1024', help='path to output dir')
    parser.add_argument("--exp-name", default="99", type=str, help="name of the experiment")

    parser.add_argument('--cos', default=False,
                        action='store_true', help='cosine lr scheduler')
    parser.add_argument('--sgd', default=False,
                        action='store_true', help='sgd optimizer')
    parser.add_argument('-lr', '--learning-rate', type=float,
                        default=1.024, help='sgd init learning rate')  # checked
    parser.add_argument('--momentum', type=float,
                        default=0.875, help='sgd momentum')  # checked
    parser.add_argument('--weight-decay', type=float,
                        default=3e-5, help='sgd weight decay')  # checked
    parser.add_argument('--adamw-lr', type=float,
                        default=0.001, help='adamw learning rate')
    parser.add_argument('--adamw-weight-decay', type=float,
                        default=0.01, help='adamw weight decay')

    parser.add_argument('--model', type=str,
                        default='resnet18', help='student model name')

    parser.add_argument('--keep-topk', type=int, default=1000,
                        help='keep topk logits for kd loss')
    parser.add_argument('-T', '--temperature', type=float,
                        default=3.0, help='temperature for distillation loss')
    parser.add_argument('--fkd-path', type=str,
                        default=None, help='path to fkd label')
    parser.add_argument('--mix-type', default=None, type=str,
                        choices=['mixup', 'cutmix', None], help='mixup or cutmix or None')
    parser.add_argument('--fkd_seed', default=42, type=int,
                        help='seed for batch loading sampler')

    args = parser.parse_args()

    args.mode = 'fkd_load'
    return args

def main():
    args = get_args()
    script_name = os.path.basename(__file__)  # Get the script's filename
    save_arguments(script_name, args)

    if not torch.cuda.is_available():
        raise Exception("need gpu to train!")


    assert os.path.exists(args.train_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Data loading
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     #std=[0.229, 0.224, 0.225])
    normalize = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = ImageFolder_FKD_MIX(
        fkd_path=args.fkd_path,
        mode=args.mode,
        args_epoch=args.epochs,
        args_bs=args.batch_size,
        root=args.train_dir,
        transform=ComposeWithCoords(transforms=[
            RandomResizedCropWithCoords(size=224,
                                        scale=(0.08, 1),
                                        interpolation=InterpolationMode.BILINEAR),
            RandomHorizontalFlipWithRes(),
            transforms.ToTensor(),  # normalize,
        ]))

    generator = torch.Generator()
    generator.manual_seed(args.fkd_seed)
    sampler = torch.utils.data.RandomSampler(train_dataset, generator=generator)

    # only main process, no worker process
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
    #     num_workers=0, pin_memory=True,
    #     prefetch_factor=None)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
        num_workers=args.workers, pin_memory=True)

    # load validation data
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.val_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),  # normalize,
        ])),
        batch_size=int(args.batch_size/4), shuffle=False,
        num_workers=args.workers, pin_memory=True)
    print('load data successfully')


    # load student model
    print("=> loading student model '{}'".format(args.model))

    num_class_map = get_num_class_map(DATASETS)
    assert args.dataset in num_class_map
    args.num_classes = num_class_map[args.dataset]

    # model = torchvision.models.__dict__[args.model](pretrained=False, num_classes=args.num_classes)
    # model = resnet18(args, num_classes=args.num_classes)
    model = ModelFactory.create(args.model, args, args.num_classes)
    if args.dataset == 'tiny-imagenet' and args.model.startswith("resnet"):
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
        model.maxpool = nn.Identity()

    model = nn.Sequential(normalize, model)
    model = nn.DataParallel(model).cuda()
    model.train()

    if args.sgd:
        optimizer = torch.optim.SGD(get_parameters(model),
                                    lr=args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(get_parameters(model),
                                      lr=args.adamw_lr,
                                      weight_decay=args.adamw_weight_decay)

    if args.cos == True:
        scheduler = LambdaLR(optimizer,
                             lambda step: 0.5 * (1. + math.cos(math.pi * step / args.epochs)) if step <= args.epochs else 0, last_epoch=-1)
    else:
        scheduler = LambdaLR(optimizer,
                             lambda step: (1.0-step/args.epochs) if step <= args.epochs else 0, last_epoch=-1)


    args.best_acc1=0
    args.optimizer = optimizer
    args.scheduler = scheduler
    args.train_loader = train_loader
    args.val_loader = val_loader

    for epoch in range(args.start_epoch, args.epochs):
        train(model, args, epoch)

        if (epoch % 30 == 0 and epoch > 209) or epoch == args.epochs - 1:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, output_dir=args.output_dir, epoch=epoch)
            if epoch == args.epochs - 1:
                _ = validate(model, args, all_attack_dicts, epoch)
            else:
                _ = validate(model, args, quick_attack_dicts, epoch)
            # remember best acc@1 and save checkpoint

        scheduler.step()


def adjust_bn_momentum(model, iters):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 1 / iters


def train(model, args, epoch=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    optimizer = args.optimizer
    scheduler = args.scheduler
    loss_function_kl = nn.KLDivLoss(reduction='batchmean')

    model.train()
    t1 = time.time()
    args.train_loader.dataset.set_epoch(epoch)
    for batch_idx, batch_data in enumerate(args.train_loader):
        images, target, flip_status, coords_status = batch_data[0]
        mix_index, mix_lam, mix_bbox, soft_label = batch_data[1:]

        images = images.cuda()
        target = target.cuda()
        soft_label = soft_label.cuda().float()  # convert to float32
        images, _, _, _ = mix_aug(images, args, mix_index, mix_lam, mix_bbox)

        optimizer.zero_grad()
        assert args.batch_size % args.gradient_accumulation_steps == 0
        small_bs = args.batch_size // args.gradient_accumulation_steps

        # images.shape[0] is not equal to args.batch_size in the last batch, usually
        if batch_idx == len(args.train_loader) - 1:
            accum_step = math.ceil(images.shape[0] / small_bs)
        else:
            accum_step = args.gradient_accumulation_steps

        for accum_id in range(accum_step):
            partial_images = images[accum_id * small_bs: (accum_id + 1) * small_bs]
            partial_target = target[accum_id * small_bs: (accum_id + 1) * small_bs]
            partial_soft_label = soft_label[accum_id * small_bs: (accum_id + 1) * small_bs]

            output = model(partial_images)
            prec1, prec5 = accuracy(output, partial_target, topk=(1, 5))

            output = F.log_softmax(output/args.temperature, dim=1)
            partial_soft_label = F.softmax(partial_soft_label/args.temperature, dim=1)
            loss = loss_function_kl(output, partial_soft_label)
            # loss = loss * args.temperature * args.temperature
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            n = partial_images.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

        optimizer.step()



        # output = model(images)
        # prec1, prec5 = accuracy(output, target, topk=(1, 5))
        # output = F.log_softmax(output/args.temperature, dim=1)
        # soft_label = F.softmax(soft_label/args.temperature, dim=1)

        # loss = loss_function_kl(output, soft_label)
        # # loss = loss * args.temperature * args.temperature

        # n = images.size(0)
        # objs.update(loss.item(), n)
        # top1.update(prec1.item(), n)
        # top5.update(prec5.item(), n)

        # if batch_idx == 0:
        #     optimizer.zero_grad()

        # # do not support accumulate gradient, batch_size is fixed to 1024
        # assert args.gradient_accumulation_steps == 1
        # if args.gradient_accumulation_steps > 1:
        #     loss = loss / args.gradient_accumulation_steps

        # loss.backward()

        # if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or batch_idx == len(args.train_loader) - 1:
        #     optimizer.step()
        #     optimizer.zero_grad()

    printInfo = 'TRAIN Iter {}: lr = {:.6f},\tloss = {:.6f},\t'.format(epoch, scheduler.get_last_lr()[0], objs.avg) + \
                'Top-1 acc = {:.6f},\t'.format(top1.avg) + \
                'Top-5 acc = {:.6f},\t'.format(top5.avg) + \
                'train_time = {:.6f}'.format((time.time() - t1))
    if epoch % 10 == 0:
        print(printInfo)
    t1 = time.time()

def evaluate_model_under_attack(model, args, attack_dict=None, epoch=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_function = nn.CrossEntropyLoss()

    model.eval()

    if attack_dict is not None:
        attack_type = attack_dict['type']
        if attack_type == 'MIM':
            # Use custom MIM attack
            attack_params = {k: v for k, v in attack_dict.items() if k != 'type'}
            attacker = mim_attack
        else: 
            attack_params = {k: v for k, v in attack_dict.items() if k != 'type'}
            attacker = torchattacks.__dict__[attack_type](model, **attack_params)

    t1 = time.time()

    for data, target in args.val_loader:
        target = target.type(torch.LongTensor)
        data, target = data.cuda(), target.cuda()

        if attack_dict is not None:
            if attack_type == 'MIM':
                data = attacker(model, data, target, **attack_params)
            else:
                data = attacker(data, target)

        with torch.no_grad():
            output = model(data)
            loss = loss_function(output, target)

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        n = data.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

    logInfo = 'TEST Iter {}: loss = {:.6f},\t'.format(epoch, objs.avg) + \
              'Top-1 acc = {:.6f},\t'.format(top1.avg) + \
              'Top-5 acc = {:.6f},\t'.format(top5.avg) + \
              'val_time = {:.6f}'.format(time.time() - t1)
    print(logInfo)

    return top1.avg


def validate(model, args, attack_dicts, epoch=None):

    accuracies = {}
    # Clean validation
    accuracies['clean'] = evaluate_model_under_attack(model, args, epoch=epoch)

    # Under different adversarial attacks
    for attack_name, attack_dict in attack_dicts.items():
        key = f'robust_{attack_name}'
        accuracies[key] = evaluate_model_under_attack(model, args, attack_dict=attack_dict, epoch=epoch)

    print(accuracies)

    # Saving aggregated performance
    filename = f"../log/{args.exp_name}.json"
    with open(filename, 'r') as f:
        data = json.load(f)
    data.append(accuracies)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

    return accuracies['clean']


def save_checkpoint(state, output_dir=None,epoch=None):
    if epoch is None:
        path = output_dir + '/' + 'checkpoint.pth.tar'
    else:
        path = output_dir + f'/checkpoint_{epoch}.pth.tar'
    torch.save(state, path)


def mim_attack(model, data, target, epsilon, alpha, iters, decay):
    """Perform MIM attack on the input data."""
    original_data = data.clone().detach()
    data.requires_grad = True
    momentum = torch.zeros_like(data)

    for _ in range(iters):
        output = model(data)
        model.zero_grad()
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        data_grad = data.grad.data

        # Update the momentum
        momentum = decay * momentum + data_grad / torch.norm(data_grad, p=1)
        data = data.detach() + alpha * momentum.sign()
        data = torch.clamp(data, original_data - epsilon, original_data + epsilon)
        data = torch.clamp(data, 0, 1)  # assuming data is normalized between 0 and 1
        data.requires_grad = True

    return data



if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn')
    main()