'''This code is modified from https://github.com/liuzechun/Data-Free-NAS'''

import os
import sys
import random
import argparse
import collections
import numpy as np
from PIL import Image

import torch
import torch.utils
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
import torch.utils.data.distributed
from torch.utils.data import DataLoader

from utils import *
sys.path.append('..')
from models.factory import ModelFactory
import shutil

def prepare_model(args):
    # model = models.__dict__[args.model](pretrained=not args.ckpt_path, num_classes=args.num_classes)
    # model = resnet18(args, num_classes=args.num_classes)
    model = ModelFactory.create(args.model, args, args.num_classes)

    # model_teacher = models.get_model(args.model, num_classes=1000) #num_classes=args.num_classes)
    if args.dataset == 'tiny-imagenet' and args.model.startswith("resnet"):
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
        model.maxpool = nn.Identity()
    if args.ckpt_path:
        checkpoint = torch.load(args.ckpt_path)
        model.load_state_dict(checkpoint["model"], strict=False)

    model = nn.DataParallel(model).cuda()
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

def prepare_data(args, dataset_root):
    if args.image_size == 224:
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        dataset = datasets.ImageFolder(root=dataset_root, transform=preprocess)
        sorted_classes = sorted(dataset.class_to_idx.items(), key=lambda x: x[1])
        sorted_class_names = [item[0] for item in sorted_classes]
    elif args.dataset == 'cifar10':
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ])
        dataset = datasets.CIFAR10(root=dataset_root, train=True, download=True, transform=preprocess)
        sorted_class_names = dataset.classes
    elif args.dataset == 'cifar100':
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
        ])
        dataset = datasets.CIFAR100(root=dataset_root, train=True, download=True, transform=preprocess)
        sorted_class_names = dataset.classes

    return dataset, sorted_class_names

def get_images(args, model_teacher, hook_for_display, ipc_id):
    print("get_images call")
    save_every = 100
    batch_size = args.batch_size

    best_cost = 1e4

    feature_hook = EmbedderHook(model_teacher.module.avgpool)

    # setup target labels
    # targets_all = torch.LongTensor(np.random.permutation(1000))
    n_class = args.num_classes
    targets_all = torch.LongTensor(np.arange(n_class))

    for kk in range(0, n_class, batch_size):
        loss_r_feature_layers = []
        start_cls = kk
        end_cls = min(kk+batch_size, n_class)
        targets = targets_all[start_cls:end_cls].to('cuda')
        for module in model_teacher.modules():
            if isinstance(module, nn.BatchNorm2d):
                loss_r_feature_layers.append(BNFeatureHook(module, args.per_class_bn, start_cls, end_cls))
        data_type = torch.float
        inputs = torch.randn((targets.shape[0], 3, args.image_size, args.image_size), requires_grad=True, device='cuda',
                             dtype=data_type)
        iterations_per_layer = args.iteration
        lim_0, lim_1 = args.jitter , args.jitter

        optimizer = optim.Adam([inputs], lr=args.lr, betas=[0.5, 0.9], eps = 1e-8)
        lr_scheduler = lr_cosine_policy(args.lr, 0, iterations_per_layer) # 0 - do not use warmup
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()

        for iteration in range(iterations_per_layer):
            # learning rate scheduling
            lr_scheduler(optimizer, iteration, iteration)
            if args.cda:
                min_crop = 0.08
                max_crop = 1.0

                # strategy: start with whole image with mix crop of 1, then lower to 0.08
                # easy to hard
                min_crop = 0.08
                max_crop = 1.0
                if iteration < args.milestone * iterations_per_layer:
                    if args.easy2hard_mode == "step":
                        min_crop = 1.0
                    elif args.easy2hard_mode == "linear":
                        # min_crop linear decreasing: 1.0 -> 0.08
                        min_crop = 0.08 + (1.0 - 0.08) * (1 - iteration / (args.milestone * iterations_per_layer))
                    elif args.easy2hard_mode == "cosine":
                        # min_crop cosine decreasing: 1.0 -> 0.08
                        min_crop = 0.08 + (1.0 - 0.08) * (
                                    1 + np.cos(np.pi * iteration / (args.milestone * iterations_per_layer))) / 2

                aug_function = transforms.Compose(
                    [
                        # transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
                        transforms.RandomResizedCrop(args.image_size, scale=(min_crop, max_crop)),
                        transforms.RandomHorizontalFlip(),
                    ]
                )
            else:
                aug_function = transforms.Compose([
                    transforms.RandomResizedCrop(args.image_size),
                    transforms.RandomHorizontalFlip(),
                ])
            inputs_jit = aug_function(inputs)

            # apply random jitter offsets
            off1 = random.randint(0, lim_0)
            off2 = random.randint(0, lim_1)
            inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))

            # forward pass
            optimizer.zero_grad()
            outputs = model_teacher(inputs_jit)

            loss_main = criterion(outputs, targets)

            # R_feature loss
            rescale = [args.first_bn_multiplier] + [1. for _ in range(len(loss_r_feature_layers)-1)]
            loss_r_bn_feature = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)])

            # combining losses
            loss_aux = args.r_bn * loss_r_bn_feature

            loss = loss_main + loss_aux

            if iteration % save_every==0:
                print("------------iteration {}----------".format(iteration))
                print("total loss", loss.item())
                print("main criterion", loss_main.item())
                print("weighted_aux_loss", loss_aux.item())
                print("loss_r_bn_feature", loss_r_bn_feature.item())

                # comment below line can speed up the training (no validation process)
                # if hook_for_display is not None:
                    # hook_for_display(inputs, targets)

            # do image update
            loss.backward()
            optimizer.step()

            # clip color outlayers
            inputs.data = clip(inputs.data)

            if best_cost > loss.item() or iteration == 1:
                best_inputs = inputs.data.clone()

        if args.store_best_images:
            best_inputs = inputs.data.clone() # using multicrop, save the last one
            best_inputs = denormalize(best_inputs)
            save_images(args, best_inputs, targets, ipc_id)

        # to reduce memory consumption by states of the optimizer we deallocate memory
        optimizer.state = collections.defaultdict(dict)
        for hook in loss_r_feature_layers:
            hook.close()
    torch.cuda.empty_cache()

def save_images(args, images, targets, ipc_id):
    for id in range(images.shape[0]):
        if targets.ndimension() == 1:
            class_id = targets[id].item()
        else:
            class_id = targets[id].argmax().item()

        if not os.path.exists(args.syn_data_path):
            os.mkdir(args.syn_data_path)

        # save into separate folders
        dir_path = '{}/new{:03d}'.format(args.syn_data_path, class_id)
        place_to_store = dir_path +'/class{:03d}_id{:03d}.jpg'.format(class_id,ipc_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(place_to_store)

def validate(input, target, model):
    def accuracy(output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    with torch.no_grad():
        output = model(input)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

    print("Verifier accuracy: ", prec1.item())


def main_syn(args):
    ipc_id = args.ipc_id
    if not os.path.exists(args.syn_data_path):
        os.makedirs(args.syn_data_path)

    model_teacher = prepare_model(args)

    model_verifier = models.__dict__[args.verifier_arch](pretrained=True)
    model_verifier = model_verifier.cuda()
    model_verifier.eval()
    for p in model_verifier.parameters():
        p.requires_grad = False

    hook_for_display = lambda x,y: validate(x, y, model_verifier)
    get_images(args, model_teacher, hook_for_display, ipc_id)

def get_args():
    parser = argparse.ArgumentParser(
        "SRe2L: recover data from pre-trained model")
    """Data save flags"""
    parser.add_argument("--dataset", default="imagenet", type=str,
                        choices=DATASETS, help="dataset name")
    parser.add_argument('--exp-name', type=str, default='test',
                        help='name of the experiment, subfolder under syn_data_path/{dataset}')
    parser.add_argument('--syn-data-path', type=str,
                        default='./syn_data', help='where to store synthetic data')
    parser.add_argument('--real-data-path', type=str,
                        default='', help='where to find the real data')
    parser.add_argument('--ipc', type=int, default=50,
                        help='number of synthetic images per class')
    parser.add_argument('--store-best-images', action='store_true',
                        help='whether to store best images')

    """Experiment variables"""
    parser.add_argument('--wb', type=str2bool, nargs='?', const=True, default=False,
                        help="whether to use Wasserstein barycenter method")
    parser.add_argument('--weight-wb', type=str2bool, nargs='?', const=True, default=False,
                        help="whether to learn the weight of barycenter points")
    parser.add_argument('--per-class-bn', type=str2bool, nargs='?', const=True, default=False,
                        help="whether to use batchnorm statistics per class")

    """Optimization related flags"""
    parser.add_argument('--batch-size', type=int,
                        default=100, help='number of images to optimize at the same time')
    parser.add_argument('--iteration', type=int, default=1000,
                        help='num of iterations to optimize the synthetic data')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate for optimization')
    parser.add_argument('--jitter', default=32, type=int, help='random shift on the synthetic data')
    parser.add_argument('--r-bn', type=float, default=0.05,
                        help='coefficient for BN feature distribution regularization')
    parser.add_argument('--first-bn-multiplier', type=float, default=10.,
                        help='additional multiplier on first bn layer of R_bn')
    parser.add_argument('--tv-l2', type=float, default=0.0001,
                        help='coefficient for total variation L2 loss')
    parser.add_argument('--l2-scale', type=float,
                        default=0.00001, help='l2 loss on the image')
    """Model related flags"""
    parser.add_argument('--model', type=str, default='resnet18',
                        help='model name from pretrained torchvision models')
    parser.add_argument('--ckpt-path', type=str, default='',
                        help='path to the teacher model checkpoint')
    parser.add_argument('--verifier', action='store_true',
                        help='whether to evaluate synthetic data with another model')
    parser.add_argument('--verifier-arch', type=str, default='mobilenet_v2',
                        help="model name from torchvision models to act as a verifier")
    parser.add_argument('--cda', type=str2bool, nargs='?', const=True, default=False,
                        help="whether to apply CDA method for curriculum learning")
    parser.add_argument("--easy2hard-mode", default="cosine", type=str, choices=["step", "linear", "cosine"])
    parser.add_argument("--milestone", default=0, type=float)
    parser.add_argument("--G", default="-1", type=str)
    args = parser.parse_args()

    args.syn_data_path = os.path.join(args.syn_data_path, args.dataset, args.exp_name)
    assert args.dataset in NUM_CLASS_MAP
    args.num_classes = NUM_CLASS_MAP[args.dataset]
    args.image_size = IMAGE_SIZE_MAP[args.dataset]

    return args


if __name__ == '__main__':
    args = get_args()
    script_name = os.path.basename(__file__)  # Get the script's filename
    save_arguments(script_name, args)
    print('r_bn: ', args.r_bn)
    print('lr: ', args.lr)
    assert args.per_class_bn is False

    weights_dir = os.path.join(os.path.dirname(args.syn_data_path), args.model, 'ipc' + str(args.ipc),
                               str(int(args.wb)) + '_' + str(int(args.weight_wb)))

    if not os.path.exists(args.syn_data_path):
        os.makedirs(args.syn_data_path)
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)


    for ipc_id in range(0, args.ipc):
        args.ipc_id = ipc_id
        print('ipc_id = ', ipc_id)
        main_syn(args)
    
