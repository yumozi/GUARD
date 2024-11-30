'''This code is modified from https://github.com/liuzechun/Data-Free-NAS'''

import torch
import torch.nn as nn
from torch import distributed
import numpy as np
import json
import os
import argparse

DATASETS = {"imagenet", "tiny-imagenet", "imagenette", "cifar10", "cifar100", "wolf", "birds", "fruits", "cats", "a", "b", "c", "d", "e"}

def get_num_class_map(datasets):
    num_class_map = dict()
    for ds in datasets:
        d = ds.lower()
        if d == "imagenet":
            num_class_map[d] = 1000
        elif d == "tiny-imagenet":
            num_class_map[d] = 200
        elif d == "cifar100":
            num_class_map[d] = 100
        else:
            num_class_map[d] = 10
    return num_class_map

def get_image_size_map(datasets):
    image_size_map = dict()
    for ds in datasets:
        d = ds.lower()
        if d == "imagenet" or d == "imagenette":
            image_size_map[d] = 224
        elif d == "tiny-imagenet":
            # While it should be 64 according to latest official SRe2L code,
            # we've been using 224 for experiments and getting good result
            image_size_map[d] = 224
        elif 'cifar' in d:
            image_size_map[d] = 32
        else:
            image_size_map[d] = 224  # TO DO: validate
    return image_size_map

NUM_CLASS_MAP = get_num_class_map(DATASETS)
IMAGE_SIZE_MAP = get_image_size_map(DATASETS)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def save_arguments(script_name, args):
    directory = "../log/"
    filename = os.path.join(directory, f"{args.exp_name}.json")
    os.makedirs(directory, exist_ok=True)

    # If the JSON file exists, load its data. If not, initialize an empty list.
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
    else:
        data = []

    found = False
    for entry in data:
        if "script" not in entry: continue
        if entry["script"] == script_name:
            # Update the arguments for this script
            entry["args"] = vars(args)
            found = True
            break

    # If script_name was not found in the list, append a new entry
    if not found:
        entry = {
            "script": script_name,
            "args": vars(args)
        }
        data.append(entry)

    # Save the updated data back to the JSON file
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def distributed_is_initialized():
    if distributed.is_available():
        if distributed.is_initialized():
            return True
    return False


def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr


def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)


def beta_policy(mom_fn):
    def _alr(optimizer, iteration, epoch, param, indx):
        mom = mom_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group[param][indx] = mom

    return _alr


def mom_cosine_policy(base_beta, warmup_length, epochs):
    def _beta_fn(iteration, epoch):
        if epoch < warmup_length:
            beta = base_beta * (epoch + 1) / warmup_length
        else:
            beta = base_beta
        return beta

    return beta_policy(_beta_fn)


def clip(image_tensor, use_fp16=False):
    '''
    adjust the input based on mean and variance
    '''
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor


def tiny_clip(image_tensor):
    """
    adjust the input based on mean and variance, using tiny-imagenet normalization
    """
    mean = np.array([0.4802, 0.4481, 0.3975])
    std = np.array([0.2302, 0.2265, 0.2262])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor


def denormalize(image_tensor, use_fp16=False):
    '''
    convert floats back to input
    '''
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor


def tiny_denormalize(image_tensor):
    """
    convert floats back to input, using tiny-imagenet normalization
    """
    mean = np.array([0.4802, 0.4481, 0.3975])
    std = np.array([0.2302, 0.2265, 0.2262])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor

class ViT_BNFeatureHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        B, N, C = input[0].shape
        mean = torch.mean(input[0], dim=[0, 1])
        var = torch.var(input[0], dim=[0, 1], unbiased=False)
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(module.running_mean.data - mean, 2)
        self.r_feature = r_feature

    def close(self):
        self.hook.remove()


class BNFeatureHook():
    def __init__(self, module, per_class_bn=False, start=None, end=None):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.per_class_bn = per_class_bn
        self.start = start
        self.end = end

    def hook_fn(self, module, input, output):
        if self.per_class_bn:
            raise NotImplementedError("This function has not been implemented yet.")

        else:
            nch = input[0].shape[1]
            mean = input[0].mean([0, 2, 3])
            var = input[0].permute(1, 0, 2, 3).contiguous().reshape([nch, -1]).var(1, unbiased=False)
            r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
                module.running_mean.data - mean, 2)
            self.r_feature = r_feature

    def close(self):
        self.hook.remove()

class EmbedderHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.feature = None

    def hook_fn(self, module, input, output):
        self.feature = output.view(output.shape[0], -1)

    def close(self):
        self.hook.remove()

def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0

    return loss_var_l1, loss_var_l2


# modified from Alibaba-ImageNet21K/src_files/models/utils/factory.py
def load_model_weights(model, model_path):
    state = torch.load(model_path, map_location="cpu")

    Flag = False
    if "state_dict" in state:
        # resume from a model trained with nn.DataParallel
        state = state["state_dict"]
        Flag = True

    for key in model.state_dict():
        if "num_batches_tracked" in key:
            continue
        p = model.state_dict()[key]

        if Flag:
            key = "module." + key

        if key in state:
            ip = state[key]
            # if key in state['state_dict']:
            #     ip = state['state_dict'][key]
            if p.shape == ip.shape:
                p.data.copy_(ip.data)  # Copy the data of parameters
            else:
                print("could not load layer: {}, mismatch shape {} ,{}".format(key, (p.shape), (ip.shape)))
        else:
            print("could not load layer: {}, not in checkpoint".format(key))
    return model