import os
import torch
from torch import nn
import torch.distributed
import torchvision
from torchvision.transforms import functional as t_F
import numpy as np
import json
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

def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        normalized = normalize_fn(tensor, self.mean, self.std)
        return normalized

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)

class RandomResizedCropWithCoords(torchvision.transforms.RandomResizedCrop):
    def __init__(self, **kwargs):
        super(RandomResizedCropWithCoords, self).__init__(**kwargs)

    def __call__(self, img, coords):
        try:
            reference = (coords.any())
        except:
            reference = False
        if not reference:
            i, j, h, w = self.get_params(img, self.scale, self.ratio)
            coords = (i / img.size[1],
                      j / img.size[0],
                      h / img.size[1],
                      w / img.size[0])
            coords = torch.FloatTensor(coords)
        else:
            i = coords[0].item() * img.size[1]
            j = coords[1].item() * img.size[0]
            h = coords[2].item() * img.size[1]
            w = coords[3].item() * img.size[0]
        return t_F.resized_crop(img, i, j, h, w, self.size,
                                 self.interpolation), coords


class ComposeWithCoords(torchvision.transforms.Compose):
    def __init__(self, **kwargs):
        super(ComposeWithCoords, self).__init__(**kwargs)

    def __call__(self, img, coords, status):
        for t in self.transforms:
            if type(t).__name__ == 'RandomResizedCropWithCoords':
                img, coords = t(img, coords)
            elif type(t).__name__ == 'RandomCropWithCoords':
                img, coords = t(img, coords)
            elif type(t).__name__ == 'RandomHorizontalFlipWithRes':
                img, status = t(img, status)
            else:
                img = t(img)
        return img, status, coords


class RandomHorizontalFlipWithRes(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, status):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """

        if status is not None:
            if status == True:
                return t_F.hflip(img), status
            else:
                return img, status
        else:
            status = False
            if torch.rand(1) < self.p:
                status = True
                return t_F.hflip(img), status
            return img, status


    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


def get_FKD_info(fkd_path):
    def custom_sort_key(s):
        # Extract numeric part from the string using regular expression
        numeric_part = int(s.split('_')[1].split('.tar')[0])
        return numeric_part

    max_epoch = len(os.listdir(fkd_path))
    if not fkd_path.endswith("/"):
        fkd_path += "/"
    batch_list = sorted(os.listdir(os.path.join(
        fkd_path + 'epoch_0')), key=custom_sort_key)
    batch_size = torch.load(os.path.join(
        fkd_path, 'epoch_0', batch_list[0]))[1].size()[0]
    last_batch_size = torch.load(os.path.join(
        fkd_path, 'epoch_0', batch_list[-1]))[1].size()[0]
    num_img = batch_size * (len(batch_list) - 1) + last_batch_size

    print('======= FKD: dataset info ======')
    print('path: {}'.format(fkd_path))
    print('num img: {}'.format(num_img))
    print('batch size: {}'.format(batch_size))
    print('max epoch: {}'.format(max_epoch))
    print('================================')
    return max_epoch, batch_size, num_img

class ImageFolder_FKD_MIX(torchvision.datasets.ImageFolder):
    def __init__(self, fkd_path, mode, args_epoch=None, args_bs=None, **kwargs):
        self.fkd_path = fkd_path
        self.mode = mode
        super(ImageFolder_FKD_MIX, self).__init__(**kwargs)
        self.batch_config = None  # [list(coords), list(flip_status)]
        self.batch_config_idx = 0  # index of processing image in this batch
        if self.mode == 'fkd_load':
            max_epoch, batch_size, num_img = get_FKD_info(self.fkd_path)
            if args_epoch > max_epoch:
                raise ValueError(f'`--epochs` should be no more than max epoch.')
            if args_bs != batch_size:
                raise ValueError('`--batch-size` should be same in both saving and loading phase. Please use `--gradient-accumulation-steps` to control batch size in model forward phase.')
            # self.img2batch_idx_list = torch.load('/path/to/img2batch_idx_list.tar')
            self.img2batch_idx_list = get_img2batch_idx_list(num_img=num_img, batch_size=batch_size, epochs=max_epoch)
            self.epoch = None

    def __getitem__(self, index):
        path, target = self.samples[index]

        if self.mode == 'fkd_save':
            coords_ = None
            flip_ = None
        elif self.mode == 'fkd_load':
            if self.batch_config == None:
                raise ValueError('config is not loaded')
            assert self.batch_config_idx <= len(self.batch_config[0])

            coords_ = self.batch_config[0][self.batch_config_idx]
            flip_ = self.batch_config[1][self.batch_config_idx]

            self.batch_config_idx += 1
        else:
            raise ValueError('mode should be fkd_save or fkd_load')

        sample = self.loader(path)

        if self.transform is not None:
            sample_new, flip_status, coords_status = self.transform(sample, coords_, flip_)
        else:
            flip_status = None
            coords_status = None

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample_new, target, flip_status, coords_status

    def load_batch_config(self, img_idx):
        """Use the `img_idx` to locate the `batch_idx`

        Args:
            img_idx: index of the first image in this batch
        """
        assert self.epoch != None
        batch_idx = self.img2batch_idx_list[self.epoch][img_idx]
        batch_config_path =  os.path.join(self.fkd_path, 'epoch_{}'.format(self.epoch), 'batch_{}.tar'.format(batch_idx))

        # [coords, flip_status, mix_index, mix_lam, mix_bbox, soft_label]
        config = torch.load(batch_config_path)
        self.batch_config_idx = 0
        self.batch_config = config[:2]
        return config[2:]

    def set_epoch(self, epoch):
        self.epoch = epoch



def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(images, args, rand_index=None, lam=None, bbox=None):
    if args.mode == 'fkd_save':
        rand_index = torch.randperm(images.size()[0]).cuda()
        lam = np.random.beta(args.cutmix, args.cutmix)
        bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
    elif args.mode == 'fkd_load':
        assert rand_index is not None and lam is not None and bbox is not None
        rand_index = rand_index.cuda()
        lam = lam
        bbx1, bby1, bbx2, bby2 = bbox
    else:
        raise ValueError('mode should be fkd_save or fkd_load')

    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
    return images, rand_index.cpu(), lam, [bbx1, bby1, bbx2, bby2]


def mixup(images, args, rand_index=None, lam=None):
    if args.mode == 'fkd_save':
        rand_index = torch.randperm(images.size()[0]).cuda()
        lam = np.random.beta(args.mixup, args.mixup)
    elif args.mode == 'fkd_load':
        assert rand_index is not None and lam is not None
        rand_index = rand_index.cuda()
        lam = lam
    else:
        raise ValueError('mode should be fkd_save or fkd_load')

    mixed_images = lam * images + (1 - lam) * images[rand_index]
    return mixed_images, rand_index.cpu(), lam, None


def mix_aug(images, args, rand_index=None, lam=None, bbox=None):
    if args.mix_type == 'mixup':
        return mixup(images, args, rand_index, lam)
    elif args.mix_type == 'cutmix':
        return cutmix(images, args, rand_index, lam, bbox)
    else:
        return images, None, None, None

def get_img2batch_idx_list(num_img = 50000, batch_size = 1024, seed=42, epochs=300):
    train_dataset = torch.utils.data.TensorDataset(torch.arange(num_img))
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = torch.utils.data.RandomSampler(train_dataset, generator=generator)
    batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=batch_size, drop_last=False)

    img2batch_idx_list = []
    for epoch in range(epochs):
        img2batch_idx = {}
        for batch_idx, img_indices in enumerate(batch_sampler):
            img2batch_idx[img_indices[0]] = batch_idx

        img2batch_idx_list.append(img2batch_idx)
    return img2batch_idx_list


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

