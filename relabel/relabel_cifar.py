import argparse
import os
import time

from collections import OrderedDict
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from imagenet_ipc import ImageFolderIPC
import torchattacks
from utils_fkd import DATASETS, NUM_CLASS_MAP, NormalizeByChannelMeanStd

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Post-Training")
parser.add_argument("--dataset", default="imagenet", type=str,
                        choices=DATASETS, help="dataset name")
parser.add_argument('--model', type=str, default='resnet18',
                        help='model name from pretrained torchvision models')
parser.add_argument("--exp-name", default="99", type=str, help="name of the experiment")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument("--resume", "-r", action="store_true", help="resume from checkpoint")
parser.add_argument("--output-dir", default="./save", type=str)
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--check-ckpt", default=None, type=str)
parser.add_argument("--batch-size", default=128, type=int)

parser.add_argument("--weight-decay", default=1e-4, type=float)
parser.add_argument("--syn-data-path", default="", type=str)
parser.add_argument('--real-data-path', type=str,
                        default='', help='where to find the real data')
parser.add_argument("--teacher-path", default="", type=str)
parser.add_argument("--ipc", default=50, type=int)

args = parser.parse_args()

args.num_classes = NUM_CLASS_MAP[args.dataset]

if args.check_ckpt:
    checkpoint = torch.load(args.check_ckpt)
    acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]
    print(f"==> test ckp: {args.check_ckpt}, acc: {acc}, epoch: {start_epoch}")
    exit()


if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

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

def save_checkpoint(state, output_dir=None, epoch=None):
    if epoch is None:
        path = os.path.join(output_dir, 'checkpoint.pth.tar')
    else:
        path = os.path.join(output_dir, f'checkpoint_{epoch}.pth.tar')
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

device = "cuda" if torch.cuda.is_available() else "cpu"
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

normalize = NormalizeByChannelMeanStd(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

# Data
print("==> Preparing data..")
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)

print("=> Using IPC setting of ", args.ipc)
trainset = ImageFolderIPC(root=args.syn_data_path, transform=transform_train, ipc=args.ipc)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

if args.dataset == "cifar100":
    testset = torchvision.datasets.CIFAR100(root=args.real_data_path, train=False, download=True, transform=transform_test)
else:
    testset = torchvision.datasets.CIFAR10(root=args.real_data_path, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# Model
print("==> Building model..")

model = torchvision.models.get_model("resnet18", num_classes=args.num_classes)
model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
model.maxpool = nn.Identity()
model = nn.Sequential(normalize, model)

model = model.to(device)
if device == "cuda":
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

model_teacher = torchvision.models.get_model("resnet18", num_classes=args.num_classes)
model_teacher.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
model_teacher.maxpool = nn.Identity()

checkpoint = torch.load(args.teacher_path)
model_teacher.load_state_dict(checkpoint['state_dict'])
model_teacher = nn.Sequential(normalize, model_teacher)
model_teacher = nn.DataParallel(model_teacher).cuda()

if args.resume:
    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
    checkpoint = torch.load("./checkpoint/ckpt.pth")
    model.load_state_dict(checkpoint["state_dict"])
    acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
args.temperature = 30
loss_function_kl = nn.KLDivLoss(reduction="batchmean")


def mixup_data(x, y, alpha=0.8):
    """
    Returns mixed inputs, mixed targets, and mixing coefficients.
    For normal learning
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# Train
def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        inputs, target_a, target_b, lam = mixup_data(inputs, targets)

        optimizer.zero_grad()
        outputs = model(inputs)

        soft_label = model_teacher(inputs).detach()
        outputs_ = F.log_softmax(outputs / args.temperature, dim=1)
        soft_label = F.softmax(soft_label / args.temperature, dim=1)

        loss = loss_function_kl(outputs_, soft_label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print(f"Epoch: [{epoch}], Acc@1 {100.*correct/total:.3f}, Loss {train_loss/(batch_idx+1):.4f}")

# Test
def validate(model, args, attack_dicts, epoch=None):

    accuracies = {}
    # Clean validation
    accuracies['clean'] = evaluate_model_under_attack(model, args, epoch=epoch)

    # Under different adversarial attacks
    for attack_name, attack_dict in attack_dicts.items():
        key = f'robust_{attack_name}'
        accuracies[key] = evaluate_model_under_attack(model, args, attack_dict=attack_dict, epoch=epoch)

    print(accuracies)

    return accuracies['clean']

def evaluate_model_under_attack(model, args, attack_dict=None, epoch=None):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

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
    for batch_idx, (inputs, targets) in enumerate(testloader):
        targets = targets.type(torch.LongTensor)
        inputs, targets = inputs.to(device), targets.to(device)

        if attack_dict is not None:
            if attack_type == 'MIM':
                inputs = attacker(model, inputs, targets, **attack_params)
            else:
                inputs = attacker(inputs, targets)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    top1_acc = 100.*correct/total
    logInfo = 'TEST Iter {}: loss = {:.6f},\t'.format(epoch, test_loss/(batch_idx+1)) + \
              'Top-1 acc = {:.4f},\t'.format(100.*correct/total) + \
              'val_time = {:.4f}'.format(time.time() - t1)
    print(logInfo)
    return top1_acc

start_time = time.time()
for epoch in range(start_epoch, start_epoch + args.epochs):
    train(epoch)
    # fast test
    if (epoch % 40 == 0 and epoch > int(args.epochs*0.75)) or epoch == args.epochs - 1:
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
    scheduler.step()
end_time = time.time()
print(f"total time: {end_time - start_time} s")