# Modified from https://github.com/kuangliu/pytorch-cifar

import argparse
import os
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from utils import DATASETS, str2bool, NUM_CLASS_MAP, mkdir

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--dataset", default="imagenet", type=str,
                        choices=DATASETS, help="dataset path")
parser.add_argument("--data-path", default="/media/techt/DATA/data/tiny-imagenet", type=str, help="dataset path")
parser.add_argument("--model", default="resnet18", type=str, help="model name")
parser.add_argument('--cure', type=str2bool, nargs='?', const=True, default=False,
                    help="whether to use CURE for curvature regularization")
parser.add_argument('--lamda', type=float, default=100, help='lamda for CURE method')
parser.add_argument('--h', type=float, default=3., help='h for CURE method')  # default=3 according to original paper
parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
parser.add_argument("--exp-name", default="99", type=str, help="name of the experiment")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument("--resume", "-r", action="store_true", help="resume from checkpoint")
parser.add_argument("--output-dir", default="./output", type=str)
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--check-ckpt", default=None, type=str)
args = parser.parse_args()

if args.output_dir:
    args.save_dir = os.path.join(args.output_dir, args.dataset + '_' + args.model, args.exp_name)
    mkdir(args.save_dir)

if args.check_ckpt:
    checkpoint = torch.load(args.check_ckpt)
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]
    print(f"==> test ckp: {args.check_ckpt}, acc: {best_acc}, epoch: {start_epoch}")
    exit()


if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)


device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print("==> Preparing data..")

data_mean = (0.4914, 0.4822, 0.4465)
data_std = (0.2023, 0.1994, 0.2010)
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std),
    ]
)

if args.dataset == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform_test)
else:
    trainset = torchvision.datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# Model
print("==> Building model..")

args.num_classes = NUM_CLASS_MAP[args.dataset]
model = torchvision.models.get_model(args.model, num_classes=args.num_classes)
model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
model.maxpool = nn.Identity()


net = model.to(device)
if device == "cuda":
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
    checkpoint = torch.load("./checkpoint/ckpt.pth")
    net.load_state_dict(checkpoint["net"])
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

# Train
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = net(inputs)
        loss = criterion(outputs, targets)

        if args.cure:
            inputs.requires_grad_()
            net.eval()
            outputs = net(inputs)
            loss_z = criterion(outputs, targets)
            loss_z.backward()
            grad = inputs.grad.data
            z = grad.detach()  # z = torch.sign(grad).detach()
            z = args.h * (z + 1e-7) / (z.reshape(z.size(0), -1).norm(dim=1)[:, None, None, None] + 1e-7)
            inputs.grad.zero_()
            net.zero_grad()

            outputs_pos = net(inputs + z)
            outputs_orig = net(inputs)

            loss_pos = criterion(outputs_pos, targets)
            loss_orig = criterion(outputs_orig, targets)
            grad_diff = torch.autograd.grad((loss_pos - loss_orig), inputs,
                                            create_graph=True)[0]
            reg = grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1)
            net.zero_grad()
            curv_loss = torch.mean(args.lamda * reg)

            loss += curv_loss
            net.train()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print(f"Epoch: [{epoch}], Acc@1 {100.*correct/total:.3f}, Loss {train_loss/(batch_idx+1):.4f}")


# Test
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(f"Test: Acc@1 {100.*correct/total:.3f}, Loss {test_loss/(batch_idx+1):.4f}")

    # Save checkpoint.
    acc = 100.0 * correct / total
    # if acc > best_acc:
    # save last checkpoint
    if True:
        state = {
            "state_dict": net.module.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        # if not os.path.isdir('checkpoint'):
        #     os.mkdir('checkpoint')

        path = os.path.join(args.save_dir, "checkpoint.pth")
        torch.save(state, path)
        best_acc = acc


start_time = time.time()
for epoch in range(start_epoch, start_epoch + args.epochs):
    train(epoch)
    # fast test
    if epoch % 10 == 0 or epoch == args.epochs - 1:
        test(epoch)
    scheduler.step()
end_time = time.time()
print(f"total time: {end_time - start_time} s")