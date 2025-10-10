'''Train CIFAR10 with PyTorch.'''
from calendar import Calendar
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import random
import numpy as np
import wandb

import models 

from utils import progress_bar
from dataclasses import asdict

LOG_EVERY_BATCH = False  # For fine-grained analysis or debugging

# Metrics
def init_wandb(cfg, run_name):
  """Minimal wandb setup"""
  os.environ["WANDB__SERVICE_WAIT"] = "600"
  os.environ["WANDB_SILENT"] = "true"
  wandb.init(
    project = 'LN-variants', 
    name = run_name,
    dir = '/home/ltrochelmann/LN-variants/logs/wandb',
    config = asdict(cfg)
  )


def compute_grad_norms(model):
  """Computes gradient l2 norms for all parameters.
    
  Returns:
      dict: A dictionary mapping parameter names to their gradient norms
  """
  grad_norms = {}
  for name, param in model.named_parameters():
    if param.grad is None:
      continue
    with torch.no_grad():
      grad_norms[f"grad_l2-norm/{name}"] = param.grad.norm(p=2).item()
  return grad_norms


def get_ln_param_stats(model):
  """Computes mean and std of all normalisation-related affine parameters.
  
  Returns:
      dict: A dictionary mapping parameter names to their mean and std.
  """
  ln_param_stats = {}
  with torch.no_grad():
    for name, param in model.named_parameters():
      if any(sub in name for sub in ["bn1.weight", "bn2.weight", "shortcut.1.weight", "norm.weight"]):
        ln_param_stats[f"LN_gain_mean/{name}"] = param.data.mean().item()
        ln_param_stats[f"LN_gain_std/{name}"] = param.data.std().item()
      elif any(sub in name for sub in ["bn1.bias", "bn2.bias", "shortcut.1.bias", "norm.bias"]):
        ln_param_stats[f"LN_bias_mean/{name}"] = param.data.mean().item()
        ln_param_stats[f"LN_bias_std/{name}"] = param.data.std().item()
  return ln_param_stats


# Args
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    """--Bit-wise seeded determinism if needed--
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False"""


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# Model
print('==> Building model..')

# net_cfg = models.CNNConfig(
#     n_blocks = 4,
#     norm_config = "no-norm",      # Choose from "pre-norm", "post-norm", "no-norm".
#     norm_variant = "LayerNorm",   # Choose from "LayerNorm", "RMSNorm"
#     use_res_scale = False,
#     use_gain = True,
#     use_bias = True,
#     norm_eps = 1e-6
# )
# net = models.NormCNN(net_cfg)

net_cfg = models.MLPConfig(
    n_layers=8,
    norm_config="pre-norm",     # Choose from "pre-norm", "post-norm", "no-norm". If use_residual=False, "pre-norm"="post-norm"
    norm_variant="LayerNorm",   # Choose from "LayerNorm", "RMSNorm"
    use_res_scale=False,
    use_gain=True,
    use_bias=True,
    norm_eps=1e-6,
    use_residual=True,          # True -> Residual blocks; False -> FFN
    use_relu=True
)
net = models.NormMLP(net_cfg)

init_wandb(net_cfg, 'PreNormResidual_8L_sanitycheck')


net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    if not LOG_EVERY_BATCH:
        global epoch_train_loss
        global epoch_train_acc
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        if LOG_EVERY_BATCH:
            metrics = {
                "epoch": epoch,
                "lr": args.lr,
                "train_loss": train_loss/(batch_idx+1),
                "train_acc": 100.*correct/total
            }
            metrics.update(compute_grad_norms(net))
            metrics.update(get_ln_param_stats(net))
            wandb.log(metrics)
    if not LOG_EVERY_BATCH:
        epoch_train_loss = train_loss
        epoch_train_acc = 100.*correct/total
    

def test(epoch):
    global best_acc
    global final_test_loss
    global final_test_acc

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

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        final_test_loss = test_loss
        final_test_acc = 100.*correct/total

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    if not LOG_EVERY_BATCH:
        epoch_train_loss = 0
        epoch_train_acc = 0
    final_test_loss = 0
    final_test_acc = 0
    train(epoch)
    test(epoch)
    
    metrics = {
        "epoch": epoch,
        "lr": args.lr,
        "valid/valid_loss": final_test_loss,
        "valid/valid_acc": final_test_acc
    }
    if not LOG_EVERY_BATCH:
       metrics["train/train_loss"] = epoch_train_loss
       metrics["train/train_acc"] = epoch_train_acc

    metrics.update(compute_grad_norms(net))
    metrics.update(get_ln_param_stats(net))
    wandb.log(metrics)

    scheduler.step()
