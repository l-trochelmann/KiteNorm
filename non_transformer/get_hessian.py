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
import matplotlib
import matplotlib.pyplot as plt

import models 
from PyHessian.pyhessian import hessian
from PyHessian.density_plot import density_generate

from utils import progress_bar
from dataclasses import asdict


def save_esd_plot(eigenvalues, weights, out_path):
    eig = np.asarray(eigenvalues, dtype=float)
    wts = np.asarray(weights, dtype=float)
    density, grids = density_generate(eig, wts, sigma_squared=1e-6)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.semilogy(grids, density + 1e-7)
    ax.set_xlabel("Eigenvalue", fontsize=12)
    ax.set_ylabel("Density (log scale)", fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    ax.set_xlim(np.min(eig) - 1, np.max(eig) + 1)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


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

net_cfg = models.CNNConfig(
    n_blocks = 16,                 # blocks per stage, with 4 stages.
    norm_config = "post-norm",      # Choose from "pre-norm", "post-norm", "no-norm".
    norm_variant = "LayerNorm",   # Choose from "LayerNorm", "RMSNorm"
    use_res_scale = True,
    use_gain = True,
    use_bias = True,
    norm_eps = 1e-6
)
net = models.NormCNN(net_cfg)

# net_cfg = models.MLPConfig(
#     n_layers=8,                   # total blocks
#     norm_config="no-norm",     # Choose from "pre-norm", "post-norm", "no-norm". If use_residual=False, "pre-norm"="post-norm"
#     norm_variant="LayerNorm",   # Choose from "LayerNorm", "RMSNorm"
#     use_res_scale=True,
#     use_gain=True,
#     use_bias=True,
#     norm_eps=1e-6,
#     use_residual=True,          # True -> Residual blocks; False -> FFN
#     use_relu=True
# )
# net = models.NormMLP(net_cfg)

MODEL_NAME = "PostNormNet_4x16L_mid-low-sigma"

net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Hessian ESD at initialisation
net.eval()

probe_inputs, probe_targets = next(iter(trainloader)) # Take a single probe batch
probe_inputs, probe_targets = probe_inputs.to(device), probe_targets.to(device)

hess_obj = hessian(net, criterion, (probe_inputs, probe_targets), cuda=(device == 'cuda'))
density_eigen, density_weight = hess_obj.density(iter=100, n_v=10)

out_dir = os.path.expanduser("~/LN-variants/results")
out_path = os.path.join(out_dir, f"init_{MODEL_NAME}.pdf")
out_pdf = save_esd_plot(density_eigen, density_weight, out_path=out_path)
print(f"ESD plot saved to: {out_pdf}")
