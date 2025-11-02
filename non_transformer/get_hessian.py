'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import tempfile
import json
import argparse
import random
import numpy as np
import wandb

import models 

from pathlib import Path
from itertools import islice
from PyHessian.pyhessian import hessian

from dataclasses import asdict


# Args
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--config', required=True, type=str, help='path to JSON config file for model')
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

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# Model (match main.py interface)
print('==> Building model from config..')
with open(args.config, 'r') as f:
    cfg_dict = json.load(f)

arch = cfg_dict.get("arch", "").lower()
opt_name = cfg_dict.get("optim", "").lower()
cfg_model = {k: v for k, v in cfg_dict.items() if k not in ("arch", "optim")}

if arch == "cnn":
    net_cfg = models.CNNConfig(**cfg_model)
    net = models.NormCNN(net_cfg)
elif arch == "mlp":
    net_cfg = models.MLPConfig(**cfg_model)
    net = models.NormMLP(net_cfg)
else:
    raise ValueError(f"Unsupported or missing 'arch' in config: {cfg_dict.get('arch')}")

# Derive an output name like main.py’s run_name
MODEL_NAME = Path(args.config).stem

net = net.to(device)

criterion = nn.CrossEntropyLoss()


# Hessian ESD at initialisation over a subset of the training set
net.eval()
wandb.init(project="LN-variants", name=f"init-{MODEL_NAME}", config = asdict(net_cfg))

num_batches = 4  # Size of the subset (target: 4x128)
train_subset = list(islice(trainloader, num_batches))

hess_obj = hessian(net, criterion, dataloader=train_subset, cuda=(device == 'cuda'))
iter_steps, n_v = 100, 10
density_eigen, density_weight = hess_obj.density(iter=iter_steps, n_v=n_v)

# also compute the maximum eigenvalue on the same subset
max_eigs, _ = hess_obj.eigenvalues(maxIter=200, tol=1e-4, top_n=1)
lambda_max = float(max_eigs[0])

artifact = wandb.Artifact(f"hessian-esd_{MODEL_NAME}", type="hessian-esd")

# pack everything into a single .npz (raw nodes/weights + metadata)
with tempfile.TemporaryDirectory() as tmpd:
    npz_path = os.path.join(tmpd, f"init_{MODEL_NAME}.npz")
    np.savez_compressed(
        npz_path,
        eigs=np.asarray(density_eigen, dtype=float),
        wts=np.abs(np.real_if_close(np.asarray(density_weight))),
        lambda_max=lambda_max,
        iter=iter_steps,
        n_v=n_v,
        sigma_squared=1e-6,   # keep whatever you used/plan to use for smoothing
        overhead=0.01,
        arch=arch,
        model_name=MODEL_NAME,
        seed=seed,
        subset_batches=4
    )
    artifact.add_file(npz_path)

    wandb.log_artifact(artifact)

wandb.finish()

print("Hessian ESD data logged to W&B as an artifact.")
