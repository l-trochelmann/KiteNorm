import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Blocks
class NormBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True)
        else:
            self.shortcut = nn.Identity()

class PreNormBlock(NormBlock):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__(in_planes, planes, stride)
        self.norm = nn.GroupNorm(1, in_planes)

    def forward(self, x):
        out = self.norm(x)
        out = F.relu(self.conv1(out))
        out = out + self.shortcut(x)
        return out

class PostNormBlock(NormBlock):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__(in_planes, planes, stride)
        self.norm = nn.GroupNorm(1, planes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = out + self.shortcut(x)
        out = self.norm(out)
        return out

class NoNormBlock(NormBlock):
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = out + self.shortcut(x)
        return out

# Architectures
class BlockNormNet(nn.Module):
    def __init__(self, block, num_blocks, use_res_scale, is_post_norm, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.is_post_norm = is_post_norm
        self.use_res_scale = use_res_scale

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if self.is_post_norm:
            self.norm = nn.GroupNorm(1,64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        if self.use_res_scale:  # yields cumulative variance that is invariant to n_blocks. Derived for the conv layer specifically.
            res_scale = math.sqrt(2.0) / math.sqrt(num_blocks)  
        for stride in strides:
            b = block(self.in_planes, planes, stride)
            if self.use_res_scale:
                with torch.no_grad():
                    b.conv1.weight.mul_(res_scale)
            layers.append(b)
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        if self.is_post_norm:
            out = self.norm(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# Nets. n_blocks sets the number of blocks per spatial resolution
def PreNormNet(n_blocks, use_res_scale):
    return BlockNormNet(PreNormBlock, [n_blocks, n_blocks, n_blocks, n_blocks], use_res_scale, is_post_norm=False)

def PostNormNet(n_blocks, use_res_scale):
    return BlockNormNet(PostNormBlock, [n_blocks, n_blocks, n_blocks, n_blocks], use_res_scale, is_post_norm=True)

def NoNormNet(n_blocks, use_res_scale):
    return BlockNormNet(NoNormBlock, [n_blocks, n_blocks, n_blocks, n_blocks], use_res_scale, is_post_norm=False)

