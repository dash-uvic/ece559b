import math
import torch
from glob import glob
import os,sys
import torch
import torchvision.transforms.functional as FF
from PIL import Image
import numpy as np


def _keep_last(fn, keep_last):
    dir_path = os.path.dirname(fn)
    files = set(glob(os.path.join(dir_path, "*"))) - set(glob(os.path.join(dir_path, '*best.pt')))
    if len(files) <= keep_last: return

    files = sorted(files, key=lambda t: -os.stat(t).st_mtime)
    del files[:keep_last]
    for old_files in files:
        print(f"| deleting {old_files}")
        os.remove(old_files)

def save_gif(imgs, fn, keep_last=None):
    pils = []
  
    for img in imgs:
        if isinstance(img, torch.Tensor):
            img = img.detach()
            img = FF.to_pil_image(img)
        else:
            img = Image.fromarray(np.uint8(img))

        pils.append(img)

    pils[0].save(fn, save_all=True, append_images=pils[1:], duration=1000, loop=0)

    if keep_last is not None:
        _keep_last(fn, keep_last)

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
