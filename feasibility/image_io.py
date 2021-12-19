from glob import glob
import os,sys
import numpy as np
from PIL import Image
import subprocess, psutil
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as FF
from torchvision.utils import make_grid
from torchvision.utils import draw_bounding_boxes 
import matplotlib.pyplot as plt
plt.rcParams["savefig.bbox"] = 'tight'

from data_io import inv_normalize

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

def save_image(img, fn):
    img = img.squeeze(0).detach()
    img = FF.to_pil_image(img)
    print(f"Saving: {fn}")
    img.save(fn, "PNG")

def save_images(imgs, fns):
    pils = []
    for img,fn in zip(imgs,fns):
        img = img.squeeze(0).detach()
        img = FF.to_pil_image(img)
        print(f"Saving: {fn}")
        img.save(fn, "PNG")

def plot_scores():
    plt.figure(2)
    plt.clf()
    scores_t = torch.tensor(episode_scores, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(scores_t.numpy())
    # Take 100 episode scores and plot them too
    if len(scores_t) >= 100:
        means = scores_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def visualize(data, boxes, gt, prefix="", save_image=False): 

    data = (inv_normalize(data)*255).type(torch.uint8)
    for idx, (img,box,gt_) in enumerate(zip(data, boxes, gt)):
        box = torch.stack([box, gt_])
        result = draw_bounding_boxes(img, box, colors=["blue", "red"], width=1)
        show(result, fn=f"images/{prefix}_{idx:03d}.jpg", save_image=save_image)

def show(imgs, fn="", save_image=False):
    plt.clf() 
    fig = plt.figure(0)
    if not isinstance(imgs, list):
        imgs = [imgs]
    axs = fig.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = FF.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    
    if save_image:
        plt.savefig(fn) 
    else:
        #manager = plt.get_current_fig_manager()
        #manager.full_screen_toggle()
        plt.pause(1) 
        #plt.show()
