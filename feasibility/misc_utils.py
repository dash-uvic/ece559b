from glob import glob
import os,sys
import numpy as np
from PIL import Image
import subprocess, psutil
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as FF
import matplotlib.pyplot as plt
plt.rcParams["savefig.bbox"] = 'tight'

mean  = [0.485, 0.456, 0.406]
std   = [0.229, 0.224, 0.225]
normalize = T.Normalize(mean, std)
mask_norm = T.Normalize(0.5,0.5)
inv_normalize = T.Normalize(
        mean=[-mean[0]/std[0], -mean[1]/std[1], -mean[2]/std[2]],
        std=[1/std[0], 1/std[1], 1/std[2]]
    )

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

def print_config_info(args):
    import torchvision 
    print("-"*20)
    print(f"VERSIONING")
    print("-"*20)
    print(f"GIT: {get_repo_name()}:{get_hash_tag()}:{get_branch_name()}") 
    print(f"PYTHON: {sys.version}")
    print(f"TORCH: {torch.__version__}")
    print(f"TORCHVISION: {torchvision.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    print(f"N_GPU: {torch.cuda.device_count()}") 

    print("-"*20)
    print("")
    
    print("-"*20)
    print("PARAMETERS")
    print("-"*20)
    for arg in vars(args):
        obj = getattr(args, arg)
        if type(obj) == tuple:
            obj = ','.join(map(str, obj)) 
        if obj is None:
            obj = "None"
        
        try:
            print(f"{arg:<20} {obj:>}")
        except:
            print(f"{arg:<20} {obj}")
    print("-"*20)
    print("", flush=True)

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print("============ Memory Usage =================")
    print(f"(RAM) : {process.memory_info().rss * 1e-9:.03f} GB")
    print(f"(RAM) : {process.memory_percent():.1f} %")
    for device_idx in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(device_idx)
        print(f"(GPU:{device_idx}) : {alloc/(1024**3):.03f} GB")
    print("===========================================")
    print("", flush=True)

def get_hash_tag():
    short_hash = subprocess.check_output("git rev-parse --short HEAD".split()).strip().decode("utf-8")
    dirty = subprocess.check_output("git status".split()).decode("utf8")
    dirty = "" if dirty.find("modified") < 0 else "-dirty"
    return f"{short_hash}{dirty}" 

def get_repo_name():
    output = subprocess.check_output("git rev-parse --show-toplevel".split()).strip()
    output = subprocess.check_output(f"basename {output}".split()).strip()
    return output.decode("utf-8")[:-1] 

def get_branch_name():
    output = subprocess.check_output("git branch".split())#.strip()
    branch = [ x for x in output.decode("utf-8").split("\n") if x.startswith("*")]
    return branch[0][2:]

def deepcopy(func):
    def wrapper(self, *args):
        args = [ copy.deepcopy(x) for x in args ]
        return func(self, *args)
    return wrapper
