import torch
import torch.nn as nn
from gym import spaces
import numpy as np
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
import os,sys
from itertools import count

from model import QNetworkWithExtractor, QNetwork, QNetworkV2
from grad import BackPropagation, GradCAM

from misc_utils import print_config_info
from data_io import trainval_dataset, test_dataset
from utils import iou_score

device = torch.device("cuda")

batch_size=16
image_size=224
lr=1e-3
data_dir="/home/memet/Projects/data"
IOU_TH=0.5

def create_balanced_set(bboxes, gts, regression):
    bs = gts.size(0)
    mask = torch.zeros((bs, 1, image_size, image_size))
    labels = torch.zeros((bs, 1))
    targets = []

    for i, (box,gt) in enumerate(zip(bboxes, gts)):
        xmin = min(box[0], box[2])
        ymin = min(box[1], box[3])
        xmax = max(box[0], box[2])
        ymax = max(box[1], box[3])
        box = np.array([xmin,ymin,xmax,ymax])
        if regression:
            labels[i] = iou_score(box, gt) 
        else:
            labels[i] = 1. if iou_score(box, gt) > IOU_TH else 0.
        mask[i, :, xmin:xmax, ymin:ymax] = 1
        targets.append(torch.from_numpy(box))

    #keep the batch balanced by swapping out some negative samples with gt
    balance=batch_size//2 - labels.sum()
    for i, (gt,label) in enumerate(zip(gts,labels)):
        if balance == 0: break
        if label == 1: continue
        delta = torch.randn(4)*5
        orig = gt.clone()
        gt = torch.clip(gt + delta, 0, image_size-1)
        xmin = int(min(gt[0], gt[2]))
        ymin = int(min(gt[1], gt[3]))
        xmax = int(max(gt[0], gt[2]))
        ymax = int(max(gt[1], gt[3]))
        mask[i, :, xmin:xmax, ymin:ymax] = 1
        #just to be safe, it should be 1
        gt = np.array([xmin,ymin,xmax,ymax])
        if regression:
            labels[i] = iou_score(gt, orig)
        else:
            labels[i] = 1. if iou_score(gt, orig) > IOU_TH else 0.
        balance -= 1
        targets[i] = torch.from_numpy(gt)

    targets = torch.stack(targets)
    mask[mask==0]=-1
    return mask, labels, targets 

def create_random(bboxes, gts, regression):
    bs = gts.size(0)
    mask = torch.zeros((bs, 1, image_size, image_size))
    labels = torch.zeros((bs, 1))
    targets = []

    for i, (box,gt) in enumerate(zip(bboxes, gts)):
        xmin = min(box[0], box[2])
        ymin = min(box[1], box[3])
        xmax = max(box[0], box[2])
        ymax = max(box[1], box[3])
        box = np.array([xmin,ymin,xmax,ymax])
        if regression:
            labels[i] = iou_score(box, gt) 
        else:
            labels[i] = 1. if iou_score(box, gt) > IOU_TH else 0.
        mask[i, :, xmin:xmax, ymin:ymax] = 1
        targets.append(torch.from_numpy(box))

    targets = torch.stack(targets)
    mask[mask==0]=-1
    return mask, labels, targets 

def create_groundtruth(dummy, gts, regression):
    bs = gts.size(0)
    mask = torch.zeros((bs, 1, image_size, image_size))
    labels = torch.ones((bs, 1))
    
    for i, (gt,label) in enumerate(zip(gts,labels)):
        xmin,ymin,xmax,ymax = list(map(int, gt))
        mask[i, :, xmin:xmax, ymin:ymax] = 1

    mask[mask==0]=-1
    return mask, labels, gts 

def create_jitter_gt(dummy, gts, regression):
    bs = gts.size(0)
    mask = torch.zeros((bs, 1, image_size, image_size))
    labels = torch.zeros((bs, 1))
    targets = []

    D = 10 
    for t in count():
        for i, gt in enumerate(gts):
            if labels[i] == 1: continue

            delta = torch.randn(4)*D
            orig = gt.clone()
            gt = torch.clip(gt + delta, 0, image_size-1)
            xmin = int(min(gt[0], gt[2]))
            ymin = int(min(gt[1], gt[3]))
            xmax = int(max(gt[0], gt[2]))
            ymax = int(max(gt[1], gt[3]))
            gt = np.array([xmin,ymin,xmax,ymax])
            mask[i, :, xmin:xmax, ymin:ymax] = 1
           
            if regression:
                labels[i] = iou_score(gt, orig)
            else:
                labels[i] = 1. if iou_score(gt, orig) > IOU_TH else 0.
            if t == 0:
                targets.append(torch.from_numpy(gt))
            else:
                targets[i] = torch.from_numpy(gt)

        if labels.sum() >= 0.9:
            break
        D-=1

    targets = torch.stack(targets)
    mask[mask==0]=-1
    return mask, labels, targets 

def train(network, checkpoint, dataset, regression, args):
    batch_size = args.batch_size
    image_size = args.image_size
    lr         = args.lr
    data_dir   = args.data_dir
    epochs     = args.epochs
    IOU_TH     = args.iou_threshold
  
    resume = True 
    start_epoch = 1
    if os.path.exists(checkpoint):
        print(f"{checkpoint} exists")
        ckpt = torch.load(checkpoint)
        regression = ckpt['regression']
        dataset = ckpt['dataset']
        network = ckpt['network']
        start_epoch = ckpt['epoch']
        args_      = ckpt["args"]
        batch_size = args_.batch_size
        image_size = args_.image_size
        lr         = args_.lr
        IOU_TH     = args_.iou_threshold
        
        inp = input("Start from scratch (this will erase existing ckpt)? [YyNn]")
        if inp.upper().startswith('Y'):
            resume = False

    action_space = spaces.Box(low=0, high=image_size-1, shape=(batch_size, 4), dtype=np.uint8)
    train_loader, val_loader = trainval_dataset(dataset, data_dir, batch_size, image_size) 
   
    poster_version=False
    if network.startswith("qnet"):
        if network.startswith("qnetv2"):
            #raise Exception(f"`{network}` cannot be done with mask version") 
            print(f"!!! {network} does not use the mask")
            arch = network.split("+")[1]
            net = QNetworkV2(arch=arch, num_actions=1)
            poster_version=True
        else:
            net = QNetwork((4, image_size), num_actions=1)
    else:
        net = QNetworkWithExtractor(arch=network, num_actions=1, num_channels=4) 
    
    print(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    if regression:
        loss_fn = nn.MSELoss()
    else:
        loss_fn = nn.BCEWithLogitsLoss()
    net.cuda()
   
    if not resume:
        print(f" ... evaluating existing model")
        net.load_state_dict(ckpt['model'])
        test_loader = test_dataset(dataset, data_dir, batch_size, image_size) 
        evaluate(net, test_loader, regression, poster_version)
        return
  
    best_res = 100 if regression else 0
    fmt      = "err" if regression else "acc"
    
    for epoch in range(start_epoch, epochs+1):
        net.train()

        losses = 0
        total_res = 0
        num_images = 0

        iterator = tqdm(enumerate(train_loader), total=len(train_loader)) 
        for idx, (img,bboxes,_,_) in iterator:
            #16x4
            bboxes = torch.stack(bboxes[0][:-1]).T
            for target_fn in [create_balanced_set, create_groundtruth, create_jitter_gt, create_random]: 
                #16x4
                mask,targets,box = target_fn(action_space.sample(), bboxes, regression)
                if poster_version:
                    box = box.to(device)
                    img = img.to(device)
                    y_hat = torch.sigmoid(net(img,box))
                else:
                    #16x4x224x224
                    x = torch.cat([img, mask], dim=1).to(device)
                    y_hat = torch.sigmoid(net(x))
                
                targets = targets.to(device)
                loss = loss_fn(y_hat, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.detach().item()
                
                if regression:
                    res = torch.abs(y_hat - targets).sum() 
                else:
                    y_hat = y_hat > IOU_TH 
                    res = (y_hat == targets).sum() 
                
                total_res += res
                num_images += targets.size(0)

                iterator.set_description(f"[Train:{epoch}][ratio: {targets.sum()/targets.size(0):.2f}] {fmt} = {res / targets.size(0)}")
       
        total_res = total_res / num_images
        losses = losses / num_images
        print(f"[Train:{epoch}] {fmt} = {total_res}  loss = {losses}")
        
        save_result = False
        if epoch % 5 == 0:
            results = evaluate(net, val_loader, regression, poster_version)
            total_res = np.mean(list(results.values()))
            print(f"[Val:{epoch}] val={total_res} | {results}")
            if regression:
                if total_res <= best_res: save_result = True
            else:
                if total_res >= best_res: save_result = True
            
        """
        save_result = False
        if regression:
            if total_res <= best_res: save_result = True
        else:
            if total_res >= best_res: save_result = True
        """
        if save_result: 
            best_res = total_res
            ckpt = {
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                 fmt : total_res,
                'loss' : losses,
                'epoch' : epoch+1,
                'network': network,
                'dataset' : dataset,
                'regression' : regression,
                'args' : args
                }
            try:
                torch.save(ckpt, checkpoint)
                print(f"  [Saved model] best_res = {total_res}")
            except:
                print(f"Failed to save model: {checkpoint}") 
            

    loader = test_dataset(dataset, data_dir, batch_size, image_size) 
    results = evaluate(net, loader, regression, poster_version)
    print(f"[Test (latest)] val={np.mean(list(results.values()))} | {results}")
    net.load_state_dict(ckpt['model'])
    results = evaluate(net, loader, regression, poster_version)
    print(f"[Test (best)] val={np.mean(list(results.values()))} | {results}")



@torch.no_grad()
def evaluate(net, loader, regression, poster_version):
    net.eval()
    action_space = spaces.Box(low=0, high=image_size-1, shape=(batch_size, 4), dtype=np.uint8)

    fmt = "err" if regression else "acc"
    results = {"Balanced" : 0, "Ground-Truth" : 0, "Jitter" : 0, "Random" : 0}
    functions = [create_balanced_set, create_groundtruth, create_jitter_gt, create_random]
    for target_name, target_fn in zip(results.keys(), functions): 
        total_res = 0
        num_images = 0
        
        iterator = tqdm(enumerate(loader), total=len(loader)) 
        for idx, (img,bboxes,_,_) in iterator:
            #16x4
            bboxes = torch.stack(bboxes[0][:-1]).T
            #16x4
            mask,targets,box = target_fn(action_space.sample(), bboxes, regression)
           
            if poster_version:
                box = box.to(device)
                img = img.to(device)
                y_hat = torch.sigmoid(net(img,box))
            else:
                #16x4x224x224
                x = torch.cat([img, mask], dim=1).to(device)
                y_hat = torch.sigmoid(net(x))
            
            targets = targets.to(device)
            
            if regression:
                res = torch.abs(y_hat - targets).sum() 
            else:
                y_hat = y_hat > IOU_TH
                res = (y_hat == targets).sum() 
            
            total_res += res 
            num_images += targets.size(0)
            iterator.set_description(f"[{target_name}][ratio: {targets.sum()/targets.size(0):.2f}] {fmt} = {res / targets.size(0)}")
        results[target_name] = (total_res / num_images).item()
        print(f"[{target_name}] {fmt} = {total_res/num_images}")
   
    return results


if __name__ == "__main__":
    from misc_utils import get_hash_tag
    commit = get_hash_tag()
    
    import argparse
    parser = argparse.ArgumentParser(description="validation tests")
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    parser.add_argument("-n", "--network", type=str, default="resnet50", choices=["resnet50", "googlenet", "qnet", "qnetv2+resnet50", "qnetv2+googlenet"])
    parser.add_argument("-d", "--dataset", type=str, default="cats", choices=["cats", "seaships"])
    parser.add_argument("-r", "--regression", action="store_true")
    parser.add_argument("-b", "--batch-size", type=int, default=batch_size) 
    parser.add_argument("-i", "--image-size", type=int, default=image_size) 
    parser.add_argument("-l", "--lr", type=int, default=lr) 
    parser.add_argument("-e", "--epochs", type=int, default=100) 
    parser.add_argument("--data-dir", type=str, default=data_dir) 
    parser.add_argument("--iou-threshold", type=int, default=IOU_TH) 
    args = parser.parse_args()
    
    if args.checkpoint is None:
        args.checkpoint=f"ckpt-{commit}-{args.network}-{args.dataset}{'-reg' if args.regression else ''}.pt"

    print_config_info(args)

    train(args.network, checkpoint=args.checkpoint, dataset=args.dataset, regression=args.regression, args=args)
