"""
Script to see which images are not recogized as cat in the pretrained classifiers

TODO:
    1. Output the names of the files that failed
    2. Visualize the gradients for those that are successful
"""
import sys, os
import numpy as np
import json
import torch
import requests
import torchvision.models as models

from data_io import trainval_dataset, test_dataset

os.environ['TORCH_HOME'] = '/home/memet/Projects/data/model_zoo' 

possible_archs = [
"mobilenet_v3_small",
"resnet18",
"resnet50",
"alexnet",
"vgg16",
"vgg16_bn",
"vgg19_bn",
"squeezenet",
"densenet",
"inception",
"googlenet",
"shufflenet",
"mobilenet_v2",
"mobilenet_v3_large",
"resnext50_32x4d",
"wide_resnet50_2",
"mnasnet",
"efficientnet_b0",
"efficientnet_b1",
"efficientnet_b2",
"efficientnet_b3",
"efficientnet_b4",
"efficientnet_b5",
"efficientnet_b6",
"efficientnet_b7",
"regnet_y_400mf",
"regnet_y_800mf",
"regnet_y_1_6gf",
"regnet_y_3_2gf",
"regnet_y_8gf",
"regnet_y_16gf",
"regnet_y_32gf",
"regnet_x_400mf",
"regnet_x_800mf",
"regnet_x_1_6gf",
"regnet_x_3_2gf",
"regnet_x_8gf",
"regnet_x_16gf",
"regnet_x_32gf"]

def load_class_list(url="https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"):
    fn = os.path.basename(url)
    if not os.path.exists(fn):
        r = requests.get(url, allow_redirects=True)
        open(fn, 'wb').write(r.content)

    class_idx = json.load(open(fn))
    cat_related_classes = { 
                    "281": ["n02123045", "tabby"],
                    "283": ["n02123394", "Persian_cat"], 
                    "284": ["n02123597", "Siamese_cat"], 
                    "285": ["n02124075", "Egyptian_cat"], 
                    "286": ["n02125311", "cougar"], 
                    "287": ["n02127052", "lynx"], 
                    "288": ["n02128385", "leopard"], 
                    "289": ["n02128757", "snow_leopard"], 
                    "290": ["n02128925", "jaguar"], 
                    "291": ["n02129165", "lion"], 
                    "292": ["n02129604", "tiger"], 
                    "293": ["n02130308", "cheetah"],
                    "282": ["n02123159", "tiger_cat"],
                    }
    total_set = set(class_idx)
    subset    = set(cat_related_classes)

    reduced_idx = []
    reduced_names = {}
    for name in total_set.intersection(subset):
        reduced_idx.append(int(name)) 
        reduced_names[int(name)] = cat_related_classes[name][1]

    return reduced_idx, reduced_names

@torch.no_grad()
def evaluate(net, loader, mapper):
    dset = loader.dataset
    all_misclassified = [] 
    for img,_,idx,_ in loader:
        img = img.to(device)
        output = net(img)#[:,mapper]
        pred = output.data.max(1)[1]
        misclassified = [os.path.basename(dset.data[i]) for i,p in enumerate(pred) if p not in mapper] 
        all_misclassified.extend(misclassified)
    return all_misclassified

device = torch.device("cuda")
mapper, names = load_class_list()
train_loader, val_loader = trainval_dataset("cats", "/home/memet/Projects/data", batch_size=32, image_size=224)
loader = test_dataset("cats", "/home/memet/Projects/data", batch_size=32, image_size=224)

print(mapper)
with open("misclassified.csv", "w") as fp:
    ignore_these = []
    for net_name in ["googlenet", "resnet50"]: #possible_archs:
        try:
            net = getattr(models, net_name)(pretrained=True)
        except Exception as e:
            continue

        print(f'{net_name}:')
        net.cuda()
        net.eval()

        misclassified1 = evaluate(net, train_loader, mapper)
        misclassified2 = evaluate(net, val_loader, mapper)
        misclassified3 = evaluate(net, loader, mapper)
        misclassified = misclassified1+misclassified2+misclassified3
        ignore_these += misclassified
        print("total_misclassified: ", len(misclassified))
        fp.write(f"{net_name},{len(misclassified)},{','.join(misclassified)}\n")

print(list(set(ignore_these)))
