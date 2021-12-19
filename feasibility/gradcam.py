import torch
import torchvision.models as M
import torchvision.transforms as T
import numpy as np
import csv
import cv2
import os,sys
from skimage.io import imsave, imread
from skimage.transform import resize
from skimage.color import gray2rgb

from grad import BackPropagation, GradCAM
from model import QNetworkWithExtractor, QNetwork
from utils import normalize

transform = T.Compose([T.ToPILImage(),
                    T.ToTensor(),
                    normalize,
                    ])
    
def dataset(boxes_fn="/home/memet/Projects/data/oxford-III-pets/train.csv", image_rows=224, image_cols=224):
    total = sum(1 for line in open(boxes_fn))
    imgs = np.ndarray((total, image_rows, image_cols, 3), dtype=np.uint8)
    imgs_bb = np.ndarray((total, 4), dtype=np.float32)
    bad_images = 0 
    i = 0
    with open(boxes_fn, "r") as fh:
        csv_reader = csv.reader(fh, delimiter=',')
        for item in csv_reader:
            if os.path.basename(item[0])[0].islower():
                continue

            image_name = os.path.join("/home/memet/Projects/data/oxford-III-pets/", item[0])
            bbox = list(map(int, item[1:]))
            try:
                img = imread(image_name)
            except Exception as e:
                print(e)
                print("!! Failed to read file: {}".format(image_name))
                bad_images += 1
                continue

            width,height = img.shape[0:2]
            wx = image_cols / width
            hx = image_rows / height

            img = np.uint8(resize(img, (image_rows, image_cols), preserve_range=True))
            if len(img.shape) == 2:
                img = gray2rgb(img)

            bbox[0] *= wx
            bbox[2] *= wx
            bbox[1] *= hx
            bbox[3] *= hx
           
            bbox[0] = max(bbox[0], 0)
            bbox[1] = max(bbox[1], 0)
            bbox[2] = min(bbox[2], image_cols-1)
            bbox[3] = min(bbox[3], image_rows-1)

            img = np.array([img])
            img_bb = np.array([bbox]) #[miny, minx, maxy, maxx]
            
            imgs[i] = img  
            imgs_bb[i] = img_bb

            i += 1
    
    print('Loading done.')

    print("Found {} bad files.".format(bad_images))
    for i in np.arange(0, bad_images):
        imgs = np.delete(imgs, (-1), axis=0)
        imgs_bb = np.delete(imgs_bb, (-1), axis=0)

    return imgs, imgs_bb
  
def to_tensor(np_array):
    tensor = np_array.transpose((2, 0, 1)) 
    tensor = np.ascontiguousarray(tensor, dtype=np.float32) / 255.
    tensor = torch.from_numpy(tensor)
    tensor = transform(tensor).unsqueeze(0)
    return tensor

def grad2img(gradient):
    print(gradient.shape)
    gradient = gradient.cpu().squeeze(0).numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    return np.uint8(gradient)

arch="resnet50"
arch="googlenet"
input_shape=(3,224)
#net = QNetworkWithExtractor(arch, num_actions, input_shape[0])
net = getattr(M, arch)(pretrained=True)
net.train()
#gradients = BackPropagation(model=net)
gradients = GradCAM(model=net)
TODO

data,bboxes = dataset()

net.eval()
for idx,(img,bbox) in enumerate(zip(data, bboxes)):
    probs, ids = gradients.forward(to_tensor(img))
    gradients.backward(ids=ids[:, [0]]) #0 = top result
    print(gradients.available())
    data_grad = gradients.generate("layer1.0.conv1") #conv1
    data_grad_np = grad2img(data_grad)
    data_grad_np = np.repeat(data_grad_np, 3, axis=-1)
    print(data_grad_np.shape)
    out = np.concatenate([img, data_grad_np], axis=1)
    print(out.shape, out.dtype, out.max())
    cv2.imwrite(f"gradients/{idx:04d}.png", out)

    #cv2.imshow("gradient", out)
    #cv2.waitKey(0)
