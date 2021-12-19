import os
import numpy as np

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imsave, imread
from skimage import measure
from skimage import draw
from skimage.color import gray2rgb
import numpy as np
from scipy import ndimage as ndi
from random import shuffle, randrange
import csv   
import cv2 as cv


class Environment:

    def __init__(self, data_dir, transform, mode="mask", 
                                            iou_threshold=0.5,
                                            single=False, 
                                            step=15, rsize=15, 
                                            image_size=224, 
                                            max_steps=100):

        assert mode in ["mask", "draw"], f"`mode` must be one of: mask, draw"

        self.single      = single 
        self.data_dir    = data_dir
        self.boxes_fn    = os.path.join(data_dir, "train.csv")
        self.transform   = transform
        self.iou_th      = iou_threshold
        self.mode        = mode
        self.max_steps   = max_steps

        self.total_score = 0
        self.image_rows  = image_size
        self.image_cols  = image_size

        self._step       = 1
        self._state      = None
        self._cur_idx    = -1
        self._grid       = step
        self._rsize      = rsize
       
        #y,x,h,w
        
        self._action_coords = np.array(
                                [[-step,0,-step,0], #up 
                                 [0,+step,0,+step], #right
                                 [+step,0,+step,0], #down
                                 [0,-step,0,-step], #left
                                 [0,0,+rsize,0], #resize height
                                 [0,0,-rsize,0], 
                                 [0,0,0,+rsize], #resize width
                                 [0,0,0,-rsize]], dtype=np.float32)
        self.num_actions = len(self._action_coords)
        

        #Load and shuffle training 
        self._data, self._bbox = self._create()
       
        #Don't shuffle when using single mode for reproducability 
        if not self.single:
            idx = list(zip(self._data, self._bbox))
            shuffle(idx)
            self._data, self._bbox = zip(*idx) 

        assert len(self._data) == len(self._bbox)

    @property
    def size(self):
        return 1 if self.single else len(self._data) 

    def __repr__(self):
        return f"Environment: {self.data_dir} : shape = {self.image_rows} | num_states = {self.size} | num_actions={self.num_actions} | iou_th={self.iou_th} | step={self._grid} | rsize={self._rsize}"

    def _gen_random_bbox(self):
        min_len = self._grid*2
        while True:
            #random size and location, with min size of _grid
            x,y = randrange(min_len,self.image_cols-min_len),randrange(min_len,self.image_rows-min_len)
            #h,w = randrange(min_len, self.image_cols-x), randrange(min_len, self.image_rows-y)
            h,w = randrange(min_len, min_len*4), randrange(min_len, min_len*4)
            bbox  = np.array([x,y,x+w,y+h])
            if self._is_valid(bbox):
                return bbox

    def reset(self, random_box=True):
        self._cur_idx    = 0 if self.single else (self._cur_idx + 1) % len(self._data) 
        self.total_score = 0
        self._step       = 1
       
        if random_box:
            #random box
            self._curr_bbox = self._gen_random_bbox()
        else:
            #center box
            w, h = self._grid, self._grid
            x, y = self.image_cols//2 - w//2, self.image_rows//2-h//2
            self._curr_bbox  = np.array([x,y,x+w,y+h])
        
        #Setup discrete states and visual output
        if self.mode == "draw":
            self._state      = self._draw_bboxes(self._data[self._cur_idx].copy(), [self._curr_bbox], fill=False) 
        else:
            self._state = self._data[self._cur_idx].copy()

        mask = self.bbox2mask(self._curr_bbox)
        self._state_render  = self._draw_bboxes(self._data[self._cur_idx].copy(), [self._curr_bbox, self._bbox[self._cur_idx]], colors=[(0,0,225),(0,255,0)], fill=False)
        
        self._state_render = np.hstack([self._state_render, np.repeat(mask, 3, axis=-1)])

        return self.to_tensor(self._state)

    def bbox2mask(self, bbox):
        mask = np.zeros((self.image_rows, self.image_cols, 1), dtype=np.uint8)
        ymin,xmin,ymax,xmax = list(map(int, bbox))
        mask[ymin:ymax, xmin:xmax] = 255
        return mask

    def render(self, show=False):
        if show:
            cv.imshow("training", self._state_render)
            cv.waitKey(1)
        return self._state_render.copy()

    def get_location(self):
        return self._curr_bbox

    def close(self):
        cv.destroyWindows()

    def state(self):
        return self._state.copy()

    #TODO: avoid all this by using albumentation library?
    #Or make my own normalize?
    def to_tensor(self, state):
        screen = state.transpose((2, 0, 1)) 
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255.
        screen = torch.from_numpy(screen)
        screen = self.transform(screen).unsqueeze(0)
        
        if self.mode == "mask":

            mask = self.bbox2mask(self._curr_bbox)
            mask = mask.transpose((2, 0, 1)) 
            mask = np.ascontiguousarray(mask, dtype=np.float32) / 255.
            mask = torch.from_numpy(mask)
            mask = F.interpolate(mask.unsqueeze(0), tuple(screen.shape[2:]), mode="nearest")
            mask[mask == 0] = -1 
            
            #mask = self.transform(mask)
            # Resize, Normalize, and add a batch dimension (BCHW)
            #return self.transform(screen).unsqueeze(0)
            return torch.cat([screen, mask], dim=1)
        
        return screen

    def _is_valid(self, bbox):
        if bbox[0] >= bbox[2]: return False
        if bbox[1] >= bbox[3]: return False
        return True

    def apply_action(self, action):
        self._prev_bbox = self._curr_bbox.copy()
        self._curr_bbox = np.add(self._curr_bbox, self._action_coords[action])
    
        is_inside = self._is_inside(self._curr_bbox)
        
        #clip the box if it's outside the bounds of the image
        self._curr_bbox = np.clip(self._curr_bbox, 0, self.image_rows-1) #assume square image 
        
        #make sure the box is properly formatted
        xmin = min(self._curr_bbox[0], self._curr_bbox[2])
        ymin = min(self._curr_bbox[1], self._curr_bbox[3])
        xmax = max(self._curr_bbox[0], self._curr_bbox[2])
        ymax = max(self._curr_bbox[1], self._curr_bbox[3])
        self._curr_bbox = np.array([xmin,ymin,xmax,ymax])
        return is_inside 

    def step(self, action, verbose=True):
       
        is_inside = self.apply_action(action) 
        
        #print(self._prev_bbox, self._action_coords[action], self._curr_bbox)
        
        iou_score = self._iou(self._curr_bbox)
        
        #calculate reward
        reward = 1+iou_score if iou_score > 0.2 else -1
       
        #determine if we have a terminal state
        episode_done = False 
        fmt_str = ""
        if iou_score > self.iou_th:
            episode_done = True
            reward = 100 #rare event, so reward highly
            fmt_str = f"done (iou: {iou_score})"
    
        if self._step == self.max_steps:
            episode_done = True
            fmt_str = f"done (max steps)"
           
        if not is_inside:
            episode_done = True
            fmt_str = f"done (outside of box)"

        self.total_score += float(reward)
        self._step += 1 
        
        if episode_done:
            if verbose:
                print(f"  | step={self._step}, score={self.total_score:.2f} | {fmt_str}")
            return None, float(reward), episode_done, {"iou" : iou_score}
       
        #transition to next state
        if self.mode == "draw":
            self._state      = self._draw_bboxes(self._data[self._cur_idx].copy(), [self._curr_bbox], fill=False) 
        else:
            self._state      = self._data[self._cur_idx].copy()
        
        mask = self.bbox2mask(self._curr_bbox)
        self._state_render  = self._draw_bboxes(self._data[self._cur_idx].copy(), [self._curr_bbox, self._bbox[self._cur_idx]], colors=[(0,0,225),(0,255,0)], fill=False) 
        self._state_render = np.hstack([self._state_render, np.repeat(mask, 3, axis=-1)])

        return self.to_tensor(self._state), float(reward), episode_done, {"iou" : iou_score} 


    def _create(self):
        total = sum(1 for line in open(self.boxes_fn))
        imgs = np.ndarray((total, self.image_rows, self.image_cols, 3), dtype=np.uint8)
        imgs_bb = np.ndarray((total, 4), dtype=np.float32)
        bad_images = 0 
        i = 0
        with open(self.boxes_fn, "r") as fh:
            csv_reader = csv.reader(fh, delimiter=',')
            for item in csv_reader:
                image_name = item[0]
                bbox = list(map(int, item[1:]))
                try:
                    img = imread(image_name)
                except Exception as e:
                    print(e)
                    print("!! Failed to read file: {}".format(image_name))
                    bad_images += 1
                    continue

                width,height = img.shape[0:2]
                wx = self.image_cols / width
                hx = self.image_rows / height

                img = np.uint8(resize(img, (self.image_rows, self.image_cols), preserve_range=True))
                if len(img.shape) == 2:
                    img = gray2rgb(img)

                bbox[0] *= wx
                bbox[2] *= wx
                bbox[1] *= hx
                bbox[3] *= hx
               
                bbox[0] = max(bbox[0], 0)
                bbox[1] = max(bbox[1], 0)
                bbox[2] = min(bbox[2], self.image_cols-1)
                bbox[3] = min(bbox[3], self.image_rows-1)

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

    #assumes a square image
    def _is_inside(self, bbox):
        if np.any(bbox < 0) or np.any(bbox >= self.image_rows):
            return False
        return True

    def _iou(self, bbox):
        gtbox = self._bbox[self._cur_idx]
        
        yA = max(bbox[0], gtbox[0])
        xA = max(bbox[1], gtbox[1])
        yB = min(bbox[2], gtbox[2])
        xB = min(bbox[3], gtbox[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        bboxArea = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
        gtboxArea = (gtbox[2] - gtbox[0] + 1) * (gtbox[3] - gtbox[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(bboxArea + gtboxArea - interArea)

        # return the intersection over union value
        return iou
   
    @staticmethod
    def draw_polygon(image, xmin,ymin,xmax,ymax, color=(0,0,255)):
        sign = 1
        H,W = image.shape[:2]
        for i in np.arange(0,3):
            xmin, ymin, xmax, ymax = xmin+(i*sign), ymin+(i*sign), xmax+(i*sign),ymax+(i*sign)
            xx,yy = draw.polygon_perimeter([xmin, xmin, xmax, xmax, xmin], [ymin, ymax, ymax, ymin, ymin])
            xx = np.clip(xx, 0, H-1)
            yy = np.clip(yy, 0, W-1)
            image[xx, yy] = color
            sign *= -1

    def _box(self, bbox):
        xmin,ymin,xmax,ymax = bbox 
        xmin = max(xmin,0)
        ymin = max(ymin,0)
        xmax = min(xmax, self.image_cols-1)
        ymax = min(ymax, self.image_rows-1)
        return xmin,ymin,xmax,ymax

    def _draw_bboxes(self, image, bboxes, colors=[(0,0,255)], fill=False):
       

        for bbox,color in zip(bboxes,colors):
            ymin,xmin,ymax,xmax = self._box(bbox)
            
            try:
                if not fill:
                    self.draw_polygon(image, ymin,xmin,ymax,xmax,color)
                    #yy,xx = draw.rectangle_perimeter((ymin,xmin), (ymax,xmax))  
                    #image[yy,xx] = color
                else:
                    xx,yy = draw.polygon([xmin, xmin, xmax, xmax, xmin], [ymin, ymax, ymax, ymin, ymin])
                    image[xx, yy] = color 
            except Exception as e:
                print(e)
                print("You shouldn't see me ... draw out of bounds.")
                continue 
        
        return image

