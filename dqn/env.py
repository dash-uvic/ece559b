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

SUBSET=10
IGNORE = ['Siamese_176.jpg', 'Egyptian_Mau_112.jpg', 'Siamese_147.jpg', 'Bengal_106.jpg', 'Russian_Blue_104.jpg', 'Ragdoll_18.jpg', 'Bombay_113.jpg', 'Abyssinian_179.jpg', 'Sphynx_176.jpg', 'Sphynx_19.jpg', 'Maine_Coon_162.jpg', 'Maine_Coon_151.jpg', 'Egyptian_Mau_150.jpg', 'Ragdoll_191.jpg', 'Birman_137.jpg', 'Egyptian_Mau_192.jpg', 'Ragdoll_197.jpg', 'Abyssinian_156.jpg', 'Ragdoll_153.jpg', 'Egyptian_Mau_101.jpg', 'Russian_Blue_200.jpg', 'Abyssinian_176.jpg', 'Siamese_125.jpg', 'Egyptian_Mau_105.jpg', 'Persian_193.jpg', 'Bengal_102.jpg', 'Siamese_103.jpg', 'Bengal_142.jpg', 'Russian_Blue_106.jpg', 'Bengal_166.jpg'] 

class Environment:

    def __init__(self, data_dir, transform, split="train",
                                            mode="mask", 
                                            iou_threshold=0.5,
                                            single=False, 
                                            delta=5, 
                                            random_box=False,
                                            image_size=224, 
                                            max_steps=0):

        assert mode in ["mask", "draw", "poster"], f"`mode` must be one of: mask, draw, poster"

        self.single      = single 
        self.data_dir    = data_dir
        self.boxes_fn    = os.path.join(data_dir, f"{split}.csv") 
        self.transform   = transform
        self.iou_th      = iou_threshold
        self.mode        = mode
        self.max_steps   = max_steps
        self.random_box  = random_box

        self.total_score = 0
        self.image_rows  = image_size
        self.image_cols  = image_size

        self._step       = 1
        self._state      = None
        self._cur_idx    = -1
        self._delta      = delta 
       
        #y,x,h,w
        if self.random_box:
            self._action_coords = np.array(
                                [[-delta,0,-delta,0], #up 
                                 [0,+delta,0,+delta], #right
                                 [+delta,0,+delta,0], #down
                                 [0,-delta,0,-delta], #left
                                 [0,0,0,0], #done
                                 [0,0,+delta,0], #resize height
                                 [0,0,-delta,0], 
                                 [0,0,0,+delta], #resize width
                                 [0,0,0,-delta]
                                 ], dtype=np.float32)

            """
            elif self.max_steps > 0:
                self._action_coords = np.array(
                                        [[0,+delta,0,0], #-> 
                                         [0,0,0,-delta], #<-
                                         [+delta,0,0,0], #v
                                         [0,0,-delta,0], #^
                                         ], dtype=np.float32)
            """
        else:
            self._action_coords = np.array(
                                    [[0,+delta,0,0], #-> 
                                     [0,0,0,-delta], #<-
                                     [+delta,0,0,0], #v
                                     [0,0,-delta,0], #^
                                     [0,0,0,0], #done or NOP
                                     ], dtype=np.float32)
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
        return f"Environment: {self.data_dir} : shape = {self.image_rows} | num_states = {self.size} | num_actions={self.num_actions} | iou_th={self.iou_th} | step={self._delta}"

    def _gen_random_bbox(self):
        min_len = self._delta*2
        while True:
            #random size and location, with min size of delta 
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

        if self.random_box or random_box:
            self._curr_bbox = self._gen_random_bbox()
        else:
            self._curr_bbox = np.array([1,1,self.image_rows-1, self.image_cols-1])

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
            
            return torch.cat([screen, mask], dim=1)
        
        return screen

    def _is_valid(self, bbox):
        if bbox[0] >= bbox[2]: return False
        if bbox[1] >= bbox[3]: return False
        return True

    def apply_action(self, action):
        self._prev_bbox = self._curr_bbox.copy()
        self._curr_bbox = np.add(self._curr_bbox, self._action_coords[action])
    
        is_valid = self._is_valid(self._curr_bbox) and self._is_inside(self._curr_bbox)
        
        #assert self._is_inside(self._curr_bbox), f"Never should go outside of image: {self._prev_bbox}->{self._action_coords[action]}->{self._curr_bbox}"

        #clip the box if it's outside the bounds of the image
        self._curr_bbox = np.clip(self._curr_bbox, 0, self.image_rows-1) #assume square image 

        #make sure the box is properly formatted
        xmin = min(self._curr_bbox[0], self._curr_bbox[2])
        ymin = min(self._curr_bbox[1], self._curr_bbox[3])
        xmax = max(self._curr_bbox[0], self._curr_bbox[2])
        ymax = max(self._curr_bbox[1], self._curr_bbox[3])
        self._curr_bbox = np.array([xmin,ymin,xmax,ymax])
        
        return is_valid 

    @property
    def curr_pos(self):
        return torch.from_numpy(self._curr_bbox).unsqueeze(0)

    def _shape(self, box1):
        box2 = self._bbox[self._cur_idx]
        h1, w1 = box1[2] - box1[0], box1[3]-box1[1]
        h2, w2 = box2[2] - box2[0], box2[3]-box2[1]
        return abs(h1/w1 - h2/w2)

    def step(self, action, verbose=True):
       
        is_valid = self.apply_action(action) 
        
        #print(self._prev_bbox, self._action_coords[action], self._curr_bbox)
        
        iou_score = self._iou(self._curr_bbox)
        shape_score = self._shape(self._curr_bbox)

        #calculate reward
        #reward = 1 if iou_score > self.iou_th else 0 
        #reward = 1+iou_score if iou_score > 0.2 else -1 #TODO: try next
        #reward = 1 if iou_score > self.iou_th else 0 

        #determine if we have a terminal state
        reward       = 0
        episode_done = False 
        fmt_str = ""
        if iou_score > self.iou_th:
            reward          = 10
            episode_done    = True
            fmt_str         = f"done(1) (reward={reward}, iou: {iou_score})"
       
        if iou_score == 0:
            episode_done = True
            reward       = -100
            fmt_str      = f"done(2) (reward={reward}, iou: {iou_score})"
        
        if self.max_steps > 0:
            if self._step >= self.max_steps:
                episode_done = True
                fmt_str      = f"done(6:max steps) (step={self._step}/{self.max_steps}, reward={reward})"
        else:
            #This does not trigger if max_steps is set 
            if action == 4 and episode_done: #agent is right, but it can be a bad right ...!
                reward         = 10 if iou_score > 0.5 else -10 
                fmt_str        = f"done(3:good agent) (reward={reward}, iou: {iou_score})"
            elif action == 4 and not episode_done: #agent is wrong :(
                reward          = -10
                episode_done    = True
                fmt_str         = f"done(4:bad agent) (reward={reward}, iou: {iou_score})"

        if not is_valid:
            reward          = -100
            episode_done    = True
            fmt_str         = f"done(5:no area) (reward={reward})"

        self.total_score += float(reward)
        self._step += 1 
        
       
        #transition to next state
        if self.mode == "draw":
            self._state      = self._draw_bboxes(self._data[self._cur_idx].copy(), [self._curr_bbox], fill=False) 
        else:
            self._state      = self._data[self._cur_idx].copy()
        
        mask = self.bbox2mask(self._curr_bbox)
        self._state_render  = self._draw_bboxes(self._data[self._cur_idx].copy(), [self._curr_bbox, self._bbox[self._cur_idx]], colors=[(0,0,225),(0,255,0)], fill=False) 
        self._state_render = np.hstack([self._state_render, np.repeat(mask, 3, axis=-1)])
        
        if episode_done:
            if verbose:
                print(f"  | step={self._step}, score={self.total_score:.2f} | {fmt_str}")
            #return None, float(reward), episode_done, {"iou" : iou_score}

        return self.to_tensor(self._state), float(reward), episode_done, {"iou" : iou_score} 



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

    def _create(self):
        total = sum(1 for line in open(self.boxes_fn))
        imgs = np.ndarray((total, self.image_rows, self.image_cols, 3), dtype=np.uint8)
        imgs_bb = np.ndarray((total, 4), dtype=np.float32)
        bad_images = 0 
        i = 0
        with open(self.boxes_fn, "r") as fh:
            csv_reader = csv.reader(fh, delimiter=',')
            next(csv_reader) #skip header
            for item in csv_reader:
                image_name = item[0]

                if os.path.basename(image_name)[0].islower(): #cats only
                    continue
                
                if os.path.basename(image_name) in IGNORE: #ignore images the pretrained got wrong
                    print(f"Ignoring difficult image: {image_name}")
                    continue
                
                try:
                    bbox = list(map(int, item[1:]))
                    img = imread(os.path.join(self.data_dir, image_name))
                except Exception as e:
                    print(e)
                    print("!! Failed to read file or bbox: {} ({})".format(os.path.join(self.data_dir, image_name), item))
                    bad_images += 1
                    continue

                height,width = img.shape[0:2]
                wx = self.image_cols / width
                hx = self.image_rows / height

                img = np.uint8(resize(img, (self.image_rows, self.image_cols), preserve_range=True, anti_aliasing=True))
                if len(img.shape) == 2:
                    img = gray2rgb(img)

                bbox[0] *= hx
                bbox[2] *= hx
                bbox[1] *= wx
                bbox[3] *= wx
               
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

        print(f"Found {bad_images} bad files to remove.")
        for i in np.arange(0, bad_images):
            imgs = np.delete(imgs, (-1), axis=0)
            imgs_bb = np.delete(imgs_bb, (-1), axis=0)

        print(f"Total images: {len(imgs)}")
        imgs = imgs[:SUBSET]
        imgs_bb = imgs_bb[:SUBSET]
        print(f"Total subset images: {len(imgs)}")
        print(imgs_bb)

        assert len(imgs) == len(imgs_bb)

        return imgs, imgs_bb
