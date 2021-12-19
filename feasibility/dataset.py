import torch
from torch.utils.data import Dataset
import os
import cv2
import csv

class SeaShips(Dataset):
    def __init__(self, root='data',
                       augmentations=None,
                       train=True,
                       ):
        super().__init__()

        self.root_dir = root
        self.data_dir = os.path.join(root, 'SeaShips')
        self.ann_dir = self.data_dir
        self.img_dir = os.path.join(self.data_dir, 'JPEGImages')
        self.augmentations=augmentations
        
        if not os.path.isdir(self.data_dir):
            raise Exception(f'{self.data_dir} does not exist.')

        fn = 'train_one_boat.csv' if train else 'test_one_boat.csv'
        fn = 'train_small_one_boat.csv' if train else 'test_one_boat.csv'
        with open(os.path.join(self.ann_dir, fn), 'r') as fp:
            fp.readline()
            csv = [ line.rstrip().split(",") for line in fp.readlines() ]
            self.data = [ os.path.join(self.img_dir, x[0]) for x in csv ]
            self.targets = [ list(map(int, x[1:]))+['boat'] for x in csv]

        assert len(self.data) == len(self.targets), f"data and target sizes don't match: {len(self.data)} != {len(self.targets)})"
        self.data = tuple(self.data)
        self.targets = tuple(self.targets)

    def __str__(self):
        format_str = (f"Dataset SeaShips\n"
               f"Number of datapoints: {len(self.data)}\n"
               f"Root location: {self.data_dir}\n")
        return format_str
    
    @property
    def num_classes(self):
        return 1

    def __len__(self):
        return len(self.data)  

    def __getitem__(self, idx):
        img = cv2.imread(self.data[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        target = [self.targets[idx]]
     
        replay = {}
        if self.augmentations:
            sample = self.augmentations(image=img, bboxes=target)
            img     = sample["image"]
            target  = sample["bboxes"]
            replay  = sample['replay']
      
        return (img, target, idx, replay)

class Cats(Dataset):

    def __init__(self, root='data', 
                       augmentations=None,
                       train=True,
                       ):
        super(Cats, self).__init__()

        self.root_dir = root
        self.data_dir = os.path.join(root, 'oxford-III-pets') 
        self.train = train
        self.augmentations = augmentations


        if not os.path.isdir(self.data_dir):
            raise Exception(f'{self.data_dir} does not exist.') 
        split = 'train' if self.train else 'test' 
        self.data = []
        self.targets = []

        with open(os.path.join(self.data_dir, f"{split}.csv")) as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            next(reader, None) #skip header
            for filenm, ymin, xmin, ymax, xmax in reader:
                if os.path.basename(filenm)[0].islower(): continue 

                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                self.targets.append([xmin,ymin,xmax,ymax,"cat"])
                self.data.append(os.path.join(self.data_dir, filenm))

        assert len(self.data) == len(self.targets)
    
    def __str__(self):
        format_str = (f"Dataset Cats\n"
               f"Number of datapoints: {len(self.data)}\n"
               f"Root location: {self.data_dir}\n")
        return format_str

    @property
    def num_classes(self):
        return 1 

    def __len__(self):
        return len(self.data)  
    
    def np_to_pil(self, x):
        return Image.fromarray(x.squeeze() * 255.)

    def __getitem__(self, idx):
         
        img = cv2.imread(self.data[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        target = [self.targets[idx]]
        
        replay = {}
        if self.augmentations:
            sample = self.augmentations(image=img, bboxes=target)
            img     = sample["image"]
            target  = sample["bboxes"]
            replay  = sample['replay']
    

        return (img, target, idx, replay)

class CatsDraw(Dataset):

    def __init__(self, root='data', 
                       augmentations=None,
                       train=True,
                       ):
        super(Cats, self).__init__()

        self.root_dir = root
        self.data_dir = os.path.join(root, 'oxford-III-pets') 
        self.train = train
        self.augmentations = augmentations


        if not os.path.isdir(self.data_dir):
            raise Exception(f'{self.data_dir} does not exist.') 
        split = 'train' if self.train else 'test' 
        self.data = []
        self.targets = []

        with open(os.path.join(self.data_dir, f"{split}.csv")) as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            next(reader, None) #skip header
            for filenm, ymin, xmin, ymax, xmax in reader:
                if os.path.basename(filenm)[0].islower(): continue 

                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                self.targets.append([xmin,ymin,xmax,ymax,"cat"])
                self.data.append(os.path.join(self.data_dir, filenm))

        assert len(self.data) == len(self.targets)
    
    def __str__(self):
        format_str = (f"Dataset Cats\n"
               f"Number of datapoints: {len(self.data)}\n"
               f"Root location: {self.data_dir}\n")
        return format_str

    @property
    def num_classes(self):
        return 1 

    def __len__(self):
        return len(self.data)  
    
    def np_to_pil(self, x):
        return Image.fromarray(x.squeeze() * 255.)

    def __getitem__(self, idx):
         
        img = cv2.imread(self.data[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      
        target = self.targets[idx]
        #ymin,xmin,ymax,xmax = target
        #img_box = cv2.rectangle(img, (ymin,xmin), (ymax,xmax), (0,0,255), 3) 

        target = [target]
        
        replay = {}
        if self.augmentations:
            sample = self.augmentations(image=img, bboxes=target)
            img     = sample["image"]
            target  = sample["bboxes"]
            replay  = sample['replay']
    

        return (img, target, idx, replay)
