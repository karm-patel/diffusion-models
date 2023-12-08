import os
import zipfile
# from natsort import natsorted
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
# from model import VAE

import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchsummary import summary
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchmetrics

DATA_DIR_PATH = "dataset"
DATA_PATH = os.path.join(DATA_DIR_PATH, 'animal-faces')
DATA_URL = 'https://www.kaggle.com/datasets/andrewmvd/animal-faces/download?datasetVersionNumber=1'
device = torch.device("cuda")
cpu_device = torch.device("cpu")

class AnimalfaceDataset(Dataset):
    def __init__(self, transform, type='train', label_dict = {"dog":0, "cat":1, "wild":2} , 
                 img_width=128, debug=False) -> None:
        self.transform = transform
        # self.root_dir specifies weather you are at afhq/train or afhq/val directory
        self.label_dict = label_dict
        self.root_dir = os.path.join(DATA_PATH, type)
        assert os.path.exists(self.root_dir), f"Check for the dataset, it is not where it should be. If not present, you can download it by clicking above {DATA_URL}"
        subdir = os.listdir(self.root_dir)
        self.image_names = []
        
        for category in subdir:
            subdir_path = os.path.join(self.root_dir, category)
            self.image_names+=os.listdir(subdir_path)
        
        if debug:
            self.image_names = self.image_names[:256]
        self.img_arr = torch.zeros((len(self.image_names), 3, img_width ,img_width))
        self.labels = torch.zeros(len(self.image_names))
            
        for i,img_name in enumerate(tqdm(self.image_names)):
            label = self.label_dict[img_name.split("_")[1]]
            img_path = os.path.join(self.root_dir, img_name.split("_")[1], img_name)
            # Load image and convert it to RGB
            img = Image.open(img_path).convert('RGB')
            # Apply transformations to the image
            img = self.transform(img)
            self.img_arr[i] = img
            self.labels[i] = label
        
            
    def __getitem__(self, idx):
        return self.img_arr[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.image_names)

