import torch
import cv2
import numpy as np
import os
import glob as glob
import pandas as pd 
from xml.etree import ElementTree as et
from torchvision import DataLoader
from config import TRAIN_DIR,VALID_DIR,RESIZE_TO,BATCH_SIZE,CLASSES
from utils import get_train_transform,get_valid_transform,collate_fn

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path, width, height, classes,df, transforms=None):
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.classes = classes
        self.df=df
        
        # get all the image paths in sorted order
        self.image_paths = glob.glob(f"{self.dir_path}/*.jpg")
        self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)
    def __getitem__(self, idx):
        # capture the image name and the full image path
        image_name = self.df.loc[idx, 'image_name']
       
        #image_path = os.path.join(self.dir_path, image_name)
        # read the image
        image = cv2.imread(image_name)
        # convert BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0
        
        # capture the corresponding XML file for getting the annotations
        #annot_filename = image_name[:-4] + '.xml'
        #annot_file_path = os.path.join(self.dir_path, annot_filename)
        
        boxes = []
        labels = []
        #tree = et.parse(annot_file_path)
        #root = tree.getroot()
        
        # get the height and width of the image
        image_width = image.shape[1]
        image_height = image.shape[0]
        

        xmin_final = (self.df.loc[idx,'xmin']/image_width)*self.width
        xmax_final = (self.df.loc[idx,'xmax',]/image_width)*self.width
        ymin_final = (self.df.loc[idx,'ymin']/image_height)*self.height
        yamx_final = (self.df.loc[idx,'ymax']/image_height)*self.height
            
        boxes.append([xmin_final, ymin_final, xmax_final, yamx_final])
        labels.append(self.df.loc[idx,'labels'])
        
        # bounding box to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # area of the bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # no crowd instances
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # labels to tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # prepare the final `target` dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id
        # apply the image transforms
        if self.transforms:
            sample = self.transforms(image = image_resized,
                                     bboxes = target['boxes'],
                                     labels = labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
            
        return image_resized, target
    def __len__(self):
        return len(self.df.image_name.tolist())
train_data=pd.read_csv('train_data.csv')
valid_data=pd.read_csv('valid_data.csv')
def create_train_dataset():
    train_dataset = CustomDataset(TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES,train_data, get_train_transform())
    return train_dataset
def create_valid_dataset():
    valid_dataset = CustomDataset(VALID_DIR, RESIZE_TO, RESIZE_TO, CLASSES,valid_data, get_valid_transform())
    return valid_dataset
def create_train_loader(train_dataset, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return train_loader
def create_valid_loader(valid_dataset, num_workers=0):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return valid_loader
# execute datasets.py using Python command from Terminal...
# ... to visualize sample images
# USAGE: python datasets.py
if __name__ == '__main__':
    # sanity check of the Dataset pipeline with sample visualization
    dataset = CustomDataset(
        TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES,train_data
    )
    print(f"Number of training images: {len(dataset)}")
    
    # function to visualize a single sample
    def visualize_sample(image, target):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        #image=image.permute(2,1,0).cpu().numpy()
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        for box_num in range(len(target['boxes'])):
            box = target['boxes'][box_num]
            label = CLASSES[target['labels'][box_num]]
            bbox = Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(bbox)
            ax.text(box[0], box[1]-5, label, fontsize=12, color='r')

        plt.show()
    NUM_SAMPLES_TO_VISUALIZE = 5
    for i in range(NUM_SAMPLES_TO_VISUALIZE):
        image, target = dataset[i]
        visualize_sample(image, target)
