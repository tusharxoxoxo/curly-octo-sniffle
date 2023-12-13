import glob
import numpy as np
import tqdm
import pandas as pd
import cv2
import os 
from config import CLASSES,TRAIN_DIR,VALID_DIR
from xml.etree import ElementTree as et
def csv_extracter(dir_path):
     image_paths = glob.glob(f"{dir_path}/*.jpg")
     all_images = [image_path.split('/')[-1] for image_path in image_paths]
     all_images = sorted(all_images)
     entities=[]
     for img in tqdm(all_images):
        image_name=os.path.join(dir_path,img)
        image=cv2.imread(image_name)
        width=image.shape[1]
        height=image.shape[0]
        annot_filename = img[:-4] + '.xml'
        annot_file_path = os.path.join(dir_path, annot_filename)
        tree = et.parse(annot_file_path)
        root = tree.getroot()
        # box coordinates for xml files are extracted and corrected for image size given
        for member in root.findall('object'):
            # map the current object name to `classes` list to get...
            # ... the label index and append to `labels` list
            labels=CLASSES.index(member.find('name').text)
            
            # xmin = left corner x-coordinates
            xmin = int(member.find('bndbox').find('xmin').text)
            # xmax = right corner x-coordinates
            xmax = int(member.find('bndbox').find('xmax').text)
            # ymin = left corner y-coordinates
            ymin = int(member.find('bndbox').find('ymin').text)
            # ymax = right corner y-coordinates
            ymax = int(member.find('bndbox').find('ymax').text)
            entities.append([image_name,img,width,height,labels,xmin,ymin,xmax,ymax])
     columns_name=['image_name','img','width','height','labels','xmin','ymin','xmax','ymax']
     return pd.DataFrame(entities,columns=columns_name)
train_data=csv_extracter(TRAIN_DIR)
valid_data=csv_extracter(VALID_DIR)
ref_ymax=train_data[train_data['ymax']>416]
ref_xmax=train_data[train_data['xmax']>416]
def delete(df1,df2):
    # create a boolean mask based on the values in df2
    mask = df1.isin(df2.to_dict('list')).all(axis=1)

    # filter df1 to keep only the rows not in df2
    df1 = df1[~mask]
    return df1
train_data=delete(train_data,ref_ymax)
train_data=delete(train_data,ref_xmax)
val_ymax=valid_data[valid_data['ymax']>416]
val_xmax=valid_data[valid_data['xmax']>416]
valid_data=delete(valid_data,val_ymax)
valid_data=delete(valid_data,val_xmax)
train_data.reset_index(inplace=True)
valid_data.reset_index(inplace=True)
train_data.to_csv('train_data',index=None)
valid_data.to_csv('valid_data',index=None)