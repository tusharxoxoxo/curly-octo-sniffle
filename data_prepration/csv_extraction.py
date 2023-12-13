import os 
import numpy as np
import pandas as pd 
import shutil
import glob
import xml.etree.ElementTree as xt
from sklearn import preprocessing
from data_agumentation import image_aug
from tqdm import tqdm
data=os.listdir('data')
#path1 = sorted(glob('/kaggle/input/weapon-dataset/train/*.xml'))# get all elements in path with xml extension

def exteract_xml_contents(annot_directory):

    import xml.etree.ElementTree as ET

    # Parse the XML file
    tree = ET.parse(annot_directory)

    # Get the root element
    root = tree.getroot()

    # Extract the information
    filename = root.find('filename').text
    size = root.find('size')
    width = size.find('width').text
    height = size.find('height').text
    object = root.find('object')
    if object==None:
        return -1
    name = object.find('name').text
    bndbox = object.find('bndbox')
    xmin = bndbox.find('xmin').text
    ymin = bndbox.find('ymin').text
    xmax = bndbox.find('xmax').text
    ymax = bndbox.find('ymax').text

    return filename,width,height,name,xmin,ymin,xmax,ymax
