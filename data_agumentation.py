from imgaug import augmenters as iaa
import pandas as pd
import os 
import imageio
import re
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
# Function to convert bounding box image into DataFrame 
# Define all the Augmentations you want to apply to your dataset
# We're setting random `n` agumentations to 2. 
image_augmentations = iaa.SomeOf( 2,
    [                                 
    # Scale the Images
    iaa.Affine(scale=(0.5, 1.5)),
 
    # Rotate the Images
    iaa.Affine(rotate=(-60, 60)),
 
    # Shift the Image
    iaa.Affine(translate_percent={"x":(-0.3, 0.3),"y":(-0.3, 0.3)}),
 
    # Flip the Image
    iaa.Fliplr(1),
 
    # Increase or decrease the brightness
    iaa.Multiply((0.5, 1.5)),
 
    # Add Gaussian Blur
    iaa.GaussianBlur(sigma=(1.0, 3.0)),
     
    # Add Gaussian Noise
    iaa.AdditiveGaussianNoise(scale=(0.03*255, 0.05*255))
 
])
def bounding_boxes_to_df(bounding_boxes_object):
 
    # Convert Bounding Boxes Object to Array
    bounding_boxes_array = bounding_boxes_object.to_xyxy_array()
     
    # Convert the array into DataFrame
    df_bounding_boxes = pd.DataFrame(bounding_boxes_array,columns=['xmin', 'ymin', 'xmax', 'ymax'])
     
    # Return the DataFrame
    return df_bounding_boxes
def image_aug(df, images_path, aug_images_path, augmentor=image_augmentations, multiple=1):
     
    # Fill this DataFrame with image attributes
    augmentations_df = pd.DataFrame(
        columns=['filename','width','height','class', 'xmin', 'ymin', 'xmax','ymax'])
    # Group the data by filenames
    print(df.shape)
    grouped_df = df.groupby('filename')
 
    # Create the directory for all augmentated images
    if not os.path.exists(aug_images_path):
      os.mkdir(aug_images_path)
 
    # Create directories for each class of augmentated images
    for folder in df['class'].unique():
      if not os.path.exists(os.path.join(aug_images_path, folder)):
        os.mkdir(os.path.join(aug_images_path, folder))
 
    for i in range(multiple):
       
      # Post Fix we add to the each different augmentation of one image
      image_postfix = str(i)
 
      # Loop to perform the augmentations
      for filename in df['filename'].unique():
 
        augmented_path = os.path.join(aug_images_path, filename)+image_postfix+'.jpg'
 
        # Take one image at a time with its information
        single_image = grouped_df.get_group(filename)
        single_image = single_image.reset_index()
        single_image = single_image.drop(['index'], axis=1)   
         
        # Read the image
        image = imageio.imread(os.path.join(images_path, filename))
 
        # Get bounding box
        bounding_box_array = single_image.drop(['filename', 'width', 'height','class'], axis=1).values
 
        # Give the bounding box to imgaug library
        bounding_box = BoundingBoxesOnImage.from_xyxy_array(bounding_box_array, shape=image.shape)
 
        # Perform random 2 Augmentations
        image_aug, bounding_box_aug = augmentor(image=image,bounding_boxes=bounding_box)
         
        # Discard the the bounding box going out the image completely   
        bounding_box_aug = bounding_box_aug.remove_out_of_image()
 
        # Clip the bounding box that are only partially out of th image
        bounding_box_aug = bounding_box_aug.clip_out_of_image()
 
        # Get rid of the the image if bounding box was discarded  
        if re.findall('Image...', str(bounding_box_aug)) == ['Image([]']:
            pass
         
        else:
         
          # Create the augmented image file
          imageio.imwrite(augmented_path, image_aug) 
 
          # Update the image width and height after augmentation
          info_df = single_image.drop(['xmin', 'ymin', 'xmax', 'ymax'], axis=1)    
          for index, _ in info_df.iterrows():
              info_df.at[index, 'width'] = image_aug.shape[1]
              info_df.at[index, 'height'] = image_aug.shape[0]
 
          # Add the prefix to each image to differentiate if required
          info_df['filename'] = info_df['filename'].apply(lambda x: x + image_postfix + '.jpg')
 
          # Create the augmented bounding boxes dataframe 
          bounding_box_df = bounding_boxes_to_df(bounding_box_aug)
 
          # Concatenate the filenames, height, width and bounding boxes 
          aug_df = pd.concat([info_df, bounding_box_df], axis=1)
 
          # Add all the information to augmentations_df we initialized above
          augmentations_df = pd.concat([augmentations_df, aug_df])            
       
    # Remove index
    augmentations_df = augmentations_df.reset_index()
    augmentations_df = augmentations_df.drop(['index'], axis=1)
 
    # Return the Dataframe
    return augmentations_df
 
