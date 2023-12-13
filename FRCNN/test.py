
import torch
from config import CLASSES,DEVICE,OUT_DIR
from model import create_model
import glob
import cv2
import numpy as np

# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
NUM_CLASSES=len(CLASSES)
# load the best model and trained weights
model = create_model(num_classes=NUM_CLASSES)
checkpoint = torch.load('best_model (2).pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()
# directory where all the images are present
DIR_TEST = '/kaggle/input/weapon-data/test'
test_images = glob.glob(f"{DIR_TEST}/*.jpg")
print(f"Test instances: {len(test_images)}")
# define the detection threshold...
# ... any detection having score below this will be discarded
detection_threshold = 0.8
# to count the total number of images iterated through
frame_count = 0
# to keep adding the FPS for each image
total_fps = 0 
import matplotlib.pyplot as plt
# define a function to display an image with bounding boxes and labels
def display_image(image, boxes, labels):
    fig, ax = plt.subplots(1)
    # display the image
    ax.imshow(image)
    # draw bounding boxes and labels
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        bbox = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                              fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(bbox)
        ax.text(xmin, ymin, label, fontsize=12,
                bbox=dict(facecolor='red', alpha=0.5))
    # remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    plt.savefig(f"{OUT_DIR}/{image_name}", bbox_inches='tight')
    plt.show()

for i in range(len(test_images)):
    if i==30:
        break
    # get the image file name for saving output later on
    image_name = test_images[i].split('/')[-1].split('.')[0]
    image = cv2.imread(test_images[i])
    orig_image = image.copy()
    # BGR to RGB
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # make the pixel range between 0 and 1
    image /= 255.0
    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(np.float)
    # convert to tensor
    image = torch.tensor(image, dtype=torch.float).cuda()
    # add batch dimension
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model(image)   
    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # carry further only if there are detected boxes
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # filter out boxes according to `detection_threshold`
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        # get all the predicited class names
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        display_image(orig_image, draw_boxes, pred_classes)
    print(f"Image {i+1} done...")
    print('-'*50)

print('TEST PREDICTIONS COMPLETE')
