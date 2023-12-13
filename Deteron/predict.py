import os 
import matplotlib.pyplot as plt
# some utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")
# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg=get_cfg()
#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "deteron/model_final.pth")  # path to the model we just trained
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
# predictor = DefaultPredictor(cfg)
# configuration of the trained model
cfg.merge_from_file("config_knives.yml") #configuration file
cfg.MODEL.DEVICE='cpu'
cfg.MODEL.WEIGHTS = ("deteron/model_final.pth") #path of the trained model
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.7 # For RETINANET Model
predictor1 = DefaultPredictor(cfg)
from detectron2.utils.visualizer import ColorMode
path='images_test'
list_1=os.listdir(path)
for e,img in enumerate(list_1):
    if e<=40:
        continue
    elif e>=20:
        break
    img_path=os.path.join(path,img)
    im1 = plt.imread(img_path)
    # im1 = cv2.imread('/content/g2.jpg')
    #im1 = cv2.imread('/content/test5.jpg')
    outputs1 = predictor1(im1)
    v = Visualizer(im1[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    threshed=[]
    for i in range(0, len(outputs1['instances'].scores)):
        if outputs1['instances'].scores[i]>0.75:
            threshed.append(i)
    v = v.draw_instance_predictions(outputs1["instances"][threshed].to("cpu"))
    plt.figure(figsize=(20,20))
    plt.imshow(v.get_image()[:, :, ::-1])
    plt.savefig(str(int(e))+'.png')
    plt.show()


