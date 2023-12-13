import torch
BATCH_SIZE=8
RESIZE_TO=416
NUM_EPOCHS=10
NUM_WORKERS=2
DEVICE=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
CLASSES=['Marked Null','gun','knife']
NUM_CLASSES=len(CLASSES)
VISUALIZATION_TRANSFORMED_IMAGES=True
TRAIN_DIR=''
VALID_DIR=''
OUT_DIR='Threat_detection_using-Deep_learning/FRCNN/'