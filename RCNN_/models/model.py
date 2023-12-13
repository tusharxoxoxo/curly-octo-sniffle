from data_prepration.utils import device
from torchvision import transforms, models, datasets
import torch.nn as nn
import torch
from torch_snippets import *

def model_extender():
    vgg_backbone = models.vgg16(pretrained=True)
    vgg_backbone.classifier = nn.Sequential()
    for param in vgg_backbone.parameters():
        param.requires_grad = False
    vgg_backbone.eval().to(device)
    return vgg_backbone
class RCNN(nn.Module):
    def __init__(self,label2target):
        super().__init__()
        feature_dim = 25088
        self.backbone = model_extender()
        self.cls_score = nn.Linear(feature_dim, len(label2target))
        self.bbox = nn.Sequential(
              nn.Linear(feature_dim, 512),
              nn.ReLU(),
              nn.Linear(512, 4),
              nn.Tanh(),
            )
        self.cel = nn.CrossEntropyLoss()
        self.sl1 = nn.L1Loss()
    def forward(self, input):
        feat = self.backbone(input)
        cls_score = self.cls_score(feat)
        bbox = self.bbox(feat)
        return cls_score, bbox
    def calc_loss(self, probs, _deltas, labels, deltas):
        detection_loss = self.cel(probs, labels)
        ixs, = torch.where(labels != 0)
        _deltas = _deltas[ixs]
        deltas = deltas[ixs]
        self.lmb = 10.0
        if len(ixs) > 0:
            regression_loss = self.sl1(_deltas, deltas)
            return detection_loss + self.lmb * regression_loss, detection_loss.detach(), regression_loss.detach()
        else:
            regression_loss = 0
            return detection_loss + self.lmb * regression_loss, detection_loss.detach(), regression_loss