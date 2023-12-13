from data_prepration.utils import read_data,extract_candidates,extract_iou
import cv2
import numpy as np
from data_prepration.open_images import OpenImages
from torch_snippets import *

def processing(N,csv_dir,image_dir):
        FPATHS, GTBBS, CLSS, DELTAS, ROIS, IOUS = [], [], [], [], [], []
        data=read_data(csv_dir)
        print(data.shape)
        ds=OpenImages(data,image_dir)
        im, bbs, clss, _ = ds[9]
        show(im, bbs=bbs, texts=clss, sz=10)

        for ix, (im, bbs, labels, fpath) in enumerate(ds):
            if(ix==N):
                break
            H, W, _ = im.shape
            candidates = extract_candidates(im)
            candidates = np.array([(x,y,x+w,y+h) for x,y,w,h in candidates])
            ious, rois, clss, deltas = [], [], [], []
            ious = np.array([[extract_iou(candidate, _bb_) for candidate in candidates] for _bb_ in bbs]).T
            for jx, candidate in enumerate(candidates):
                cx,cy,cX,cY = candidate
                candidate_ious = ious[jx]
                best_iou_at = np.argmax(candidate_ious)
                best_iou = candidate_ious[best_iou_at]
                best_bb = _x,_y,_X,_Y = bbs[best_iou_at]
                if best_iou > 0.3: clss.append(labels[best_iou_at])
                else : clss.append('background')
                delta = np.array([_x-cx, _y-cy, _X-cX, _Y-cY]) / np.array([W,H,W,H])
                deltas.append(delta)
                rois.append(candidate / np.array([W,H,W,H]))
            FPATHS.append(fpath)
            IOUS.append(ious)
            ROIS.append(rois)
            CLSS.append(clss)
            DELTAS.append(deltas)
            GTBBS.append(bbs)
        FPATHS = [f'{image_dir}/{stem(f)}.jpg' for f in FPATHS] 
        FPATHS, GTBBS, CLSS, DELTAS, ROIS = [item for item in [FPATHS, GTBBS, CLSS, DELTAS, ROIS]]
        return FPATHS, GTBBS, CLSS, DELTAS, ROIS
