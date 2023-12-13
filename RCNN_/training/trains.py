from data_prepration.utils import decode
from data_prepration.utils import device
import torch
from models import model
from torch_snippets import *
from data_prepration.data_loader import data_loaders
from training.train_batchs import train_batch,validate_batch,save_model

def train(csv_dir,image_dir,main_dir,N,n_epochs,save=False):
    train_loader,test_loader,targets,label2target,target2label,background_class=data_loaders(N,csv_dir,image_dir)
    rcnn = model.RCNN(label2target).to(device)
    criterion = rcnn.calc_loss
    optimizer = torch.optim.SGD(rcnn.parameters(), lr=1e-3)
    log = Report(n_epochs)
    for epoch in range(n_epochs):

        _n = len(train_loader)
        for ix, inputs in enumerate(train_loader):
            loss, loc_loss, regr_loss, accs = train_batch(inputs, rcnn, 
                                                        optimizer, criterion)
            pos = (epoch + (ix+1)/_n)
            log.record(pos, trn_loss=loss.item(), trn_loc_loss=loc_loss, 
                    trn_regr_loss=regr_loss, 
                    trn_acc=accs.mean(), end='\r')
            
        _n = len(test_loader)
        for ix,inputs in enumerate(test_loader):
            _clss, _deltas, loss, \
            loc_loss, regr_loss, accs = validate_batch(inputs, 
                                                    rcnn, criterion)
            pos = (epoch + (ix+1)/_n)
            log.record(pos, val_loss=loss.item(), val_loc_loss=loc_loss, 
                    val_regr_loss=regr_loss, 
                    val_acc=accs.mean(), end='\r')
        if save and epoch==n_epochs-1:
            save_model(rcnn,main_dir)
    print('Training Complete--')
    return target2label,background_class


    

   