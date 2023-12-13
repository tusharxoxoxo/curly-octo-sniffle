from torch.utils.data import TensorDataset, DataLoader
from data_prepration.Data_prepration import RCNNDataset
from data_prepration.data_preprocessing import processing
from data_prepration.utils import target_available
from torch_snippets import *

def data_loaders(N,csv_dir,image_dir):
    FPATHS, GTBBS, CLSS, DELTAS, ROIS=processing(N,csv_dir,image_dir)
    targets,label2target,target2label,background_class=target_available(CLSS)
    print(label2target)
    n_train = 9*len(FPATHS)//10
    print(f'\nTrain size : {n_train}')
    train_ds = RCNNDataset(FPATHS[:n_train], 
                            ROIS[:n_train], 
                            CLSS[:n_train], 
                            DELTAS[:n_train], 
                            GTBBS[:n_train],
                            label2target)
    test_ds = RCNNDataset(FPATHS[n_train:], 
                            ROIS[n_train:], 
                            CLSS[n_train:], 
                            DELTAS[n_train:],
                            GTBBS[n_train:],
                            label2target)

    train_loader = DataLoader(train_ds,
                              batch_size=2,
                              collate_fn=train_ds.collate_fn,
                              drop_last=True)
    test_loader = DataLoader(test_ds, 
                             batch_size=2,
                             collate_fn=test_ds.collate_fn, 
                             drop_last=True)
    print('Data Loading Complete : -')
    return train_loader,test_loader,targets,label2target,target2label,background_class



