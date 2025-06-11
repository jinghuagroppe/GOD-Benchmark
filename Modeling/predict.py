import dgl as dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

def predict(model, dataset):

    model.eval()  

    cls_actuals = list()
    cls_probs = list()
    
    dataset_size = len(dataset)
    
    for k in range(dataset_size):              
        sample = dataset.iloc[k] 
        pred = model(sample.nodes2id.to(device), sample.dglG.to(device), step_min, step_max)
        
        cls_actuals.append(sample.anchors)
        pred = pred.cpu().detach()
        cls_probs.append(pred)
    
    return cls_actuals, cls_probs