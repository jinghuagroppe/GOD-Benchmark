import dgl as dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

def validate_model(model, dataset):

    model.eval()  
    
    valid_losses = []
    dataset_size = len(dataset)

    for i in range(dataset_size):
        
        sample = dataset.iloc[i]     
        pred = model(sample.nodes2id.to(device), sample.dglG.to(device), step_min, step_max)
  
        anchors = sample.anchors.to(device)
        loss = loss_cls_ious(pred, anchors)
     
        valid_losses.append(loss.item())

    return valid_losses
