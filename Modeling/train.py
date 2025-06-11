import dgl as dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

def train_model(model, dataset, batch_size):

    model.train()  

    train_losses = []
    dataset_size = len(dataset)

    for i in range(dataset_size):
        sample = dataset.iloc[i]     
        pred = model(sample.nodes2id.to(device), sample.dglG.to(device), step_min, step_max)
        anchors = sample.anchors.to(device)
        loss = loss_cls_ious(pred, anchors)
        loss.backward()

        if ((i+1) % batch_size == 0) or (i+1 == dataset_size):
            optimizer.step()
            model.zero_grad()

        train_losses.append(loss.item())

    return model, train_losses