import dgl as dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

class god(nn.Module):
    def __init__(self, n_props=3, 
                 pretrained_embedding=None, freeze_embedding=False,
                 vocab_size=None, embed_dim=None,
                 n_classes=None,
                 n_steps=None,
                ):   
        super().__init__()

        self.feat_dim = n_props*embed_dim
        self.n_classes = n_classes

        if pretrained_embedding is not None:
            self.embed = nn.Embedding.from_pretrained(
                pretrained_embedding, freeze=freeze_embedding)
        else:
            self.embed = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embed_dim,
                                      padding_idx=None)  
            
        self.gconv = ggc.GatedGraphConv(
            in_feats=self.feat_dim, out_feats=self.feat_dim, n_etypes=1) 
     
        kernel_w = self.feat_dim
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=n_classes*2,  
                      kernel_size=(1, kernel_w), stride=1, padding=0)

            
    def forward(self, prop_ids, dglG, step_min, step_max):   
        feat = self.embed(prop_ids).float().to(device)        
        n_nodes, n_props, embed_dim = feat.shape
        feat = feat.view(n_nodes, self.feat_dim)
        _, feat_steps = self.gconv(dglG, feat, step_max) 
        feat_steps = feat_steps[step_min-1:step_max].to(device)
        feat_steps = feat_steps.unsqueeze(1)

        conv2d_out = self.conv2d(feat_steps)
        conv2d_out = conv2d_out.permute(2,0,1,3).squeeze(-1) 
        conv2d_out = conv2d_out.reshape(-1,self.n_classes*2)
        out = torch.sigmoid(conv2d_out)   
        
        return out   