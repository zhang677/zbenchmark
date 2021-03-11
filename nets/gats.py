"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import torch as th
import torch.nn as nn
import dgl.function as fn
from dgl.nn import GATConv
import dgl



class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1) # h.shape[-1] = num_heads*out_feats
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits

class GATBlock(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GATBlock, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.n_hidden = num_hidden
        self.heads = heads
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, blocks ,inputs):
        h = inputs
        for l, (layer, block) in enumerate(zip(self.gat_layers, blocks)):
            if l!=self.num_layers:
                h = layer(block, h).flatten(1) # h.shape[-1] = num_heads*out_feats
            # output projection
            else:
                h = layer(block, h).mean(1)
            #print('h SHAPE: ',h.shape)
        return h
    def inference(self, g, x, val_nid,device,batch_size=32,num_workers=4):
        """
        The inference is different from the sageconv because the output of layers in 
        GAT has multiple heads. Therefore, the whole graph can't be updated layer by
        layer. Instead, we sample the graph that the update of seeds needs, and then 
        update layer by layer
        INPUT:
        g: the whole graph
        x: the whole feature
        """
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.num_layers+1) # 对全图进行val
        dataloader = dgl.dataloading.NodeDataLoader(
            g,
            val_nid,
            sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers)
        pred = []
        #for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
        for step, (input_nodes , output_nodes , blocks) in enumerate(dataloader):
            h = x[input_nodes].to(device)
            blocks = [block.int().to(device) for block in blocks]
            for l, (layer, block) in enumerate(zip(self.gat_layers, blocks)):
                if l!=self.num_layers:
                    h = layer(block, h).flatten(1) # h.shape[-1] = num_heads*out_feats
                # output projection
                else:
                    h = layer(block, h).mean(1)
            pred.append(h)
        return th.cat(pred,dim=0)