import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn.pytorch as dglnn
import time
from conv.addsageconv import SAGEConv
class SAGE(nn.Module):
    """
    In fact, the SAGE can contain other conv than SAGEconv, just use its "block" method
    """
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 n_etypes,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, 'mean'))
        #self.layers.append(dglnn.GatedGraphConv(n_hidden,n_classes,n_steps = 3, n_etypes = n_etypes))
        #self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'ginmean'))
        #self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'cheb'))
        self.layers.append(SAGEConv(n_hidden, n_classes, 'mean'))
        #self.layers.append(dglnn.GATConv(n_hidden,n_classes,num_heads=3,allow_zero_in_degree=True))
        #GAT暂时还不行,因为GAT就是针对全图的
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, device,batch_size=32,num_workers=4):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1) # 对全图进行val
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                th.arange(g.num_nodes()),
                sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers)

            #for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0]
                block = block.int().to(device) # why?
                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y # nfeats
        return y

