
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from conv.addchebconv import ChebConv
from dgl.nn.pytorch.glob import MaxPooling

class ChebNet2(nn.Module):
    def __init__(self,
                 k,
                 in_feats,
                 hiddens,
                 out_feats):
        super(ChebNet2, self).__init__()
        self.pool = nn.MaxPool1d(2,stride=2) 
        self.layers = nn.ModuleList()
        self.readout = MaxPooling()

        # Input layer
        self.layers.append(
            ChebConv(in_feats, hiddens[0], k))

        for i in range(1, len(hiddens)):
            self.layers.append(
                ChebConv(hiddens[i - 1], hiddens[i], k))

        self.cls = nn.Sequential(
            nn.Linear(hiddens[-1], out_feats),
            nn.LogSoftmax()
        )

    def forward(self, g_arr, feat):
        for i, (g, layer) in enumerate(zip(g_arr, self.layers)):
            #print("mnist.py cheb's forward g: ",g.num_nodes())
            """
            tmp = layer(g, feat, [2] * g.batch_size)
            print("mnist.py after layer: ",tmp.shape)
            tmp = tmp.transpose(-1, -2)
            print("after transpose : ",tmp.shape)
            tmp = tmp.unsqueeze(0)
            print("after unsqueezing : ",tmp.shape)
            tmp = self.pool(tmp)
            print("after pooling : ",tmp.shape)
            """
            #print("!!! i: ",i)      
            feat = self.pool(layer(g, feat).transpose(-1, -2).unsqueeze(0))\
                .squeeze(0).transpose(-1, -2) # lambda_max都设成2->, [2] * g.batch_size
            #print("mnist.py feat: ",feat.shape)
        #print("mnist.py batch_size : ",g_arr[-1].batch_size)
        tmp = self.readout(g_arr[-1], feat)
        #print("mnist.py cheb's forward after MaxPooling : ",tmp.shape)
        return self.cls(self.readout(g_arr[-1], feat))
        
class ChebNet4(nn.Module):
    def __init__(self,
                 k,
                 in_feats,
                 hiddens,
                 out_feats):
        super(ChebNet4, self).__init__()
        #self.pool = nn.MaxPool1d(2) 
        self.pool = nn.MaxPool1d(4,stride=4)
        self.layers = nn.ModuleList()
        self.readout = MaxPooling()

        # Input layer
        self.layers.append(
            ChebConv(in_feats, hiddens[0], k))

        for i in range(1, len(hiddens)):
            self.layers.append(
                ChebConv(hiddens[i - 1], hiddens[i], k))

        self.cls = nn.Sequential(
            nn.Linear(hiddens[-1], out_feats),
            nn.LogSoftmax()
        )

    def forward(self, g_arr, feat):
        g_arr = g_arr[::2] # pooling 4
        for i, (g, layer) in enumerate(zip(g_arr, self.layers)):
            #print("mnist.py cheb's forward g: ",g.num_nodes())
            """
            tmp = layer(g, feat, [2] * g.batch_size)
            print("mnist.py after layer: ",tmp.shape)
            tmp = tmp.transpose(-1, -2)
            print("after transpose : ",tmp.shape)
            tmp = tmp.unsqueeze(0)
            print("after unsqueezing : ",tmp.shape)
            tmp = self.pool(tmp)
            print("after pooling : ",tmp.shape)
            """
            #print("!!! i: ",i)      
            feat = self.pool(layer(g, feat).transpose(-1, -2).unsqueeze(0))\
                .squeeze(0).transpose(-1, -2) # lambda_max都设成2->, [2] * g.batch_size
            #print("mnist.py feat: ",feat.shape)
        #print("mnist.py batch_size : ",g_arr[-1].batch_size)
        tmp = self.readout(g_arr[-1], feat)
        #print("mnist.py cheb's forward after MaxPooling : ",tmp.shape)
        return self.cls(self.readout(g_arr[-1], feat))

class Cheb3(nn.Module):
    """
    only for homogeneous, compared with sampling([10,15]) which is also "2-layer"
    compute the whole graph twice?
    """
    def __init__(self,in_feature,hidden_size,out_feature):
        super().__init__()
        self.conv1 = ChebConv(in_feature,hidden_size,k=2)
        self.conv2 = ChebConv(hidden_size,out_feature,k=2)
    def forward(self,graph,feat):
        h = self.conv1(graph,feat)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h