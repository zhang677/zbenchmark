import argparse
import time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.nn.pytorch.conv import SAGEConv
from dgl.nn.pytorch.glob import AvgPooling

from train_full import GraphSAGE
from load_graph import load_reddit, load_ogbl, inductive_split
 
class Graph_Predictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, aggr_type):
        super(Graph_Predictor, self).__init__()
        self.conv = nn.ModuleList()
        self.conv.append(SAGEConv(in_channels, hidden_channels, aggr_type))
        for _ in range(num_layers - 1):
            self.conv.append(SAGEConv(hidden_channels, hidden_channels, aggr_type))

        self.lin = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.lin.append(nn.Linear(hidden_channels, hidden_channels))
        self.lin.append(nn.Linear(hidden_channels, out_channels))

        self.pooling = AvgPooling()

        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, h):
        h = self.dropout(h)
        for layer in self.conv[:-1]:
            h = layer(graph, h)
            h = F.relu(h)
            h = self.dropout(h)
        h = self.conv[-1](graph, h)

        for layer in self.lin[:-1]:
            h = layer(h)
            h = F.relu(h)
            h = self.dropout(h)
        h = self.lin[-1](h)
        
        h = self.pooling(graph, h)
        #h = F.softmax(h, 0)
        return h

class Predictor(nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, dropout):
        super(Predictor, self).__init__()

        self.lin = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.lin.append(nn.Linear(hidden_channels, hidden_channels))
        self.lin.append(nn.Linear(hidden_channels, out_channels))

        self.pooling = AvgPooling()

        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, h):
        for layer in self.lin[:-1]:
            h = layer(h)
            h = F.relu(h)
            h = self.dropout(h)
        h = self.lin[-1](h)
        
        h = self.pooling(graph, h)
        #h = F.softmax(h, 0)
        return h

def train(model, predictor, optimizer, train_loader, device):
    model.train()
    predictor.train()

    total_loss = total_examples = 0
    for graph, label in train_loader:
        graph = graph.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        feats = graph.ndata['feat'].to(device)
        h = model(graph, feats)
        logits = predictor(graph, h)
        try:
            loss = F.cross_entropy(logits, label.view(-1))
        except:
            from IPython import embed; embed()
        loss.backward()
        optimizer.step()
        num = label.size(0)
        total_loss += num * loss.item()
        total_examples += num
    return total_loss / total_examples






def test(model, predictor, loader, device):
    model.eval()
    predictor.eval()
    correct = num = 0
    for graph, label in loader:
        graph = graph.to(device)
        label = label.to(device)
        feats = graph.ndata['feat'].to(device)
        logits = predictor(graph, model(graph, feats))
        correct += (logits.max(axis = 1)[1] == label.view(-1)).sum()
        num += label.size(0)
    return correct / num




def main(args):
    from load_graph import load_ogbg
    train_loader, valid_loader, test_loader, in_channels, out_channels = load_ogbg(args.dataset, args.gpu)
    #from IPython import embed; embed()
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)

    model = GraphSAGE(
        in_channels, 
        args.n_hidden,
        args.n_hidden,
        args.n_layers, 
        F.relu,
        args.dropout,
        args.aggregator_type
    )

    predictor = Predictor(
        hidden_channels = args.n_hidden,
        out_channels = int(out_channels),
        num_layers = args.n_layers, 
        dropout = args.dropout,
    )

    #from IPython import embed; embed()
    if cuda:
        model.to(args.gpu)
        predictor.to(args.gpu)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )

    from tqdm import tqdm
    for epoch in tqdm(range(args.n_epochs)):
        loss = train(model, predictor, optimizer, train_loader, args.gpu)
        if epoch > 3:
            print(epoch, loss)
        if epoch % 10 == 0:
            val_acc = test(model, predictor, valid_loader, args.gpu)
            print('val acc:', val_acc.item())
    test_acc = test(model, predictor, test_loader, args.gpu)
    print('test_acc:', test_acc.item())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphSAGE')
    register_data_args(parser)
    #parser.add_argument("--dataset", type=str, default='ogbl-collab', 
    #                    help='link prediction dataset')
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=50,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of gnn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--aggregator-type", type=str, default="gcn",
                        help="Aggregator type: mean/gcn/pool/lstm")
    args = parser.parse_args()
    print(args)

    main(args)

    #support: ppa, ddi, citation2(OOM)