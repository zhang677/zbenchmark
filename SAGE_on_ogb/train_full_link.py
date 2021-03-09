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

from train_full import GraphSAGE
from load_graph import load_reddit, load_ogbl, inductive_split

class Link_Predictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(Link_Predictor, self).__init__()

        self.lin = nn.ModuleList()
        self.lin.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lin.append(nn.Linear(hidden_channels, hidden_channels))
        self.lin.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = nn.Dropout(dropout)

    def forward(self, h1, h2):
        h = h1 * h2
        for layer in self.lin[:-1]:
            h = layer(h)
            h = F.relu(h)
            h = self.dropout(h)
        h = self.lin[-1](h)
        return torch.sigmoid(h)

def train(model, predictor, optimizer, g, splitted_idx, device, batch_size):
    model.train()
    predictor.train()

    g = g.to(device)
    pos_train_edge = splitted_idx['train']['edge'].to(device)
    total_loss = total_examples = 0
    from tqdm import tqdm
    for perm in tqdm(DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True)):
        optimizer.zero_grad()
        h = model(g, g.ndata['feat'])
        
        edge = pos_train_edge[perm].t()

        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        edge = torch.randint(0, g.num_nodes(), edge.size(), dtype=torch.long,
                             device=device)
        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
        
    return total_loss / total_examples

def test(model, predictor, g, edge, device):
    model.eval()
    predictor.eval()

    g = g.to(device)
    h = model(g, g.ndata['feat'])
    pos_edge = edge['edge'].to(device).t()
    neg_edge = edge['edge_neg'].to(device).t()
    pos_pred = predictor(h[pos_edge[0]], h[pos_edge[1]])
    neg_pred = predictor(h[neg_edge[0]], h[neg_edge[1]])
    n_edges = pos_edge.shape[1] + neg_edge.shape[1]
    thres = 0.5
    acc = ((pos_pred >= thres).squeeze().sum() + (neg_pred < thres).squeeze().sum()) / n_edges
    return acc



def main(args):
    from load_graph import load_ogbl
    g, splitted_idx = load_ogbl(args.dataset)
    if args.dataset == 'ogbl-ppa':
        g.ndata['feat'] = g.ndata['feat'].float()
    

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        g = g.to(args.gpu)

    model = GraphSAGE(
        in_feats = g.ndata['feat'].shape[1], 
        n_hidden = args.n_hidden,
        n_classes = args.n_hidden,
        n_layers = args.n_layers, 
        activation = F.relu,
        dropout = args.dropout,
        aggregator_type = args.aggregator_type
    )

    predictor = Link_Predictor(
        in_channels = args.n_hidden, 
        hidden_channels = args.n_hidden, 
        out_channels = 1, 
        num_layers = args.n_layers, 
        dropout = args.dropout
    )
    
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
        loss = train(model, predictor, optimizer, g, splitted_idx, args.gpu, args.batch_size)
        if epoch > 3:
            print(epoch, loss)
        if epoch % 10 == 0:
            val_acc = test(model, predictor, g, splitted_idx['valid'], args.gpu)
            print('val acc:', val_acc.item())
    test_acc = test(model, predictor, g, splitted_idx['test'], args.gpu)
    print('test_acc:', test_acc.item())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphSAGE')
    register_data_args(parser)
    #parser.add_argument("--dataset", type=str, default='ogbl-collab', 
    #                    help='link prediction dataset')
    parser.add_argument("--batch-size", type=int, default=10000,
                        help="batch size")
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