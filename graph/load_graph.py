import dgl
import torch as th
from torch.utils.data import DataLoader
from .coarsening import coarsen

from .coordinate import get_coordinates,cheb_z2_dist
from .grid_graph import grid_graph
from torchvision import datasets, transforms
from dgl.data import register_data_args, load_data

def load_ordinary_graph(name):
    if name == 'reddit':
        dataset = dgl.data.RedditDataset()
        g=dataset[0]
        return g,dataset.num_classes
    elif name == 'citeseer':
        dataset = dgl.data.CiteseerGraphDataset()
        g=dataset[0]
        return g,dataset.num_classes
    elif name == 'cora':
        dataset = dgl.data.CoraGraphDataset()
        g=dataset[0]
        return g,dataset.num_classes
    elif name == 'pubmed':
        dataset = dgl.data.PubmedGraphDataset()
        g=dataset[0]
        return g,dataset.num_classes
    else:
        raise Exception('unknown dataset')
    


def load_ogb(name):
    from ogb.nodeproppred import DglNodePropPredDataset

    print('load', name)
    data = DglNodePropPredDataset(name=name)
    print('finish loading', name)
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]
    labels = labels[:, 0]

    graph.ndata['features'] = graph.ndata['feat']
    graph.ndata['labels'] = labels
    in_feats = graph.ndata['features'].shape[1]
    num_labels = len(th.unique(labels[th.logical_not(th.isnan(labels))]))

    # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    train_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    train_mask[train_nid] = True
    val_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    val_mask[val_nid] = True
    test_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    test_mask[test_nid] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    print('finish constructing', name)
    return graph, num_labels


def load_mnist_cheb(pool_size=2,grid_side=28, number_edges=8, metric='euclidean', coarsening_levels=4, batch_size=100):
    """
    TODO
    return three generators -- train_loader, val_loader and test_loaderfor cheb to train
    """


    A = grid_graph(grid_side, number_edges, metric)
    # grid 好的graph送给coarsen返回每一层的稀疏矩阵Weight
    L, perm = coarsen(A, coarsening_levels)
    g_arr = [dgl.from_scipy(csr) for csr in L]

    """
    TODO 
    Different batcher for different pool_size
    """
    
    def batcher4(batch):
        # pool size=4, if pool_size=2:coarsening_levels + 1
        g_batch = [[] for _ in range(coarsening_levels//2 + 1)]
        x_batch = []
        y_batch = []
        for x, y in batch:
            x = th.cat([x.view(-1), x.new_zeros(len(perm) - grid_side ** 2)], 0)
            x = x[perm]
            x_batch.append(x)
            y_batch.append(y)
            # pool size=4, if pool_size=2:coarsening_levels + 1
            for i in range(coarsening_levels + 1):
                if i%2 ==0:
                    g_batch[i//2].append(g_arr[i]) 
        """
        print("mnist.py batcher x_batch :",len(x_batch),x_batch[0].shape)
        print("mnist.py batcher y_batch :",len(y_batch),type(y_batch[0]))
        print("mnist.py batcher g_batch g_batch[0]:",len(g_batch),len(g_batch[0]),g_batch[0][0].num_nodes())
        for i in range(len(g_batch)):
            print("g_batch[%d][0] num_nodes :"%(i),g_batch[i][0].num_nodes()) 
        """
        x_batch = th.cat(x_batch).unsqueeze(-1)
        y_batch = th.LongTensor(y_batch)  # y_batch就是label
        g_batch = [dgl.batch(g) for g in g_batch]
        return g_batch, x_batch, y_batch
    def batcher2(batch):
        # pool size=4, if pool_size=2:coarsening_levels + 1
        g_batch = [[] for _ in range(coarsening_levels + 1)]
        x_batch = []
        y_batch = []
        for x, y in batch:
            x = th.cat([x.view(-1), x.new_zeros(len(perm) - grid_side ** 2)], 0)
            x = x[perm]
            x_batch.append(x)
            y_batch.append(y)
            # pool size=4, if pool_size=2:coarsening_levels + 1
            for i in range(coarsening_levels + 1):
                g_batch[i].append(g_arr[i])
        """
        print("mnist.py batcher x_batch :",len(x_batch),x_batch[0].shape)
        print("mnist.py batcher y_batch :",len(y_batch),type(y_batch[0]))
        print("mnist.py batcher g_batch g_batch[0]:",len(g_batch),len(g_batch[0]),g_batch[0][0].num_nodes())
        for i in range(len(g_batch)):
            print("g_batch[%d][0] num_nodes :"%(i),g_batch[i][0].num_nodes()) 
        """
        x_batch = th.cat(x_batch).unsqueeze(-1)
        y_batch = th.LongTensor(y_batch)  # y_batch就是label
        g_batch = [dgl.batch(g) for g in g_batch]
        return g_batch, x_batch, y_batch


    # g_arr = g_arr[::2] # pooling size = 4
    # perm = perm[::2] # pooling size = 4
    # coarsening_levels = coarsening_levels//2 # pooling size = 4
    # Compute x,y coordinate for nodes in the graph
    coordinate_arr = get_coordinates(g_arr, grid_side, coarsening_levels, perm)
    for g, coordinate_arr in zip(g_arr, coordinate_arr):
        # arr是array的简写
        g.ndata['xy'] = coordinate_arr
        g.apply_edges(cheb_z2_dist)
        # g 本身的结构和是什么数字的照片（用x_batch描述）无关，但是不同层的graph结构不同
    trainset = datasets.MNIST(root='.', train=True,download=True, transform=transforms.ToTensor())
    testset = datasets.MNIST(root='.', train=False,download=True, transform=transforms.ToTensor())
    train_pictures = 50000
    val_pictures = 60000-train_pictures
    trainset,valset = th.utils.data.random_split(trainset,[train_pictures,val_pictures])
    print("trainset :",len(trainset))
    print("valset :",len(valset))
    if pool_size == 2:
        train_loader = DataLoader(trainset,
                          batch_size=batch_size,
                          shuffle=True,
                          collate_fn=batcher2,
                          num_workers=6)
        val_loader = DataLoader(valset,
                          batch_size=batch_size,
                          shuffle=True,
                          collate_fn=batcher2,
                          num_workers=6)
        test_loader = DataLoader(testset,
                         batch_size=batch_size,
                         shuffle=False,
                         collate_fn=batcher2,
                         num_workers=6)
    elif pool_size == 4:
        train_loader = DataLoader(trainset,
                          batch_size=batch_size,
                          shuffle=True,
                          collate_fn=batcher4,
                          num_workers=6)
        val_loader = DataLoader(valset,
                          batch_size=batch_size,
                          shuffle=True,
                          collate_fn=batcher4,
                          num_workers=6)
        test_loader = DataLoader(testset,
                         batch_size=batch_size,
                         shuffle=False,
                         collate_fn=batcher4,
                         num_workers=6)        
    n_classes = 10
    in_feats = 1
    return train_loader,val_loader,test_loader,coarsening_levels,perm,n_classes,in_feats
