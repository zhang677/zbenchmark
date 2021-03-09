import dgl
import torch as th

def load_reddit():
    from dgl.data import RedditDataset

    # load reddit data
    data = RedditDataset(self_loop=True)
    g = data[0]
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']
    return g, data.num_labels

def load_ogb(name, device = th.device('cpu'), root = '/home/eva_share_users/zhuyu'):
    from ogb.nodeproppred import DglNodePropPredDataset

    print('load', name)
    data = DglNodePropPredDataset(name = name, root = root)
    print('finish loading', name)
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]
    if name == 'ogbn-mag':
        labels = labels['paper'][:, 0]
        graph = graph[('paper', 'cites', 'paper')]
    else:
        labels = labels[:, 0]

    if name == 'ogbn-proteins':
        try:
            graph.ndata['features'] = th.load('H_proteins.pt')
            graph.ndata['labels'] = graph.ndata['species']
        except:
            print('Didn\'t find feature matrix, starting to calculate...')
            graph = graph.to(device)
            graph.ndata['labels'] = graph.ndata['species']
            n = graph.num_nodes()
            H = th.zeros((n, 8)).to(device)
            g = graph.edata['feat'].to(device)
            edge = graph.edges()[1].to(device)
            mask = th.zeros((n,)).to(device)
            from tqdm import tqdm
            for i in tqdm(range(n)):
                mask = th.eq(edge, i)
                H[i, :] += th.matmul(mask.float(), g)
                H[i, :] /= graph.in_degrees(i)
            graph.ndata['features'] = H
            th.save(th.tensor(H.tolist()), 'H_proteins.pt')
            print('Feature matrix saved as H_proteins.pt')
    else:
        graph.ndata['features'] = graph.ndata['feat']

    graph.ndata['labels'] = labels
    in_feats = graph.ndata['features'].shape[1]
    num_labels = len(th.unique(labels[th.logical_not(th.isnan(labels))]))

        # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    train_mask = th.zeros((graph.num_nodes(),), dtype=th.bool)
    if name == 'ogbn-mag':
        train_nid = train_nid['paper']
        val_nid = val_nid['paper']
        test_nid = test_nid['paper']
    train_mask[train_nid] = True
    val_mask = th.zeros((graph.num_nodes(),), dtype=th.bool)
    val_mask[val_nid] = True
    test_mask = th.zeros((graph.num_nodes(),), dtype=th.bool)
    test_mask[test_nid] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    print('finish constructing', name)
    #print(graph)
    #from IPython import embed; embed()
    return graph, num_labels

def load_ogbl(name, device = th.device('cpu'), root = '/home/eva_share_users/zhuyu'):
    from ogb.linkproppred import DglLinkPropPredDataset

    print('load', name)
    data = DglLinkPropPredDataset(name = name, root = root)
    print('finish loading', name)
    splitted_idx = data.get_edge_split()
    if name == 'ogbl-citation2':
        splitted_idx['train']['edge'] = th.cat(
            (splitted_idx['train']['source_node'].unsqueeze(1), 
            splitted_idx['train']['target_node'].unsqueeze(1)),
            axis = 1)
        splitted_idx['valid']['edge'] = th.cat(
            (splitted_idx['valid']['source_node'].unsqueeze(1), 
            splitted_idx['valid']['target_node'].unsqueeze(1)),
            axis = 1)
        splitted_idx['valid']['neg_edge'] = th.cat(
            (splitted_idx['valid']['source_node'].repeat(1000).unsqueeze(1), 
            splitted_idx['valid']['target_node_neg'].view(-1).unsqueeze(1)),
            axis = 1)
        splitted_idx['test']['edge'] = th.cat(
            (splitted_idx['test']['source_node'].unsqueeze(1), 
            splitted_idx['test']['target_node'].unsqueeze(1)),
            axis = 1)
        splitted_idx['test']['neg_edge'] = th.cat(
            (splitted_idx['test']['source_node'].repeat(1000).unsqueeze(1), 
            splitted_idx['test']['target_node_neg'].view(-1).unsqueeze(1)),
            axis = 1)
    graph = data[0]
    #from IPython import embed; embed()
    return graph, splitted_idx

def load_ogbg(name, device = th.device('cpu'), root = '/home/eva_share_users/zhuyu'):
    from ogb.graphproppred import DglGraphPropPredDataset

    print('load', name)
    data = DglGraphPropPredDataset(name = name, root = root)
    #from IPython import embed; embed()
    from tqdm import tqdm
    out_channels = 0
    for graph in tqdm(data):
        if name == 'ogbg-ppa':
            graph[0].ndata['feat'] = dgl.ops.copy_e_mean(graph[0], graph[0].edata['feat'])
        else:
            ef = graph[0].edata['feat']
            edge = graph[0].edges()[1]
            H = th.zeros(graph[0].num_nodes(), 3)
            for i in range(graph[0].num_nodes()):
                mask = th.eq(edge, i)
                H[i, :] += th.matmul(mask.float(), ef.float())
                H[i, :] /= graph[0].in_degrees(i)
            graph[0].ndata['feat'] = th.cat((graph[0].ndata['feat'], H), dim = 1)
        #from IPython import embed; embed()
        in_channels = graph[0].ndata['feat'].shape[1]
        try:
            out_channels = max(out_channels, int(graph[1]))
        except:
            from IPython import embed; embed()

    
    split_idx = data.get_idx_split()
    print('finish loading', name)
    from dgl.dataloading import GraphDataLoader
    train_loader = GraphDataLoader(data[split_idx['train']], batch_size = 256, shuffle = True,)
    valid_loader = GraphDataLoader(data[split_idx['valid']], batch_size = 256, shuffle = True,)
    test_loader = GraphDataLoader(data[split_idx['test']], batch_size = 256, shuffle = True,)
    #from IPython import embed; embed()
    return train_loader, valid_loader, test_loader, in_channels, out_channels + 1




def inductive_split(g):
    """Split the graph into training graph, validation graph, and test graph by training
    and validation masks.  Suitable for inductive models."""
    train_g = g.subgraph(g.ndata['train_mask'])
    val_g = g.subgraph(g.ndata['train_mask'] | g.ndata['val_mask'])
    test_g = g
    return train_g, val_g, test_g
