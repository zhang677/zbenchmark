import dgl
import numpy as np
import torch as th
import time


def inductive_sage_data(g, n_classes, data_gpu):
    """
    main.py will call this function to process the data: split the graph(alternative) 
    and get feats and labels. All these will pack in a list
    device -1 
    TODO 
    The device conversion should be put in which file?? 
    """

    train_g = g.subgraph(g.ndata['train_mask'])
    val_g = g.subgraph(g.ndata['train_mask'] | g.ndata['val_mask'])
    test_g = g
    train_nfeat = train_g.ndata.pop('feat')  # pop之后就删除了
    val_nfeat = val_g.ndata.pop('feat')
    test_nfeat = test_g.ndata.pop('feat')
    train_labels = train_g.ndata.pop('label')
    val_labels = val_g.ndata.pop('label')
    test_labels = test_g.ndata.pop('label')

    if data_gpu != -1:
        device = th.device('cuda:%d' % data_gpu)
        train_nfeat = train_nfeat.to(device)
        train_labels = train_labels.to(device)

    # Create csr/coo/csc formats before launching training processes with multi-gpu.
    # This avoids creating certain formats in each sub-process, which saves momory and CPU.
    train_g.create_formats_()
    val_g.create_formats_()
    test_g.create_formats_()
    # Pack data
    return n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
        val_nfeat, val_labels, test_nfeat, test_labels


def gat_data(g, n_classes):
    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()
    num_feats = g.ndata['feat'].shape[1]
    return n_classes, n_edges, g, num_feats

def inductive_gat_data(g, n_classes,data_gpu):
    """
    We denote the data_gpu here
    """
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    train_g = g.subgraph(g.ndata['train_mask'])
    val_g = g.subgraph(g.ndata['train_mask'] | g.ndata['val_mask'])
    test_g = g
    train_nfeat = train_g.ndata.pop('feat')  # pop之后就删除了
    val_nfeat = val_g.ndata.pop('feat')
    test_nfeat = test_g.ndata.pop('feat')
    train_labels = train_g.ndata.pop('label')
    val_labels = val_g.ndata.pop('label')
    test_labels = test_g.ndata.pop('label')

    if data_gpu != -1:
        device = th.device('cuda:%d' % data_gpu)
        train_nfeat = train_nfeat.to(device)
        train_labels = train_labels.to(device)

    # Create csr/coo/csc formats before launching training processes with multi-gpu.
    # This avoids creating certain formats in each sub-process, which saves momory and CPU.
    train_g.create_formats_()
    val_g.create_formats_()
    test_g.create_formats_()
    # Pack data
    return n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
        val_nfeat, val_labels, test_nfeat, test_labels