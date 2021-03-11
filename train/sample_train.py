import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn.pytorch as dglnn
import time
import tqdm
from nets.sage import SAGE
from nets.gats import GATBlock
from sklearn.metrics import f1_score

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (th.argmax(pred, dim=1) == labels).int().sum() / len(pred)

def evaluate(model, g, nfeat, labels, val_nid, device,batch_size,num_workers):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, nfeat, device,batch_size,num_workers) # 因为带采样带inference不太一样
    model.train()
    score = f1_score(labels[val_nid].cpu().numpy(),th.argmax(pred[val_nid].int(),dim=1).cpu().numpy(),average='micro')
    return score , compute_acc(pred[val_nid], labels[val_nid].to(pred.device))

def gat_evaluate(model, g, nfeat, labels, val_nid, device,batch_size,num_workers):
    """
    TODO: Together with the gat inference
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g,nfeat,val_nid,device,batch_size,num_workers)
    model.train()
    score = f1_score(labels[val_nid].cpu().numpy(),th.argmax(pred.int(),dim=1).cpu().numpy(),average='micro')
    return score, compute_acc(pred,labels[val_nid].to(pred.device))
    
def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    """
    Extracts features and labels for a subset of nodes
    """
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels

#### Entry point
def inductive_sage_run(gpu, data,filename = 'result/sample_sage_run.txt',num_epochs=20,eval_every=5,log_every=5,batch_size=32,lr=0.005,fan_out='10,25',num_hidden=64,num_layers=2,dropout=0.0,shuffle=True,num_workers=4,drop_last=False):
    """
    main.py will call this function to instantiate the model(SAGE) and train/val/test
    the aggregate type isn't exposed since it will be too cumbersome to tape the command
    INPUT:
    gpu : int : the number of device(-1 means 'cpu') 
    data : tuple : see the Unpack process
    eval_every : int : use when evaluating
    log_every : int : use when enumerating the dataloader
    fan_out : str : 'int,int,...'
    num_layers : int : must be consistent with fan_out
    filename : str : file to save the training process

    OUTPUT:
    run the train
    """

    file = open(filename,'w')
    tic =time.time()
    # Unpack data
    n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
    val_nfeat, val_labels, test_nfeat, test_labels = data
    in_feats = train_nfeat.shape[1]

    #通过mask获取nid的方法

    train_nid = th.nonzero(train_g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = th.nonzero(val_g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = th.nonzero(~(test_g.ndata['train_mask'] | test_g.ndata['val_mask']), as_tuple=True)[0]

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in fan_out.split(',')],replace=False) # [10, 25]
    dataloader = dgl.dataloading.NodeDataLoader(
        train_g,
        train_nid,
        sampler,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers)

    # Define model and optimizer
    model = SAGE(in_feats,num_hidden, n_classes,num_layers, len(train_g.etypes), F.relu, dropout)

    if gpu==-1:
        device = th.device('cpu')
    else:
        device = th.device('cuda:%d' % gpu)

    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("prepare model time : %.6f s"%(time.time()-tic))

    print(model,file=file)
    
    #raise KeyError('144')
    # Training loop
    total_tic= time.time()
    max_step = 0
    avg = 0
    total_train_time = 0
    iter_tput_forward = []
    iter_tput_backward = []
    iter_tput = []
    for epoch in range(num_epochs):
        epoch_tic = time.time()
        epoch_forward = 0
        epoch_backward = 0
        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        tic_step = time.time()
        tic_step_forward = time.time()
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # Load the input features as well as output labels
            batch_inputs, batch_labels = load_subtensor(train_nfeat, train_labels,
                                                        seeds, input_nodes, device)
            blocks = [block.int().to(device) for block in blocks]
            #print("blocks:%4d: "%(step),blocks[0],blocks[1])
            #print("input_nodes: ",input_nodes.shape)
            #print("batch_inputs: ",batch_inputs.shape)
            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            iter_tput_forward.append(len(seeds) / (time.time() - tic_step_forward))
            epoch_forward += time.time() - tic_step_forward

            tic_step_backward = time.time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_tput_backward.append(len(seeds) / (time.time() - tic_step_backward))
            epoch_backward += time.time() - tic_step_backward

            iter_tput.append(len(seeds) / (time.time() - tic_step))

            if step % log_every == 0 : 
                acc = compute_acc(batch_pred, batch_labels)
                f1 = f1_score(batch_labels.cpu().numpy(),th.argmax(batch_pred.int(),dim=1).cpu().numpy(),average='micro')
                gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Train F1 {:.4f} | Speed (samples/sec) {:.4f}| Forward Speed (samples/sec) {:.4f}|Backward Speed (samples/sec) {:.4f}|GPU {:.1f} MB'.format(
                    epoch, step, loss.item(), acc.item(),f1, np.mean(iter_tput[3:]),np.mean(iter_tput_forward[3:]),np.mean(iter_tput_backward[3:]), gpu_mem_alloc),file=file)
            tic_step = time.time()
            tic_step_forward = time.time()
            max_step=step

        epoch_toc = time.time()
        print('Epoch Time(s): {:.6f} | Total steps: {:4d} | Total forward time: {:.6f}s | Total backward time: {:.6f}s '.format(epoch_toc - epoch_tic,max_step,epoch_forward,epoch_backward),file=file)
        total_train_time += epoch_toc - epoch_tic
        if epoch >= 5:
            avg += epoch_toc - epoch_tic
        if epoch % eval_every == 0 and epoch != 0:
            eval_time = time.time()
            f1,eval_acc = evaluate(model, val_g, val_nfeat, val_labels, val_nid, device,batch_size,num_workers)
            print('Eval Acc {:.4f} | F1 {:.4f} | Eval time: {:.6f} s'.format(eval_acc,f1,time.time()-eval_time),file=file)
    
    test_time = time.time()
    f1, test_acc = evaluate(model, test_g, test_nfeat, test_labels, test_nid, device,batch_size,num_workers)
    print('Test Acc: {:.4f}| F1: {:.4f}| Test time: {:.6f} s'.format(test_acc,f1,time.time()-test_time),file=file)
    print('Avg epoch time(s): {:.4f} | F1 {:.4f} | Total epochs: {:4d} | Total Time: {:.6f} s | Total Train Time:{:.6f} s'.format(avg / (epoch - 4),f1,num_epochs,time.time()-total_tic,total_train_time),file=file)

def transductive_sage_run(data_gpu,gpu,g,n_classes,filename = 'result/sample_sage_run.txt',num_epochs=20,eval_every=5,log_every=5,batch_size=32,lr=0.005,fan_out='10,25',num_hidden=64,num_layers=2,dropout=0.0,shuffle=True,num_workers=4,drop_last=False):
    """
    main.py will call this function to instantiate the model(SAGE) and train/val/test
    the aggregate type isn't exposed since it will be too cumbersome to tape the command
    INPUT:
    gpu : int : the number of device(-1 means 'cpu') 
    g : DGLGraph : the whole graph
    eval_every : int : use when evaluating
    log_every : int : use when enumerating the dataloader
    fan_out : str : 'int,int,...'
    num_layers : int : must be consistent with fan_out
    filename : str : file to save the training process

    OUTPUT:
    run the train
    """

    file = open(filename,'w')
    tic =time.time()
    # Unpack data
    if data_gpu != -1:
        data_device = th.device('cuda:%d' % data_gpu)
    else:
        data_device = th.device('cpu')
    features = g.ndata.pop('feat').to(data_device)
    labels = g.ndata.pop('label').to(data_device)
    #g.int().to(data_device)
    in_feats = features.shape[1]

    #通过mask获取nid的方法

    train_nid = th.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = th.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = th.nonzero(~(g.ndata['train_mask'] | g.ndata['val_mask']), as_tuple=True)[0]
    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in fan_out.split(',')],replace=False) # [10, 25]
    dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_nid,
        sampler,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers)

    # Define model and optimizer
    model = SAGE(in_feats,num_hidden, n_classes,num_layers, len(g.etypes), F.relu, dropout)

    if gpu==-1:
        device = th.device('cpu')
    else:
        device = th.device('cuda:%d' % gpu)

    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("prepare model time : %.6f s"%(time.time()-tic))

    print(model,file=file)
    
    #raise KeyError('144')
    # Training loop
    total_tic= time.time()
    max_step = 0
    avg = 0
    total_train_time = 0
    iter_tput_forward = []
    iter_tput_backward = []
    iter_tput = []
    for epoch in range(num_epochs):
        epoch_tic = time.time()
        epoch_forward = 0
        epoch_backward = 0
        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        tic_step = time.time()
        tic_step_forward = time.time()
        for step, (input_nodes, seeds, gen_blocks) in enumerate(dataloader):
            # Load the input features as well as output labels

            batch_inputs, batch_labels = load_subtensor(features, labels,
                                                        seeds, input_nodes, device)
            blocks = []
            tmp = 0
            for block in gen_blocks:
                blocks.append(block.int().to(device))
                tmp += block.num_edges()
                    
            #blocks = [block.int().to(device) for block in blocks]
            #print("blocks:%4d: "%(step),blocks[0],blocks[1])
            #print("input_nodes: ",input_nodes.shape)
            #print("batch_inputs: ",batch_inputs.shape)
            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            iter_tput_forward.append(tmp / (time.time() - tic_step_forward))
            epoch_forward += time.time() - tic_step_forward

            tic_step_backward = time.time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_tput_backward.append(tmp / (time.time() - tic_step_backward))
            epoch_backward += time.time() - tic_step_backward

            iter_tput.append(len(seeds) / (time.time() - tic_step))

            if step % log_every == 0 : 
                acc = compute_acc(batch_pred, batch_labels)
                f1 = f1_score(batch_labels.cpu().numpy(),th.argmax(batch_pred.int(),dim=1).cpu().numpy(),average='micro')
                gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Train F1 {:.4f} | Speed (seeds/sec) {:.4f}| Forward Speed (edges/sec) {:.4f}|Backward Speed (edges/sec) {:.4f}|GPU {:.1f} MB'.format(
                    epoch, step, loss.item(), acc.item(),f1, np.mean(iter_tput[3:]),np.mean(iter_tput_forward[3:]),np.mean(iter_tput_backward[3:]), gpu_mem_alloc),file=file)
            tic_step = time.time()
            tic_step_forward = time.time()
            max_step=step

        epoch_toc = time.time()
        print('Epoch Time(s): {:.6f} | Total steps: {:4d} | Total forward time: {:.6f}s | Total backward time: {:.6f}s '.format(epoch_toc - epoch_tic,max_step,epoch_forward,epoch_backward),file=file)
        total_train_time += epoch_toc - epoch_tic
        if epoch >= 5:
            avg += epoch_toc - epoch_tic
        if epoch % eval_every == 0 and epoch != 0:
            eval_time = time.time()
            f1,eval_acc = evaluate(model, g, features, labels, val_nid, device,batch_size,num_workers)
            print('Eval Acc {:.4f} | F1 {:.4f} | Eval time: {:.6f} s'.format(eval_acc,f1,time.time()-eval_time),file=file)
    
    test_time = time.time()
    f1, test_acc = evaluate(model, g, features, labels, test_nid, device,batch_size,num_workers)
    print('Test Acc: {:.4f}| F1: {:.4f}| Test time: {:.6f} s'.format(test_acc,f1,time.time()-test_time),file=file)
    print('Avg epoch time(s): {:.4f} | F1 {:.4f} | Total epochs: {:4d} | Total Time: {:.6f} s | Total Train Time:{:.6f} s'.format(avg / (epoch - 4),f1,num_epochs,time.time()-total_tic,total_train_time),file=file)

def batch_gat_run(data_gpu, gpu, data,filename = 'result/sample_gat_run.txt',num_out_heads=1,num_layers=1,num_hidden=8,num_heads=8,num_epochs=200,eval_every=5,log_every=5,batch_size=32,fan_out='5,10',lr=0.005,weight_decay=5e-4, in_drop=.6,attn_drop=.6,negative_slope=.2,residual=False,early_stop=False,fastmode=False,shuffle=True,drop_last=False,num_workers=4):
    """
    main.py will call this function to instantiate the model(SAGE) and train/val/test
    the aggregate type isn't exposed since it will be too cumbersome to tape the command
    INPUT:
    gpu : int : the number of device(-1 means 'cpu') 
    data : tuple : see the Unpack process
    eval_every : int : use when evaluating
    log_every : int : use when enumerating the dataloader
    fan_out : str : 'int,int,...'
    num_layers : int : must be consistent with fan_out
    filename : str : file to save the training process

    OUTPUT:
    run the train
    """

    file = open(filename,'w')
    tic =time.time()
    # Set the device
    if data_gpu == -1:
        data_device = th.device('cpu')
    else:
        data_device = th.device('cuda:%d' % data_gpu)

    # Unpack data
    n_classes,n_edges,g,num_feats = data
    features = g.ndata.pop('feat').to(data_device)
    labels = g.ndata.pop('label').to(data_device)
    #g = g.int().to(data_device)
    

    # Set node ids for further sampler
    train_nid = th.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = th.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = th.nonzero(~(g.ndata['train_mask'] | g.ndata['val_mask']), as_tuple=True)[0]

    # Set the heads
    heads = ([num_heads] * num_layers) + [num_out_heads]

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in fan_out.split(',')],replace=False) # [10, 25]
    dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_nid,
        sampler,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers)
    # Define model and optimizer
    model = GATBlock(num_layers, # '5,10'->1 layer
                num_feats,
                num_hidden,
                n_classes,
                heads,
                F.elu,
                in_drop,
                attn_drop,
                negative_slope,
                residual)
    if gpu == -1:
        device = th.device('cpu')
    else:
        device = th.device('cuda:%d' % gpu)
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("prepare model time : %.6f s"%(time.time()-tic))

    print(model,file=file)
    
    #raise KeyError('144')
    # Training loop
    total_tic= time.time()
    max_step = 0
    avg = 0
    total_train_time = 0
    iter_tput_forward = []
    iter_tput_backward = []
    iter_tput = []
    for epoch in range(num_epochs):
        epoch_tic = time.time()
        epoch_forward = 0
        epoch_backward = 0
        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        tic_step = time.time()
        tic_step_forward = time.time()
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # Load the input features as well as output labels
            
            batch_inputs, batch_labels = load_subtensor(features, labels,
                                                        seeds, input_nodes, device)
            blocks = [block.int().to(device) for block in blocks]
            #print("blocks:%4d: "%(step),blocks[0],blocks[1])
            #print("input_nodes: ",input_nodes.shape)
            #print("batch_inputs: ",batch_inputs.shape)
            # Compute loss and prediction
            #print("Batch Inputs:",batch_inputs.shape)
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            iter_tput_forward.append(len(seeds) / (time.time() - tic_step_forward))
            epoch_forward += time.time() - tic_step_forward

            tic_step_backward = time.time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_tput_backward.append(len(seeds) / (time.time() - tic_step_backward))
            epoch_backward += time.time() - tic_step_backward

            iter_tput.append(len(seeds) / (time.time() - tic_step))

            if step % log_every == 0 : 
                acc = compute_acc(batch_pred, batch_labels)
                f1 = f1_score(batch_labels.cpu().numpy(),th.argmax(batch_pred.int(),dim=1).cpu().numpy(),average='micro')
                gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Train F1 {:.4f} | Speed (samples/sec) {:.4f}| Forward Speed (samples/sec) {:.4f}|Backward Speed (samples/sec) {:.4f}|GPU {:.1f} MB'.format(
                    epoch, step, loss.item(), acc.item(),f1, np.mean(iter_tput[3:]),np.mean(iter_tput_forward[3:]),np.mean(iter_tput_backward[3:]), gpu_mem_alloc),file=file)
            tic_step = time.time()
            tic_step_forward = time.time()
            max_step=step

        epoch_toc = time.time()
        print('Epoch Time(s): {:.6f} | Total steps: {:4d} | Total forward time: {:.6f}s | Total backward time: {:.6f}s '.format(epoch_toc - epoch_tic,max_step,epoch_forward,epoch_backward),file=file)
        total_train_time += epoch_toc - epoch_tic
        if epoch >= 5:
            avg += epoch_toc - epoch_tic
        if epoch % eval_every == 0 and epoch != 0:
            eval_time = time.time()
            f1,eval_acc = gat_evaluate(model, g, features, labels, val_nid, device,batch_size,num_workers)
            print('Eval Acc {:.4f} | F1 {:.4f} | Eval time: {:.6f} s'.format(eval_acc,f1,time.time()-eval_time),file=file)
    
    test_time = time.time()
    f1, test_acc = gat_evaluate(model, g, features, labels, test_nid, device,batch_size,num_workers)
    print('Test Acc: {:.4f}| F1: {:.4f}| Test time: {:.6f} s'.format(test_acc,f1,time.time()-test_time),file=file)
    print('Avg epoch time(s): {:.4f} | F1 {:.4f} | Total epochs: {:4d} | Total Time: {:.6f} s | Total Train Time:{:.6f} s'.format(avg / (epoch - 4),f1,num_epochs,time.time()-total_tic,total_train_time),file=file)