import torch.nn.functional as F
import torch as th
from utils.util import EarlyStopping
import time
import numpy as np
from sklearn.metrics import f1_score
from nets.sage import SAGE
from train.sample_train import load_subtensor
from train.sample_train import compute_acc
from train.sample_train import evaluate
import dgl

def accuracy(logits, labels):
    _, indices = th.max(logits, dim=1)
    correct = th.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def gat_evaluate(model, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels)

def gat_run(data,gpu=-1,filename='result/gat_run.txt',num_out_heads=1,num_layers=1,num_hidden=8,num_heads=8,epochs=200, lr=0.005,weight_decay=5e-4, in_drop=.6,attn_drop=.6,negative_slope=.2,residual=False,early_stop=False,fastmode=False):
    """
    INPUT:
    data : list : see the Unpack process
    gpu : int : the number of the device, -1 means cpu

    OUTPUT:

    """
    from nets.gats import GAT

    file = open(filename,'w')
    if gpu == -1:
        device = th.device('cpu')
    else:
        device = th.device('cuda:%d' % gpu)

    
    # Unpack the data
    n_classes,n_edges,g,num_feats = data
    #g = g.int().to(device)
    features = g.ndata['feat'].to(device)
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    
    # create model
    heads = ([num_heads] * num_layers) + [num_out_heads] # [num_layers个num_heads,最后一个是num_out_head]组成list
    model = GAT(g,
                num_layers,
                num_feats,
                num_hidden,
                n_classes,
                heads,
                F.elu,
                in_drop,
                attn_drop,
                negative_slope,
                residual)
    print(model,file=file)
    if early_stop:
        stopper = EarlyStopping(patience=100)

    model=model.to(device)
    loss_fcn = th.nn.CrossEntropyLoss()
    # use optimizer
    optimizer = th.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)

    # initialize graph
    dur = []
    for epoch in range(epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)
        
        train_acc = accuracy(logits[train_mask], labels[train_mask])
        
        if fastmode:
            val_acc = accuracy(logits[val_mask], labels[val_mask])
            _, indices = th.max(logits[val_mask], dim=1)
            f1 = f1_score(labels[val_mask].cpu().numpy(), indices.cpu(), average='micro')
        else:
            val_acc = gat_evaluate(model, features, labels, val_mask)
            _, indices = th.max(logits[val_mask], dim=1)
            f1 = f1_score(labels[val_mask].cpu().numpy(), indices.cpu(), average='micro')
            if early_stop:
                if stopper.step(val_acc, model):
                    break
        gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
              " ValAcc {:.4f} | f1-score {:.4f} |ETputs(KTEPS) {:.2f}| GPU : {:.1f} MB ".
              format(epoch, np.mean(dur), loss.item(), train_acc,
                     val_acc, f1, n_edges / np.mean(dur) / 1000,gpu_mem_alloc),file=file)

    print()
    if early_stop:
        model.load_state_dict(th.load('es_checkpoint.pt')) # Earlystop不同文件间有耦合
    acc = gat_evaluate(model, features, labels, test_mask)
    _, indices = th.max(logits[test_mask], dim=1)
    f1 = f1_score(labels[test_mask].cpu().numpy(), indices.cpu(), average='micro')
    print("Test Accuracy {:.4f} | F1-score {:.4f}".format(acc,f1),file=file)

"""
TODO
Add ChebConv 
Add comments in every function
Copy the conv.py file to the folder
"""
def cheb_coarsen_run(data,gpu=-1,pool_size=2,hidden_list=[16,32,64,128],k=2,lr=1e-3,epoch_times = 20, log_interval = 5):
    """
    for mnist len(hidden_list) = 4 if pool_size==2 len(hidden_list)=2 if pool_size=4
    since the coarsening_levels is 4
    the data device is the same with the model device
    """
    from nets.chebs import ChebNet2,ChebNet4
    train_loader,val_loader,test_loader,coarsening_levels,perm,n_classes,in_feats = data

    def cheb_evaluate(model,data_loader,device):
        model.eval()
        hit, tot = 0, 0
        for g, x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            out = model(g, x)
            hit += (out.max(-1)[1] == y).sum().item()
            tot += len(y)
        return hit / tot

    if gpu == -1:
        device = th.device('cpu')
    else:
        device = th.device('cuda:%d' % gpu)

    if pool_size == 2:
        if len(hidden_list)!=coarsening_levels:
            raise RuntimeError("Length of hidden_list can't match length of coarsening")
        model = ChebNet2(k, in_feats, hidden_list, n_classes) #打到高维上 [32,64,128,256]如果pooling size=2
    elif pool_size == 4:
        if len(hidden_list)!=coarsening_levels//2:
            raise RuntimeError("Length of hidden_list can't match length of coarsening")
        model = ChebNet4(k, in_feats, hidden_list, n_classes)
    else:
        raise RuntimeError("No support for pooling size %d now"%(pool_size))

    model = model.to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epoch_times):
        print('epoch {} starts'.format(epoch))
        model.train()
        hit, tot = 0, 0
        loss_accum = 0
        tic = time.time()
        stime = time.time()
        for i, (g, x, y) in enumerate(train_loader):
            x = x.to(device)
            #print('Line 158 x : ',len(x))
            y = y.to(device)
            if pool_size==2:
                g = [g_i.to(device) for g_i in g]
            elif pool_size==4:
                tmp = []
                for i,g_i in enumerate(g):
                    if i%2 ==0:
                        tmp.append(g_i.to(device))
                g = tmp
            #print('Line 171 g :',len(g))
            #print('Line 172 g[0] : ',g[0])
            #print('g[0] ndata :',g[0].ndata['xy'])
            #print('g[0] edata :',g[0].edata['u'])
            #raise KeyError('173')
            out = model(g, x)
            #print('Line 163 out : ',out.shape)
            hit += (out.max(-1)[1] == y).sum().item()
            tot += len(y)
            loss = F.nll_loss(out, y)
            loss_accum += loss.item()

            if (i + 1) % log_interval == 0:
                tic = time.time()
                acc = cheb_evaluate(model,val_loader,device)
                print('{:<4d}Batch: {:<4d}|loss: {:.4f}|acc: {:.4f}|time(s): {:.4f}'.format(log_interval,i,loss_accum / log_interval, hit / tot,(tic-stime)))
                print('val acc : {:.4f}'.format(acc))
                hit, tot = 0, 0
                loss_accum = 0
                stime = time.time()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    acc = cheb_evaluate(model,val_loader,device)
    print('test acc: ', acc)

def whole_sage_run(data_gpu,gpu,g,n_classes,filename = 'result/whole_sage_run.txt',num_epochs=20,eval_every=5,log_every=5,batch_size=32,lr=0.005,num_hidden=64,num_layers=2,dropout=0.0,shuffle=True,num_workers=4,drop_last=False):
    """
    Whole Sample(batch_size)
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
    in_feats = features.shape[1]

    #通过mask获取nid的方法

    train_nid = th.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = th.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = th.nonzero(~(g.ndata['train_mask'] | g.ndata['val_mask']), as_tuple=True)[0]
    # Create PyTorch DataLoader for constructing blocks

    # Define model and optimizer
    model = SAGE(in_feats,num_hidden, n_classes,num_layers, len(g.etypes), F.relu, dropout)

    if gpu==-1:
        device = th.device('cpu')
    else:
        device = th.device('cuda:%d' % gpu)
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers)
    dataloader = dgl.dataloading.NodeDataLoader(
        g,
        th.arange(g.num_nodes()),
        sampler,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers)

    model = model.to(device)
    loss_fcn = th.nn.CrossEntropyLoss()
    optimizer = th.optim.Adam(model.parameters(), lr=lr)

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
            tmp=0
            for block in gen_blocks:
                blocks.append(block.int().to(device))
                tmp += block.num_edges()          
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

