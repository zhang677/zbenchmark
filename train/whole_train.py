import torch.nn.functional as F
import torch
from utils.utils import EarlyStopping
import time
import numpy as np
from sklearn.metrics import f1_score

def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
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
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:%d' % gpu)

    
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
    loss_fcn = torch.nn.CrossEntropyLoss()
    # use optimizer
    optimizer = torch.optim.Adam(
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
            _, indices = torch.max(logits[val_mask], dim=1)
            f1 = f1_score(labels[val_mask].cpu().numpy(), indices.cpu(), average='micro')
        else:
            val_acc = evaluate(model, features, labels, val_mask)
            _, indices = torch.max(logits[val_mask], dim=1)
            f1 = f1_score(labels[val_mask].cpu().numpy(), indices.cpu(), average='micro')
            if early_stop:
                if stopper.step(val_acc, model):
                    break
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
              " ValAcc {:.4f} | f1-score {:.4f} |ETputs(KTEPS) {:.2f}| GPU : {:.1f} MB ".
              format(epoch, np.mean(dur), loss.item(), train_acc,
                     val_acc, f1, n_edges / np.mean(dur) / 1000,gpu_mem_alloc),file=file)

    print()
    if early_stop:
        model.load_state_dict(torch.load('es_checkpoint.pt')) # Earlystop不同文件间有耦合
    acc = evaluate(model, features, labels, test_mask)
    _, indices = torch.max(logits[test_mask], dim=1)
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
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:%d' % gpu)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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

def whole_run():
    """
    用来测试简单的每次都做全图的算法
    """

