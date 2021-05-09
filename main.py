"""
TODO
if sample:
    from sample_train.py import modelname_run
else:
    from whole_train.py import modlename_run

g = load_graph(dataset) # 
data = process(g) # ChebConv MNIST 没有process的步骤
modlename_run(data,other parameters)
"""
"""
from load_graph import load_ordinary_graph,load_mnist_cheb
from process import inductive_sage_data ,gat_data
from sample_train import inductive_sage_run,transductive_sage_run
from whole_train import cheb_coarsen_run
from whole_train import gat_run
"""
from train import *
from graph import *
from utils import *
from train.whole_train import whole_sage_run
import time
import argparse

"""
#Cheb2
data_gpu = 0 # data还没有放到gpu上
data = load_mnist_cheb(2)
cheb_coarsen_run(data,data_gpu,2)
"""
"""
dataset = 'citeseer'
data_gpu = 0
train_gpu = 0 # data还没有放到gpu上
g,num_classes = load_ordinary_graph(dataset)
data = utils.gat_data(g,num_classes)
batch_gat_run(data_gpu,train_gpu,data,'result/batch_gat_{:d}_{:s}.txt'.format(train_gpu,dataset))
print('over')
"""
"""
#Cheb4
data_gpu = -1 # data还没有放到gpu上
data = load_mnist_cheb(4)
cheb_coarsen_run(data,data_gpu,4)
"""


#GAT
"""
dataset = 'citeseer'
train_gpu = 0 # data还没有放到gpu上
g,num_classes = load_ordinary_graph(dataset)
data = gat_data(g,num_classes,train_gpu)
gat_run(data,train_gpu,'result/gat_{:d}_{:s}.txt'.format(train_gpu,dataset))
print('over')
dataset = 'cora'
train_gpu = 1 # data还没有放到gpu上
g,num_classes = load_ordinary_graph(dataset)
data = gat_data(g,num_classes,train_gpu)
gat_run(data,train_gpu,'result/gat_{:d}_{:s}.txt'.format(train_gpu,dataset))
print('over')
"""
"""
dataset='pubmed'
train_gpu = 0 # data还没有放到gpu上
g,num_classes = load_ordinary_graph(dataset)
data = gat_data(g,num_classes,train_gpu)
gat_run(data,train_gpu,'result/gat_{:d}_{:s}.txt'.format(train_gpu,dataset),8)
print('over')
"""


#SAGE
"""
dataset = 'cora'
data_gpu = 0 # TODO 还有一些问题，搞不清是把谁（data 还是 model）存在哪
train_gpu = 0
g,num_classes = load_ordinary_graph(dataset)
data = utils.inductive_sage_data(g,num_classes,data_gpu)
inductive_sage_run(train_gpu,data,'result/sage_{:d}_{:s}.txt'.format(train_gpu,dataset))
print('over')
"""
"""
# inductive
dataset = 'citeseer'
data_gpu = 1
train_gpu = 1
print("load graph :")
tic = time.time()
g,num_classes = load_ordinary_graph(dataset)
print("load graph over :{:.6f} s".format(time.time()-tic))
print("inductive split :")
tic = time.time()
data = inductive_sage_data(g,num_classes,data_gpu)
print("inductive split over :{:.6f} s".format(time.time()-tic))
inductive_sage_run(train_gpu,data,'result/sage_{:d}_{:s}.txt'.format(train_gpu,dataset))
print("train over :{:.6f} s".format(time.time()-tic))
print('over')
"""
"""
#transductive
dataset = 'citeseer'
data_gpu = 1
train_gpu = 1
print("load graph :")
tic = time.time()
g,num_classes = load_ordinary_graph(dataset)
print("load graph over :{:.6f} s".format(time.time()-tic))
transductive_sage_run(data_gpu,train_gpu,g,num_classes,'result/tran_sage_{:d}_{:s}.txt'.format(train_gpu,dataset),)
print("train over :{:.6f} s".format(time.time()-tic))
print('over')
"""
"""
dataset='reddit'
data_gpu = -1
train_gpu = 0
print("load graph :")
tic = time.time()
g,num_classes = load_ordinary_graph(dataset)
print("load graph over :{:.6f} s".format(time.time()-tic)
print("inductive split :")
tic = time.time()
data = inductive_sage_data(g,num_classes,data_gpu)
print("inductive split over :{:.6f} s".format(time.time()-tic))
tic = time.time()
inductive_sage_run(train_gpu,data,'result/sage_{:d}_{:s}.txt'.format(train_gpu,dataset),num_epochs=6,eval_every=2,log_every=200,batch_size=32)
print("train over :{:.6f} s".format(time.time()-tic))
print('over')
"""

if __name__ == '__main__':
    # set configurations of the model and training process
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',type=str,default='gat',help='sage/gat/cheb')
    parser.add_argument('--data-mode',type=str,default='inductive',help='inductive/transductive')
    parser.add_argument('--train-mode',type=str,default='whole',help='sample/whole sage:sample/whole gat:sample/whole cheb:whole')
    #parser.add_argument('--train', type=str, default='gat_run', help='inductive_sage_run/transductive_sage_run/gat_run/cheb_coarsen_run/batch_gat_run/gat_run')
    parser.add_argument('--datagpu',type=int,default=-1,help='data_cpu')
    parser.add_argument('--gpu',type=int,default=-1,help='gpu of training')
    parser.add_argument('--dataset',type=str,default='citeseer',help='training dataset')
    parser.add_argument('--root',type=str,default='result',help='where to save the result')
    parser.add_argument('--num-out-heads',type=int,default=1,help='GAT num of out heads')
    parser.add_argument('--num-layers',type=int,default=2,help='GAT / GraphSAGE')
    parser.add_argument('--num-hidden',type=int,default=32,help='The hidden_size is the same for every hidden-layer')
    parser.add_argument('--hidden-list',type=str,default='16,32,64,128',help='The chebconv uses')
    parser.add_argument('--num-heads',type=int,default=8,help='Num of hidden heads')
    parser.add_argument('--eval-every',type=int,default=5,help='Should be consistent with the dataset-size')
    parser.add_argument('--log-every',type=int,default=5,help='Frequency of printing loss')
    parser.add_argument('--batch-size',type=int,default=32,help='Should be consistent with the dataset-size')
    parser.add_argument('--fan-out',type=str,default='5,10',help='Sampling need')
    parser.add_argument('--lr',type=float,default=0.005,help='Learning rate')
    parser.add_argument('--weight-decay',type=float,default=5e-4,help='GAT only')
    parser.add_argument('--in-drop',type=float,default=.6,help='GAT only')
    parser.add_argument('--attn-drop',type=float,default=.6,help='GAT only')
    parser.add_argument('--negative-slope',type=float,default=.2,help='GAT only')
    parser.add_argument('--residual',type=bool,default=False,help='GAT only')
    parser.add_argument('--fastmode',type=bool,default=False,help='GAT only stop training when val_acc drops')
    parser.add_argument('--early-stop',type=bool,default=False,help='GAT if eval begins to stop early')
    parser.add_argument('--shuffle',type=bool,default=True,help='GAT/GraphSAGE')
    parser.add_argument('--drop-last',type=bool,default=True,help='Drop last GAT/GraphSAGE')
    parser.add_argument('--num-workers',type=int,default=4,help='Num workers')
    parser.add_argument('--num-epochs',type=int,default=200,help='Epochs')
    parser.add_argument('--pool-size',type=int,default=2,help='Only used by cheb')
    parser.add_argument('--grid-side',type=int,default=28,help='Only for cheb, [28,28]')
    parser.add_argument('--number-edges',type=int,default=8,help='k-nn')
    parser.add_argument('--metric',type=str,default='euclidean',help='Only for cheb')
    parser.add_argument('--dropout',type=float,default=0.0,help='dropout in GraphSAGE')
    opt = parser.parse_args()
    filename = '{:s}/{:s}_{:s}_{:s}_{:s}_{:d}_{:d}.txt'.format(opt.root,opt.model,opt.data_mode,opt.train_mode,opt.dataset,opt.datagpu,opt.gpu)
    if opt.model == 'cheb':
        assert opt.pool_size==2 or opt.pool_size==4 , 'cheb pool size can only be 2 or 4 '
        data = load_mnist_cheb(pool_size=opt.pool_size,grid_side=opt.grid_side,number_edges=opt.number_edges,metric=opt.metric,coarsening_levels=4,batch_size=opt.batch_size)
        cheb_coarsen_run(data,opt.gpu,pool_size=opt.pool_size,hidden_list=opt.hidden_list,k=2,lr=opt.lr,epoch_times=opt.num_epochs,log_interval=opt.log_every)

    elif opt.model == 'sage':
        layer_size=[int(fanout) for fanout in opt.fan_out.split(',')]
        assert opt.num_layers==len(layer_size),'num_layers should be consistent with fan_out'
        if opt.data_mode == 'inductive':
            if opt.train_mode == 'sample':
                print("load graph :")
                tic = time.time()
                g,num_classes = load_ordinary_graph(opt.dataset)
                print("load graph over :{:.6f} s".format(time.time()-tic))
                print("inductive split :")
                tic = time.time()
                data = inductive_sage_data(g,num_classes,opt.datagpu)
                print("inductive split over :{:.6f} s".format(time.time()-tic))
                inductive_sage_run(opt.gpu,data,filename=filename,num_epochs=opt.num_epochs,
                eval_every=opt.eval_every,log_every=opt.log_every,batch_size=opt.batch_size,
                lr=opt.lr,fan_out=opt.fan_out,num_hidden=opt.num_hidden,num_layers=opt.num_layers,
                dropout=opt.dropout,shuffle=opt.shuffle,num_workers=opt.num_workers,drop_last=opt.drop_last)    
                print("train over :{:.6f} s".format(time.time()-tic))
                print('over')
            elif opt.train_mode == 'whole':
                raise KeyError('No such train_mode : {:s}'.format(opt.train_mode))
            else:
                raise KeyError('No such train_mode : {:s}'.format(opt.train_mode))

        elif opt.data_mode == 'transductive':
            if opt.train_mode == 'sample':
                print("load graph :")
                tic = time.time()
                g,num_classes = load_ordinary_graph(opt.dataset)
                print("load graph over :{:.6f} s".format(time.time()-tic))
                transductive_sage_run(data_gpu=opt.datagpu,gpu=opt.gpu,g=g,n_classes=num_classes,filename=filename,num_epochs=opt.num_epochs,
                eval_every=opt.eval_every,log_every=opt.log_every,batch_size=opt.batch_size,
                lr=opt.lr,fan_out=opt.fan_out,num_hidden=opt.num_hidden,num_layers=opt.num_layers,
                dropout=opt.dropout,shuffle=opt.shuffle,num_workers=opt.num_workers,drop_last=opt.drop_last)    
                print("train over :{:.6f} s".format(time.time()-tic))
                print('over')
            elif opt.train_mode == 'whole':
                print("load graph :")
                tic = time.time()
                g,num_classes = load_ordinary_graph(opt.dataset)
                print("load graph over :{:.6f} s".format(time.time()-tic))
                whole_sage_run(data_gpu=opt.datagpu,gpu=opt.gpu,g=g,n_classes=num_classes,filename=filename,num_epochs=opt.num_epochs,
                eval_every=opt.eval_every,log_every=opt.log_every,batch_size=opt.batch_size,
                lr=opt.lr,num_hidden=opt.num_hidden,num_layers=opt.num_layers,dropout=opt.dropout,
                shuffle=opt.shuffle,num_workers=opt.num_workers,drop_last=opt.drop_last)    
                print("train over :{:.6f} s".format(time.time()-tic))
                print('over')
            else:
                raise KeyError('No such train_mode : {:s}'.format(opt.train_mode))              

        else:
            raise KeyError('No such data_mode : {:s}'.format(opt.data_mode))

    
    elif opt.model == 'gat':
        if opt.data_mode == 'transductive':
            if opt.train_mode == 'whole':
                print("load graph :")
                tic = time.time()
                g,num_classes = load_ordinary_graph(opt.dataset)
                print("load graph over :{:.6f} s".format(time.time()-tic))
                tic = time.time()
                data = gat_data(g,num_classes)
                print("Add self loop over :{:.6f} s".format(time.time()-tic))
                gat_run(data=data,gpu=opt.gpu,filename=filename,num_out_heads=opt.num_out_heads,num_layers=opt.num_layers,
                num_hidden=opt.num_hidden,num_heads=opt.num_heads,epochs=opt.num_epochs,lr=opt.lr,weight_decay=opt.weight_decay,
                in_drop=opt.in_drop,attn_drop=opt.attn_drop,negative_slope=opt.negative_slope,residual=opt.residual,early_stop=opt.early_stop,
                fastmode=opt.fastmode)
                print("train over :{:.6f} s".format(time.time()-tic))
                print('over')

            elif opt.train_mode == 'sample':
                print("load graph :")
                tic = time.time()
                g,num_classes = load_ordinary_graph(opt.dataset)
                print("load graph over :{:.6f} s".format(time.time()-tic))
                tic = time.time()
                data = gat_data(g,num_classes)
                print("Add self loop over :{:.6f} s".format(time.time()-tic))
                batch_gat_run(data_gpu=opt.datagpu,gpu=opt.gpu,data=data,filename=filename,num_out_heads=opt.num_out_heads,num_layers=opt.num_layers,
                num_hidden=opt.num_hidden,num_heads=opt.num_heads,num_epochs=opt.num_epochs,eval_every=opt.eval_every,log_every=opt.log_every,
                batch_size=opt.batch_size,fan_out=opt.fan_out,lr=opt.lr,weight_decay=opt.weight_decay,in_drop=opt.in_drop,
                attn_drop=opt.attn_drop,negative_slope=opt.negative_slope,residual=opt.residual,early_stop=opt.early_stop,
                fastmode=opt.fastmode,shuffle=opt.shuffle,drop_last=opt.drop_last,num_workers=opt.num_workers)
                print("train over :{:.6f} s".format(time.time()-tic))
                print('over')

            else:
                raise KeyError('No such train_mode : {:s}'.format(opt.train_mode))

        elif opt.data_mode == 'inductive':
            if opt.train_mode == 'sample':
                print("load graph :")
                tic = time.time()
                g,num_classes = load_ordinary_graph(opt.dataset)
                print("load graph over :{:.6f} s".format(time.time()-tic))
                print("inductive split :")
                tic = time.time()
                data = inductive_gat_data(g,num_classes,opt.datagpu)
                print("inductive split over :{:.6f} s".format(time.time()-tic))
                tic = time.time()               
                one_graph_inductive_gat_run(gpu=opt.gpu,data=data,filename=filename,num_out_heads=opt.num_out_heads,num_layers=opt.num_layers,
                num_hidden=opt.num_hidden,num_heads=opt.num_heads,num_epochs=opt.num_epochs,eval_every=opt.eval_every,log_every=opt.log_every,
                batch_size=opt.batch_size,fan_out=opt.fan_out,lr=opt.lr,weight_decay=opt.weight_decay,in_drop=opt.in_drop,
                attn_drop=opt.attn_drop,negative_slope=opt.negative_slope,residual=opt.residual,early_stop=opt.early_stop,
                fastmode=opt.fastmode,shuffle=opt.shuffle,drop_last=opt.drop_last,num_workers=opt.num_workers)
                print("train over :{:.6f} s".format(time.time()-tic))
                print('over')
            else:
                raise KeyError('No such train_mode : {:s}'.format(opt.train_mode))

        else:
            raise KeyError('No such data_mode : {:s}'.format(opt.data_mode))

    else:
        raise KeyError('No such model : {:s}'.format(opt.model))            
    
    """
            TODO: 
            GAT inductive two ways
            Aggregate type in GraphSAGE
    """


    


    
