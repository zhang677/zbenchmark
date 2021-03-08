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
import time


"""
#Cheb2
data_gpu = -1 # data还没有放到gpu上
data = load_mnist_cheb(2)
cheb_coarsen_run(data,data_gpu,2)

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
data = inductive_sage_data(g,num_classes,data_gpu)
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

#transductive
dataset = 'citeseer'
data_gpu = 1
train_gpu = 1
print("load graph :")
tic = time.time()
g,num_classes = load_ordinary_graph(dataset)
print("load graph over :{:.6f} s".format(time.time()-tic))
transductive_sage_run(data_gpu,train_gpu,g,num_classes,'result/tran_sage_{:d}_{:s}.txt'.format(train_gpu,dataset))
print("train over :{:.6f} s".format(time.time()-tic))
print('over')
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

