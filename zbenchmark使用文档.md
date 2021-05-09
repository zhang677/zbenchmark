# zbenchmark使用文档

## 文件结构

｜——main.py

｜——utils

｜		｜——init.py

​			｜——util.py

​			｜——process.py

｜——train

｜		｜——init.py

｜		｜——sample_train.py

｜		｜——whole_train.py

｜——result

｜——nets

｜		｜——init.py

｜		｜——chebs.py

 |		 ｜——gates.py

｜		｜——gats.py

｜		｜——sage.py

｜——MNIST

｜——graph

｜		｜——coarsening.py

｜		｜——coordinate.py

｜		｜——grid_graph.py

｜		｜——load_graph.py

｜——conv

｜		｜——init.py

｜		｜——addchebconv.py

｜		｜——addsageconv.py	

## 解释

### 整体思路

/conv：把conv类的实现放在``/conv``中，如果与DGL库不同就在文件名前加上add

/graph：``coarsening.py``,``coordinate.py``,``grid_graph.py``,专门为``chebconv``使用，用来构建graph。``load_graph.py``中返回的是DGLGraph(``dict``类型)，日后如果也有需要构建Graph的算法可以放在里面。

/MNIST：是``ChebConv``的数据集

/nets：是把conv连接起来构成net，不同的sample方式，以及是否sample会影响net的构建方式。

/utils：里面是准备数据集，分割数据集(inductive)，处理数据集(GAT中加自环)

/train：是训练过程

下面介绍一些重要的文件

### main.py

#### 程序主入口

|        | inductive | transductive |
| :----: | :-------: | :----------: |
| whole  |           |   sage/gat   |
| sample | sage/gat  |   sage/gat   |

对于gat的whole+inductive,[这里](https://github.com/dmlc/dgl/tree/master/examples/pytorch/gat)有，我们当然可以自己写一个，但是这边暂时还没有放进去[**TODO**]，因为涉及到inductive的两种方式，上面采用的是下图这种。而在zbenchmark中inductive是另一种。

<img src="https://cdn.nlark.com/yuque/0/2021/png/2923750/1615119317284-c5a0fc32-b9a5-4942-8e91-97dad36107f2.png" alt="图片.png" style="zoom:25%;" />

目前只有node classification一种任务，或者说只关注在表示学习的阶段。对于模型的细微调整建议在sample_train.py或whole_train.py中直接添加新函数。

这里特别提一下ChebConv

参考了[这里](https://github.com/dmlc/dgl/tree/master/examples/pytorch/model_zoo)的代码，为了方便写代码，有许多重复计算，所以效率不高，而且针对的数据集较小不使用于本benchmark。提供了cheb2和cheb4两种模式。cheb2是用二叉树构造计算图，cheb4是用四叉树构造计算图。并需要使用相应的hidden_list。两种模式用pool_size指定。grid_side指定把图片grid成几个点，这里是28X28。在metric中指定根据什么构建图（把几何关系映射到拓扑关系）

#### 参数解释

##### model **模型**：

目前只完善地支持gat和sage，cheb由于实现的方式和 dgl框架融合不太好，所以训练中有大量冗余数据，但是由于仍是全图拉普拉斯矩阵的思路，所以暂不做优化。如果想优化可以在``chebs.py``,``coarsening.py``,``coordinate.py``,``grid_graph.py``,``addchebconv.py``以及``whole_train.py``的``cheb_coarsen_run`` 中修改

##### data_mode **如何处理图数据**：

inductive:调用``load_ordinary_graph``把大图分割成如下形式，集合的包含关系表示DGLGraph中点和边集合的包含关系,可以称之为``one_graph_inductive``。

<img src="https://cdn.nlark.com/yuque/0/2021/png/2923750/1615119565149-68de2753-1a4e-4324-a937-fad0d5092bd2.png" alt="图片.png" style="zoom:33%;" />

transductive:直接从数据集拿到DGLGraph，并在把全图的点作为train的输入

##### train_mode 如何聚合图信息：

whole: 在GraphSAGE中train时直接用``FullNeighborSampler``。GAT就直接对于全图的每一个点计算注意力系数

sample:在GraphSAGE中直接用``NeighborSampler``（典型值[10,15]）在GAT中sample出batch。但是注意二者在做inference的时候不同

##### datagpu 大概是指feature存放的位置

但是不同的函数在使用的时候不太一样。DGLGraph一般存在GPU里

| 函数                                                  | datagpu存放的数据                   |
| ----------------------------------------------------- | ----------------------------------- |
| ``/utils/process.py``:`` inductive_sage_data``        | train_nfeats \| train_labels        |
| ``/train/whole_train.py``:`` whole_sage_run``         | g,ndata['feat'] \| g.ndata['label'] |
| ``/train/sample_train.py``:`` transductive_sage_run`` | g,ndata['feat'] \| g,ndata['label'] |
| ``/train/sample_train.py``:`` batch_gat_run``         | g,ndata['feat'] \| g,ndata['label'] |
| ``/utils/process.py``:`` inductive_gat_data``         | train_nfeats \| train_labels        |

虽然表面上看起来问题不大，但是实际上inference的时候，以及计算f1-score，以及涉及到block的时候gpu很容易放不下，也很容易出现cpu和gpu，以及cuda分配的不同gpu通信拖慢速度。

##### gpu 模型的位置

主要计算发生的位置（？只是猜想未经验证[**TODO**]）

##### dataset 数据集

目前只支持``reddit,citeseer,cora,pubmed``四个，``load_ogb``中有一些，但是还没有给``main.py``暴露接口[**TODO**]

##### num_layers 大概是图卷积层的层数



| 参数          | 含义                                                         |
| ------------- | ------------------------------------------------------------ |
| root          | 存放结果的文件夹                                             |
| num_out_heads | GAT输出head数                                                |
| num_hdden     | 隐藏层feature的维度，注意如果在这里设置，那么每一个隐藏层feature维度只能相同 |
| hidden_list   | chebconv使用，用来指示每一层的feature维度                    |
| num_heads     | GAT hidden_heads的个数                                       |
| fan_out       | 用在k-hop Sample中，指示每一层选该点的几个邻居               |
| fast_mode     | 训练的一种技术，在val_acc向下掉时停止训练                    |
| early_stop    | 也是一种训练的技术，不限于模型/数据集                        |
| 其他          | 都是训练的有通用含义的参数，这里不再列举                     |

### /conv/addchebconv.py

 因为与[这里](https://github.com/dmlc/dgl/tree/master/examples/pytorch/model_zoo)不同，没有实现Monet，所以做了修改。注意在`/graph/coordinate.py`中极坐标信息存在了边的'u'中。距离信息存在了边的'v'中。由于需要配合DGL的[message_passing机制](https://docs.dgl.ai/tutorials/blitz/3_message_passing.html)实现的比较麻烦。

### /conv/addsageconv.py

与[这里](https://github.com/dmlc/dgl/tree/master/examples/pytorch/graphsage)不同，我添加了``ginmean``作为一种aggregator，也就是用mean做reducer,再过一个线性层。其实代码里也加了cheb做aggregator但是效果不好，不使用。以后也可以自己添加更多aggregator做实验**[TODO]**

### /nets/chebs.py

由于``pool_size=2``和``pool_size=4``的参数有许多具体的差别，所以分成两个类

### /nets/gats.py

class GAT是DGL自己的实现，针对whole的训练模式(所有点都参与train和test)。class GATBlock是我实现的针对sample的GAT。因为DGL中sample之后产生的是Block，Block中只存seeds和被采样点之间的边。每个gat_layer算一个不同的block(layer坐标越大Block越小)。所以，除了在Block和Graph的数据格式上与DGL实现不同，在计算的对象上也不同（如下图所示）。在inference上，先根据batch_size做FullNeighbor的采样生成dataloader。之后针对dataloader的每一个iteration做一次GAT_Layers的前传。注意inference和forward不同。forward只需要计算传进来的Block即可，而inference需要根据传进来的DGLGraph自己生成Block

<img src="/Users/zhang/Library/Application Support/typora-user-images/image-20210509091404560.png" alt="image-20210509091404560" style="zoom:40%;" />

​																					GAT

<img src="/Users/zhang/Library/Application Support/typora-user-images/image-20210509093440392.png" alt="image-20210509093440392" style="zoom:40%;" />

​																				GATBlock

### /nets/sage.py

可以在这里修改每一层sageconv的aggregator的类型。在做inference时主循环是sage_layer，每个sage_layer把经过sample后的全图做一遍前传。这是因为GAT的隐藏层中有heads的问题，造成gat_layer每一层的输出和最终结果维数不同。

### /train/sample_train.py

``evalulate``给GraphSAGE使用，``gat_evaluate``因为要调用GAT的inference所以不同。下面是调用函数的具体参数

SAGE

|        | inductive                                      | transductive                                      |
| ------ | ---------------------------------------------- | ------------------------------------------------- |
| sample | ``inductive_sage_data``+``inductive_sage_run`` | ``load_ordinary_graph``+``transductive_sage_run`` |
| whole  |                                                | ``load_ordinary_graph``+``whole_sage_run``        |

GAT

|        | inductive                                               | transductive                              |
| ------ | ------------------------------------------------------- | ----------------------------------------- |
| sample | ``load_ordinary_graph``+``one_graph_inductive_gat_run`` | ``load_ordinary_graph``+``gat_run`        |
| whole  |                                                         | ``load_ordinary_graph``+``batch_gat_run`` |

### /utils/process.py

依赖于数据集(也许是``.json``文件但这里实现的都是DGL中已有的数据集)中有``feat``,``label``等标签。同时这里有``data_gpu``使用，同时也有一些图的sample操作，日后可以在这里加一些操作**[TODO]**

## TODO

1. GAT的whole+inductive
2. 模型与数据的调度(Graph,Block格式转换以及位置)
3. sample目前只能用DGL给的[三种方式](https://docs.dgl.ai/guide/minibatch.html)，能否有扩展
4. 目前只是计时，能否有更细致的[profiling](https://docs.nvidia.com/deeplearning/frameworks/pyprof-user-guide/index.html)
5. OGB的一些数据集还没试过(在``/graph/load_graph.py``中)
6. 代码里的不规范注释和一些引用。
7. 代码肯定有冗余，希望大家一起修改