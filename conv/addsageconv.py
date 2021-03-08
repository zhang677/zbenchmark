"""Torch Module for GraphSAGE layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch
from torch import nn
from torch.nn import functional as F

import dgl
import dgl.function as fn
from dgl.utils import expand_as_pair, check_eq_shape
from dgl.utils import dgl_warning
from dgl import laplacian_lambda_max, broadcast_nodes
class SAGEConv(nn.Module):
    """
    Add aggrrgate type : ginmean and cheb(doesn't perform well)
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 activation=None):
        super(SAGEConv, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        # aggregator type: mean/pool/lstm/gcn
        if self._aggre_type == 'cheb': # default using activation TODO
            self._cheb_k = 2 # it should be consistent with the sampling since every block is a bipartitte graph 
            # in spectral method there's no fc_neigh or fc_self
            self._cheb_linear = nn.Linear(self._cheb_k * self._in_src_feats, out_feats)
            
        if aggregator_type == 'ginmean':
            self._gin_reducer = fn.mean
            self.fc_gin = nn.Linear(self._in_src_feats, out_feats) # default apply_func is nn.linear
        if aggregator_type == 'pool':
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type == 'lstm':
            self.lstm = nn.LSTM(self._in_src_feats, self._in_src_feats, batch_first=True)
        if aggregator_type != 'gcn' and aggregator_type != 'ginmean' and aggregator_type != 'cheb':
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        if aggregator_type !='ginmean' and aggregator_type != 'cheb':
            self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The linear weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The LSTM module is using xavier initialization method for its weights.
        """
        gain = nn.init.calculate_gain('relu')
        if self._aggre_type == 'ginmean':
            nn.init.xavier_uniform_(self.fc_gin.weight, gain=gain)
            init_eps = 0 # default set eps learnable, initialized by 0
            self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))
        if self._aggre_type == 'cheb':
            nn.init.xavier_uniform_(self._cheb_linear.weight, gain=gain)
        if self._aggre_type == 'pool':
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == 'lstm':
            self.lstm.reset_parameters()
        if self._aggre_type != 'gcn' and self._aggre_type != 'ginmean' and self._aggre_type != 'cheb':
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        if self._aggre_type != 'ginmean' and self._aggre_type != 'cheb':
            nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _lstm_reducer(self, nodes):
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox['m'] # (B, L, D)
        batch_size = m.shape[0]
        h = (m.new_zeros((1, batch_size, self._in_src_feats)),
             m.new_zeros((1, batch_size, self._in_src_feats)))
        _, (rst, _) = self.lstm(m, h)
        return {'neigh': rst.squeeze(0)}
    def forward(self, graph, feat):
        r"""

        Description
        -----------
        Compute GraphSAGE layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]

            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.number_of_edges() == 0:
                graph.dstdata['neigh'] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats).to(feat_dst)

            if self._aggre_type == 'mean':
                graph.srcdata['h'] = feat_src
                graph.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))
                h_neigh = graph.dstdata['neigh']
            elif self._aggre_type == 'gcn':
                check_eq_shape(feat)
                graph.srcdata['h'] = feat_src
                graph.dstdata['h'] = feat_dst     # same as above if homogeneous
                graph.update_all(fn.copy_src('h', 'm'), fn.sum('m', 'neigh'))
                # divide in_degrees
                degs = graph.in_degrees().to(feat_dst)
                h_neigh = (graph.dstdata['neigh'] + graph.dstdata['h']) / (degs.unsqueeze(-1) + 1)
            elif self._aggre_type == 'pool':
                graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
                graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'neigh'))
                h_neigh = graph.dstdata['neigh']
            elif self._aggre_type == 'lstm':
                graph.srcdata['h'] = feat_src
                graph.update_all(fn.copy_src('h', 'm'), self._lstm_reducer)
                h_neigh = graph.dstdata['neigh']
            elif self._aggre_type == 'ginmean':
                graph.srcdata['h'] = feat_src
                graph.update_all(fn.copy_src('h', 'm'), self._gin_reducer('m', 'neigh'))
                h_neigh = graph.dstdata['neigh']                
            elif self._aggre_type == 'cheb':
                def unnLaplacian(feat, D_invsqrt_left,D_invsqrt_right, graph):
                    """ Operation Feat * D^-1/2 A D^-1/2 但是如果写成矩阵乘法：D^-1/2 A D^-1/2 Feat"""
                    #tmp = torch.zeros((D_invsqrt.shape[0],D_invsqrt.shape[0])).to(graph.device)
                    # sparse tensor没有broadcast机制，最后还依赖于srcnode在feat中从0开始连续排布
                    #print("adj : ",graph.adj(transpose=False,ctx = graph.device).shape)
                    #graph.srcdata['h'] = (torch.mm((graph.adj(transpose=False,ctx = graph.device)),(feat * D_invsqrt)))*D_invsqrt[::graph.number_of_dst_nodes()]
                    #graph.update_all(fn.copy_src('h', 'm'), fn.sum('m', 'h'))
                    #return graph.srcdata['h']
                    graph.srcdata['h'] = feat * D_invsqrt_right # feat is srcfeat
                    graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                    return graph.dstdata.pop('h') * D_invsqrt_left
                D_invsqrt_right = torch.pow(graph.out_degrees().float().clamp(min=1), -0.5).unsqueeze(-1) 
                D_invsqrt_left = torch.pow(graph.in_degrees().float().clamp(min=1), -0.5).unsqueeze(-1) 
                #print("D_invsqrt shape: ",D_invsqrt.shape)
                #print(graph.__dict__)
                #print(dir(graph))
                #graph.srcdata['h']=feat_src
                #graph.dstdata['h']=feat_dst
                #g = dgl.to_homogeneous(graph,ndata=['h'])
                #dgl._ffi.base.DGLError: Expect number of features to match number of nodes (len(u)). Got 70 and 76 instead.
                #print(g)
                # since the block is different every time so it's safe to call dgl's method every time instead of calculating the l_m ahead
                try:
                    lambda_max = laplacian_lambda_max(graph)
                except BaseException:
                    # if the largest eigenvalue is not found
                    dgl_warning(
                        "Largest eigonvalue not found, using default value 2 for lambda_max",
                        RuntimeWarning)
                    lambda_max = torch.tensor(2)# .to(feat.device)
                if isinstance(lambda_max, list):
                    lambda_max = torch.tensor(lambda_max)# .to(feat.device)
                if lambda_max.dim() == 1:
                    lambda_max = lambda_max.unsqueeze(-1)  # (B,) to (B, 1)
                # broadcast from (B, 1) to (N, 1)
                # lambda_max = lambda_max * torch.ones((feat.shape[0],1))
                #re_norm = (2 / lambda_max ) * torch.ones((graph.number_of_dst_nodes(),1)).to(graph.device)
                re_norm = (2 / lambda_max.to(graph.device) ) * torch.ones((graph.number_of_dst_nodes(),1),device=graph.device)
                self._cheb_Xt = X_0 = feat_dst
                graph.srcdata['h'] = feat_src * D_invsqrt_right # feat is srcfeat
                graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                X_1 = - re_norm * graph.dstdata['h']*D_invsqrt_left + X_0 * (re_norm - 1)
                self._cheb_Xt = torch.cat((self._cheb_Xt, X_1.float()), 1)
            else:
                raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))
                

            # GraphSAGE GCN does not require fc_self.
            if self._aggre_type == 'gcn':
                rst = self.fc_neigh(h_neigh)
            elif self._aggre_type == 'ginmean':
                rst = (1 + self.eps) * h_self + h_neigh
                rst = self.fc_gin(rst)
                if self.norm is not None:
                    rst = self.norm(rst)               
                return rst
            elif self._aggre_type == 'cheb':
                rst = self._cheb_linear(self._cheb_Xt)
            else:
                rst = self.fc_self(h_self) + self.fc_neigh(h_neigh)
            
            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst
