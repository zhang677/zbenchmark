"""Torch Module for Chebyshev Spectral Graph Convolution layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
import torch.nn.functional as F
import dgl

from dgl.utils import dgl_warning
from dgl import laplacian_lambda_max, broadcast_nodes, function as fn


class ChebConv(nn.Module):
    """
    It is changed according to the coordinate.py
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 k,
                 activation=F.relu,  
                 bias=True,
                 is_mnist = False):
                     
        super(ChebConv, self).__init__()
        self._k = k
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.linear = nn.Linear(k * in_feats, out_feats, bias)
        self.is_mnist = is_mnist

    def forward(self, graph, feat, lambda_max=None):
        r"""

        Description
        -----------
        Compute ChebNet layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.
        lambda_max : list or tensor or None, optional.
            A list(tensor) with length :math:`B`, stores the largest eigenvalue
            of the normalized laplacian of each individual graph in ``graph``,
            where :math:`B` is the batch size of the input graph. Default: None.
            If None, this method would compute the list by calling
            ``dgl.laplacian_lambda_max``.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        def unnLaplacian(feat, D_invsqrt, graph):
            
            """ Operation Feat * D^-1/2 A D^-1/2 但是如果写成矩阵乘法：D^-1/2 A D^-1/2 Feat"""
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return graph.ndata.pop('h') * D_invsqrt

        with graph.local_scope():
            #一点修改，这是原来的代码
            if self.is_mnist:
                graph.update_all(fn.copy_edge('v','m'), fn.sum('m','h')) # 'v'与coordinate.py有关
                D_invsqrt = th.pow(graph.ndata.pop('h').float().clamp(min=1), -0.5).unsqueeze(-1).to(feat.device)
                
            #D_invsqrt = th.pow(graph.in_degrees().float().clamp(
            #   min=1), -0.5).unsqueeze(-1).to(feat.device)
            #print("in_degree : ",graph.in_degrees().shape)
            else:
                D_invsqrt = th.pow(graph.in_degrees().float().clamp(min=1), -0.5).unsqueeze(-1).to(feat.device)
            #print("D_invsqrt : ",D_invsqrt.shape)
            #print("ndata : ",graph.ndata['h'].shape)
            if lambda_max is None:
                try:
                    lambda_max = laplacian_lambda_max(graph)
                except BaseException:
                    # if the largest eigenvalue is not found
                    dgl_warning(
                        "Largest eigonvalue not found, using default value 2 for lambda_max",
                        RuntimeWarning)
                    lambda_max = th.Tensor(2).to(feat.device)

            if isinstance(lambda_max, list):
                lambda_max = th.Tensor(lambda_max).to(feat.device)
            if lambda_max.dim() == 1:
                lambda_max = lambda_max.unsqueeze(-1)  # (B,) to (B, 1)

            # broadcast from (B, 1) to (N, 1)
            lambda_max = broadcast_nodes(graph, lambda_max)
            re_norm = 2. / lambda_max

            # X_0 is the raw feature, Xt refers to the concatenation of X_0, X_1, ... X_t
            Xt = X_0 = feat

            # X_1(f)
            if self._k > 1:
                h = unnLaplacian(X_0, D_invsqrt, graph)
                X_1 = - re_norm * h + X_0 * (re_norm - 1)
                # Concatenate Xt and X_1
                Xt = th.cat((Xt, X_1), 1)

            # Xi(x), i = 2...k
            for _ in range(2, self._k):
                h = unnLaplacian(X_1, D_invsqrt, graph)
                X_i = - 2 * re_norm * h + X_1 * 2 * (re_norm - 1) - X_0
                # Concatenate Xt and X_i
                Xt = th.cat((Xt, X_i), 1)
                X_1, X_0 = X_i, X_1

            # linear projection
            h = self.linear(Xt)

            # activation
            if self.activation:
                h = self.activation(h)
        #print('ChebConv.py Line163 h : ',h.shape)
        return h
