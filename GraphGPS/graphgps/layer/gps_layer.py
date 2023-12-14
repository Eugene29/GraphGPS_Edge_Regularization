import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.graphgym.register as register
import torch_geometric.nn as pygnn
from performer_pytorch import SelfAttention
from torch_geometric.data import Batch
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import to_dense_batch

from graphgps.layer.bigbird_layer import SingleBigBirdLayer
from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvESLapPE

class GPSLayer(nn.Module):
    """Local MPNN + full graph attention x-former layer.
    """

    def __init__(self, dim_h,
                 local_gnn_type, global_model_type, num_heads, act='relu',
                 pna_degrees=None, equivstable_pe=False, dropout=0.0,
                 attn_dropout=0.0, layer_norm=False, batch_norm=True,
                 bigbird_cfg=None, log_attn_weights=False):
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
        self.activation = register.act_dict[act]

        self.log_attn_weights = log_attn_weights
        if log_attn_weights and global_model_type not in ['Transformer',
                                                          'BiasedTransformer']:
            raise NotImplementedError(
                f"Logging of attention weights is not supported "
                f"for '{global_model_type}' global attention model."
            )

        # Local message-passing model.
        self.local_gnn_with_edge_attr = True
        if local_gnn_type == 'None':
            self.local_model = None

        # MPNNs without edge attributes support.
        elif local_gnn_type == "GCN":
            self.local_gnn_with_edge_attr = False
            self.local_model = pygnn.GCNConv(dim_h, dim_h)
        elif local_gnn_type == 'GIN':
            self.local_gnn_with_edge_attr = False
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                   self.activation(),
                                   Linear_pyg(dim_h, dim_h))
            self.local_model = pygnn.GINConv(gin_nn)

        # MPNNs supporting also edge attributes.
        elif local_gnn_type == 'GENConv':
            self.local_model = pygnn.GENConv(dim_h, dim_h)
        elif local_gnn_type == 'GINE':
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                   self.activation(),
                                   Linear_pyg(dim_h, dim_h))
            if self.equivstable_pe:  # Use specialised GINE layer for EquivStableLapPE.
                self.local_model = GINEConvESLapPE(gin_nn)
            else:
                self.local_model = pygnn.GINEConv(gin_nn)
        elif local_gnn_type == 'GAT':
            self.local_model = pygnn.GATConv(in_channels=dim_h,
                                             out_channels=dim_h // num_heads,
                                             heads=num_heads,
                                             edge_dim=dim_h)
        elif local_gnn_type == 'PNA':
            # Defaults from the paper.
            # aggregators = ['mean', 'min', 'max', 'std']
            # scalers = ['identity', 'amplification', 'attenuation']
            aggregators = ['mean', 'max', 'sum']
            scalers = ['identity']
            deg = torch.from_numpy(np.array(pna_degrees))
            self.local_model = pygnn.PNAConv(dim_h, dim_h,
                                             aggregators=aggregators,
                                             scalers=scalers,
                                             deg=deg,
                                             edge_dim=min(128, dim_h),
                                             towers=1,
                                             pre_layers=1,
                                             post_layers=1,
                                             divide_input=False)
        elif local_gnn_type == 'CustomGatedGCN':
            self.local_model = GatedGCNLayer(dim_h, dim_h,
                                             dropout=dropout,
                                             residual=True,
                                             act=act,
                                             equivstable_pe=equivstable_pe)
        else:
            raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")
        self.local_gnn_type = local_gnn_type

        # Global attention transformer-style model.
        if global_model_type == 'None':
            self.self_attn = None
        elif global_model_type in ['Transformer', 'BiasedTransformer']:
            self.self_attn = torch.nn.MultiheadAttention(
                dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)
            # self.global_model = torch.nn.TransformerEncoderLayer(
            #     d_model=dim_h, nhead=num_heads,
            #     dim_feedforward=2048, dropout=0.1, activation=F.relu,
            #     layer_norm_eps=1e-5, batch_first=True)
        elif global_model_type == 'Performer':
            self.self_attn = SelfAttention(
                dim=dim_h, heads=num_heads,
                dropout=self.attn_dropout, causal=False)
        elif global_model_type == 'reg_Transformer':
            #### ignore num heads
#             self.self_attn = torch.nn.MultiheadAttention(
#                 dim_h, 1, dropout=self.attn_dropout, batch_first=True)
            self.self_attn = "none"
            dropout=self.attn_dropout
            self.Q = Linear_pyg(dim_h, dim_h)
            self.K = Linear_pyg(dim_h, dim_h)
            self.V = Linear_pyg(dim_h, dim_h)
#             self.O = Linear_pyg(dim_h, dim_h)
            
        elif global_model_type == "BigBird":
            bigbird_cfg.dim_hidden = dim_h
            bigbird_cfg.n_heads = num_heads
            bigbird_cfg.dropout = dropout
            self.self_attn = SingleBigBirdLayer(bigbird_cfg)
        else:
            raise ValueError(f"Unsupported global x-former model: "
                             f"{global_model_type}")
        self.global_model_type = global_model_type

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")

        # Normalization for MPNN and Self-Attention representations.
        if self.layer_norm:
            self.norm1_local = pygnn.norm.LayerNorm(dim_h)
            self.norm1_attn = pygnn.norm.LayerNorm(dim_h)
            # self.norm1_local = pygnn.norm.GraphNorm(dim_h)
            # self.norm1_attn = pygnn.norm.GraphNorm(dim_h)
            # self.norm1_local = pygnn.norm.InstanceNorm(dim_h)
            # self.norm1_attn = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
            self.norm1_attn = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # Feed Forward block.
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        self.act_fn_ff = self.activation()
        
#         self.ff_block2 = nn.Sequential(
#             nn.Linear(dim_h, dim_h * 2),
#             nn.Dropout(dropout),
#             self.activation(),
#             nn.Linear(dim_h * 2, dim_h),
#             nn.Dropout(dropout),
#             self.activation(),
#         )
        
        if self.layer_norm:
            self.norm2 = pygnn.norm.LayerNorm(dim_h)
            # self.norm2 = pygnn.norm.GraphNorm(dim_h)
            # self.norm2 = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)
#             self.norm3 = nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

#         self.forget_gate = Linear_pyg(dim_h, dim_h) ## sigmoid, applied to mpnn
#         self.gt_to_mpnn = Linear_pyg(dim_h, dim_h) ## tanh applied to gt
#         self.gt_to_mpnn_gate = Linear_pyg(dim_h, dim_h) ## sigmoid, applied to gt_to_mpnn
#         self.mpnn_to_gt = Linear_pyg(dim_h, dim_h) ## tanh applied to mpnn
#         self.mpnn_to_gt_gate = Linear_pyg(dim_h, dim_h) ## tanh applied to mpnn_to_gt
                                           
    def forward(self, batch,):
        if len(batch) != 2:
            inter_A = []
        else:
            batch, inter_A = batch
#             print(batch)

#         if hasattr(batch, "local"):
#             h = batch.local
#             h_in1 = batch.local
#         else:
        h = batch.x
        h_in1 = h  # for first residual connection
            
        h_out_list = []
        # Local MPNN with edge attributes.
        if self.local_model is not None:
            self.local_model: pygnn.conv.MessagePassing  # Typing hint.
            if self.local_gnn_type == 'CustomGatedGCN':
                es_data = None
                if self.equivstable_pe:
                    es_data = batch.pe_EquivStableLapPE
                local_out = self.local_model(Batch(batch=batch,
                                                   x=h,
                                                   edge_index=batch.edge_index,
                                                   edge_attr=batch.edge_attr,
                                                   pe_EquivStableLapPE=es_data))
                # GatedGCN does residual connection and dropout internally.
                h_local = local_out.x
                batch.edge_attr = local_out.edge_attr
            else:
                if self.local_gnn_with_edge_attr:
                    if self.equivstable_pe:
                        h_local = self.local_model(h,
                                                   batch.edge_index,
                                                   batch.edge_attr,
                                                   batch.pe_EquivStableLapPE)
                    else:
                        h_local = self.local_model(h,
                                                   batch.edge_index,
                                                   batch.edge_attr)
                else:
                    h_local = self.local_model(h, batch.edge_index)
                h_local = self.dropout_local(h_local)
                h_local = h_in1 + h_local  # Residual connection.

            if self.layer_norm:
                h_local = self.norm1_local(h_local, batch.batch)
            if self.batch_norm:
                h_local = self.norm1_local(h_local)
            h_out_list.append(h_local)
            
#         if hasattr(batch, "attn"):
# #             print("it's working attn")
#             h = batch.attn
#             h_in1 = batch.attn
            
        # Multi-head attention.
        if self.self_attn is not None:
            h_dense, mask = to_dense_batch(h, batch.batch)
            if self.global_model_type == 'Transformer':
                h_attn = self._sa_block(h_dense, None, ~mask)[mask]
            elif self.global_model_type == 'BiasedTransformer':
                # Use Graphormer-like conditioning, requires `batch.attn_bias`.
                h_attn = self._sa_block(h_dense, batch.attn_bias, ~mask)[mask]
            elif self.global_model_type == 'Performer':
                h_attn = self.self_attn(h_dense, mask=mask)[mask]
            elif self.global_model_type == 'reg_Transformer':
#                 h_attn, A = self._sa_block(h_dense, None, ~mask)
# #                 inter_A.append(A)
#                 h_attn = h_attn[mask]
                N = torch.bincount(batch.batch).max().item()
                Q, _ = to_dense_batch(self.Q(h), batch=batch.batch, max_num_nodes=N)
                K, _ = to_dense_batch(self.K(h), batch=batch.batch, max_num_nodes=N)
                V, mask = to_dense_batch(self.V(h), batch=batch.batch, max_num_nodes=N)
#                 ## 1. filling up nans as -inf before softmax. 2. will have nans for att weights due to how `to_dense_batch` works (softmaxes of only -inf)
                attention_score = Q @ torch.transpose(K, 2, 1) / (self.dim_h ** 0.5)
                if mask is not None:
                    mask2 = mask.unsqueeze(1).repeat(1, Q.size(1), 1)
                    attention_score = attention_score.masked_fill(mask2 == 0, float('-inf'))
#                 attention_score = torch.where(attention_score==float('-inf'), torch.tensor(float('-inf')), attention_score)
                attention_weights = F.softmax(attention_score, dim = -1)
# #                 attention_weights = torch.where(torch.isnan(attention_weights), torch.tensor(0.0), attention_weights)
# #                 attention_weights = F.sigmoid(attention_score / (self.emb_dim ** (1/2)))
# #                 intermediate_Adj_losses.append(attention_weights)
# #                 h_attn = torch.matmul(attention_weights, V)
                h_attn = attention_weights @ V
# #                 h_attn = self.O(h_attn)
# #                 if torch.isnan(h_attn).any():
# #                     print(h_attn)
# #                     raise err
                h_attn = h_attn[mask] ## reverting back to 2D
                
#                 print(f"h: {torch.isnan(h).any()}")
#                 print(f"h_attn: {torch.isnan(h_attn).any()}")
# #                 print(f"shouldn't be true: {torch.isnan(torch.tensor([0,0,0])).any()}")
# #                 print(f"should be true: {torch.isnan(torch.tensor([0,0,torch.nan])).any()}")
# #                 print(f"h_dense: {torch.isnan(h_dense).any()}")
#                 print(f"Q: {torch.isnan(Q).any()}")
#                 print(f"K: {torch.isnan(K).any()}")
#                 print(f"V: {torch.isnan(V).any()}")
# #                 print(f"as: {torch.isnan(attention_score).any()}")
# #                 print(f"aw: {torch.isnan(attention_weights).any()}")
#                 print("------")
#                 print(f"h_attn_mask: {torch.isnan(h_attn).any()}")
#                 inter_A.append(attention_weights)
                inter_A.append(F.sigmoid(attention_score))
    
            elif self.global_model_type == 'BigBird':
                h_attn = self.self_attn(h_dense, attention_mask=mask)
            else:
                raise RuntimeError(f"Unexpected {self.global_model_type}")
            h_attn = self.dropout_attn(h_attn)
            h_attn = h_in1 + h_attn  # Residual connection.
            if self.layer_norm:
                h_attn = self.norm1_attn(h_attn, batch.batch)
            if self.batch_norm:
                h_attn = self.norm1_attn(h_attn)
            h_out_list.append(h_attn)

        # Combine local and global outputs.
        # h = torch.cat(h_out_list, dim=-1)
        h = sum(h_out_list)
        #### communication block #### h_local, h_attn
#         forget_g = F.sigmoid(self.forget_gate(h_attn)) ## sigmoid gt, to be applied to mpnn
#         gt_to_mpnn_g = F.sigmoid(self.gt_to_mpnn_gate(h_attn)) ## sigmoid gt, to be applied to gt_to_mpnn
#         gt_to_mpnn = self.gt_to_mpnn(h_attn) * gt_to_mpnn_g ## tanh gt, multiplied with its gate 
#         ## mpnn gets updated first
#         h_local = forget_g * h_local
#         h_local = h_local + gt_to_mpnn
#         mpnn_to_gt_g = F.sigmoid(self.mpnn_to_gt_gate(h_local)) ## sigmoid to be applied to mpnn_to_gt
#         mpnn_to_gt = self.mpnn_to_gt(h_local) * mpnn_to_gt_g ## tanh applied to mpnn
#         h_attn = mpnn_to_gt
        
        # Feed Forward block.
        h = h + self._ff_block(h)
#         h_local = h_local + self._ff_block(h_local)
#         h_attn = h_attn + self.ff_block2(h_attn)

        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)
#             h_attn = self.norm2(h_attn)
#             h_local = self.norm3(h_local)

#         batch.attn = h_attn
#         batch.local = h_local
        
        batch.x = h
#         batch.x = h_attn
        if self.global_model_type == 'reg_Transformer':
#             print("hey, how are you")
            return (batch, inter_A)
        return batch        
        
    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block.
        """
        if self.global_model_type == 'reg_Transformer':
            x, A = self.self_attn(x, x, x,
                               attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask,
                               need_weights=True)
            return x, A
        elif not self.log_attn_weights:
            x = self.self_attn(x, x, x,
                               attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask,
                               need_weights=False)[0]
        else:
            # Requires PyTorch v1.11+ to support `average_attn_weights=False`
            # option to return attention weights of individual heads.
            x, A = self.self_attn(x, x, x,
                                  attn_mask=attn_mask,
                                  key_padding_mask=key_padding_mask,
                                  need_weights=True,
                                  average_attn_weights=False)
            self.attn_weights = A.detach().cpu()
        return x

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def extra_repr(self):
        s = f'summary: dim_h={self.dim_h}, ' \
            f'local_gnn_type={self.local_gnn_type}, ' \
            f'global_model_type={self.global_model_type}, ' \
            f'heads={self.num_heads}'
        return s
