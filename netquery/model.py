import torch
import torch.nn as nn

import random
from netquery.graph import _reverse_relation
from netquery.data_utils import RGCNQueryDataset

EPS = 10e-6

"""
End-to-end autoencoder models for representation learning on
heteregenous graphs/networks
"""

class MetapathEncoderDecoder(nn.Module):
    """
    Encoder decoder model that reasons over metapaths
    """

    def __init__(self, graph, enc, dec):
        """
        graph -- simple graph object; see graph.py
        enc --- an encoder module that generates embeddings (see encoders.py) 
        dec --- an decoder module that predicts compositional relationships, i.e. metapaths, between nodes given embeddings. (see decoders.py)
                Note that the decoder must be an *compositional/metapath* decoder (i.e., with name Metapath*.py)
        """
        super(MetapathEncoderDecoder, self).__init__()
        self.enc = enc
        self.dec = dec
        self.graph = graph

    def forward(self, nodes1, nodes2, rels):
        """
        Returns a vector of 'relationship scores' for pairs of nodes being connected by the given metapath (sequence of relations).
        Essentially, the returned scores are the predicted likelihood of the node pairs being connected
        by the given metapath, where the pairs are given by the ordering in nodes1 and nodes2,
        i.e. the first node id in nodes1 is paired with the first node id in nodes2.
        """
        return self.dec.forward(self.enc.forward(nodes1, rels[0][0]), 
                self.enc.forward(nodes2, rels[-1][-1]),
                rels)

    def margin_loss(self, nodes1, nodes2, rels):
        """
        Standard max-margin based loss function.
        Maximizes relationaship scores for true pairs vs negative samples.
        """
        affs = self.forward(nodes1, nodes2, rels)
        neg_nodes = [random.randint(1,len(self.graph.adj_lists[_reverse_relation[rels[-1]]])-1) for _ in range(len(nodes1))]
        neg_affs = self.forward(nodes1, neg_nodes,
            rels)
        margin = 1 - (affs - neg_affs)
        margin = torch.clamp(margin, min=0)
        loss = margin.mean()
        return loss 

class QueryEncoderDecoder(nn.Module):
    """
    Encoder decoder model that reasons about edges, metapaths and intersections
    """

    def __init__(self, graph, enc, path_dec, inter_dec):
        super(QueryEncoderDecoder, self).__init__()
        self.enc = enc
        self.path_dec = path_dec
        self.inter_dec = inter_dec
        self.graph = graph
        self.cos = nn.CosineSimilarity(dim=0)

    def forward(self, formula, queries, target_nodes):
        if formula.query_type == "1-chain" or formula.query_type == "2-chain" or formula.query_type == "3-chain":
            # a chain is simply a call to the path decoder
            return self.path_dec.forward(
                    self.enc.forward(target_nodes, formula.target_mode),
                    self.enc.forward([query.anchor_nodes[0] for query in queries], formula.anchor_modes[0]),
                    formula.rels)
        elif formula.query_type == "2-inter" or formula.query_type == "3-inter" or formula.query_type == "3-inter_chain":
            target_embeds = self.enc(target_nodes, formula.target_mode)

            embeds1 = self.enc([query.anchor_nodes[0] for query in queries], formula.anchor_modes[0])
            embeds1 = self.path_dec.project(embeds1, _reverse_relation(formula.rels[0]))

            embeds2 = self.enc([query.anchor_nodes[1] for query in queries], formula.anchor_modes[1])
            if len(formula.rels[1]) == 2:
                for i_rel in formula.rels[1][::-1]:
                    embeds2 = self.path_dec.project(embeds2, _reverse_relation(i_rel))
            else:
                    embeds2 = self.path_dec.project(embeds2, _reverse_relation(formula.rels[1]))

            if formula.query_type == "3-inter":
                embeds3 = self.enc([query.anchor_nodes[2] for query in queries], formula.anchor_modes[2])
                embeds3 = self.path_dec.project(embeds3, _reverse_relation(formula.rels[2]))

                query_intersection = self.inter_dec(embeds1, embeds2, formula.target_mode, embeds3)
            else:
                query_intersection = self.inter_dec(embeds1, embeds2, formula.target_mode)
            scores = self.cos(target_embeds, query_intersection)
            return scores
        elif formula.query_type == "3-chain_inter":
            target_embeds = self.enc(target_nodes, formula.target_mode)

            embeds1 = self.enc([query.anchor_nodes[0] for query in queries], formula.anchor_modes[0])
            embeds1 = self.path_dec.project(embeds1, _reverse_relation(formula.rels[1][0]))
            embeds2 = self.enc([query.anchor_nodes[1] for query in queries], formula.anchor_modes[1])
            embeds2 = self.path_dec.project(embeds2, _reverse_relation(formula.rels[1][1]))
            query_intersection = self.inter_dec(embeds1, embeds2, formula.rels[0][-1])
            query_intersection = self.path_dec.project(query_intersection, _reverse_relation(formula.rels[0]))
            scores = self.cos(target_embeds, query_intersection)
            return scores


    def margin_loss(self, formula, queries, hard_negatives=False, margin=1):
        if not "inter" in formula.query_type and hard_negatives:
            raise Exception("Hard negative examples can only be used with intersection queries")
        elif hard_negatives:
            neg_nodes = [random.choice(query.hard_neg_samples) for query in queries]
        elif formula.query_type == "1-chain":
            neg_nodes = [random.choice(self.graph.full_lists[formula.target_mode]) for _ in queries]
        else:
            neg_nodes = [random.choice(query.neg_samples) for query in queries]

        affs = self.forward(formula, queries, [query.target_node for query in queries])
        neg_affs = self.forward(formula, queries, neg_nodes)
        loss = margin - (affs - neg_affs)
        loss = torch.clamp(loss, min=0)
        loss = loss.mean()
        return loss 

class SoftAndEncoderDecoder(nn.Module):
    """
    Encoder decoder model that reasons about edges, metapaths and intersections
    """

    def __init__(self, graph, enc, path_dec):
        super(SoftAndEncoderDecoder, self).__init__()
        self.enc = enc
        self.path_dec = path_dec
        self.graph = graph
        self.cos = nn.CosineSimilarity(dim=0)

    def forward(self, formula, queries, source_nodes):
        if formula.query_type == "1-chain":
            # a chain is simply a call to the path decoder
            return self.path_dec.forward(
                    self.enc.forward(source_nodes, formula.target_mode), 
                    self.enc.forward([query.anchor_nodes[0] for query in queries], formula.anchor_modes[0]),
                    formula.rels)
        elif formula.query_type == "2-inter" or formula.query_type == "3-inter":
            target_embeds = self.enc(source_nodes, formula.target_mode)

            embeds1 = self.enc([query.anchor_nodes[0] for query in queries], formula.anchor_modes[0])
            embeds1 = self.path_dec.project(embeds1, _reverse_relation(formula.rels[0]))

            embeds2 = self.enc([query.anchor_nodes[1] for query in queries], formula.anchor_modes[1])
            if len(formula.rels[1]) == 2:
                for i_rel in formula.rels[1][::-1]:
                    embeds2 = self.path_dec.project(embeds2, _reverse_relation(i_rel))
            else:
                    embeds2 = self.path_dec.project(embeds2, _reverse_relation(formula.rels[1]))

            scores1 = self.cos(target_embeds, embeds1)
            scores2 = self.cos(target_embeds, embeds2)
            if formula.query_type == "3-inter":
                embeds3 = self.enc([query.anchor_nodes[2] for query in queries], formula.anchor_modes[2])
                embeds3 = self.path_dec.project(embeds3, _reverse_relation(formula.rels[2]))
                scores3 = self.cos(target_embeds, embeds2)
                scores = scores1 * scores2 * scores3
            else:
                scores = scores1 * scores2
            return scores
        else:
            raise Exception("Query type not supported for this model.")

    def margin_loss(self, formula, queries, hard_negatives=False, margin=1):
        if not "inter" in formula.query_type and hard_negatives:
            raise Exception("Hard negative examples can only be used with intersection queries")
        elif hard_negatives:
            neg_nodes = [random.choice(query.hard_neg_samples) for query in queries]
        elif formula.query_type == "1-chain":
            neg_nodes = [random.choice(self.graph.full_lists[formula.target_mode]) for _ in queries]
        else:
            neg_nodes = [random.choice(query.neg_samples) for query in queries]

        affs = self.forward(formula, queries, [query.target_node for query in queries])
        neg_affs = self.forward(formula, queries, neg_nodes)
        loss = margin - (affs - neg_affs)
        loss = torch.clamp(loss, min=0)
        loss = loss.mean()
        return loss 


from torch_scatter import scatter_add, scatter_max, scatter_mean, scatter_min
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Parameter as Param
from torch_geometric.nn import inits


class RGCNConv(MessagePassing):
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_0 \cdot \mathbf{x}_i +
        \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int): Number of bases used for basis-decomposition.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_relations,
                 num_bases,
                 bias=True):
        super(RGCNConv, self).__init__('add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        if num_bases == 0:
            self.basis = Param(torch.Tensor(num_relations, in_channels, out_channels))
            self.att = None
        else:
            self.basis = Param(torch.Tensor(num_bases, in_channels, out_channels))
            self.att = Param(torch.Tensor(num_relations, num_bases))
        self.root = Param(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Param(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.att is None:
            size = self.num_relations * self.in_channels
        else:
            size = self.num_bases * self.in_channels
            inits.uniform(size, self.att)

        inits.uniform(size, self.basis)
        inits.uniform(size, self.root)
        inits.uniform(size, self.bias)

    def forward(self, x, edge_index, edge_type, edge_norm=None):
        """"""
        if x is None:
            x = torch.arange(
                edge_index.max().item() + 1,
                dtype=torch.long,
                device=edge_index.device)

        return self.propagate(
            edge_index, x=x, edge_type=edge_type, edge_norm=edge_norm)

    def message(self, x_j, edge_type, edge_norm):
        if self.att is None:
            w = self.basis.view(self.num_relations, -1)
        else:
            w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))

        if x_j.dtype == torch.long:
            w = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + x_j
            out = torch.index_select(w, 0, index)
            return out if edge_norm is None else out * edge_norm.view(-1, 1)
        else:
            w = w.view(self.num_relations, self.in_channels, self.out_channels)
            w = torch.index_select(w, 0, edge_type)
            out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)
            return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):
        if x.dtype == torch.long:
            out = aggr_out + self.root
        else:
            out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)


class RGCNEncoderDecoder(nn.Module):
    def __init__(self, graph, embed_dim, num_bases, readout='sum',
                 scatter_op='add', dropout=0, weight_decay=1e-3):
        super(RGCNEncoderDecoder, self).__init__()
        self.num_entities = sum(map(len, graph.full_sets.values()))
        self.graph = graph
        self.emb_dim = embed_dim

        # TODO: in the future this can be an external, more complex module
        #   that implements things like normalization and node aggregation.
        #   See encoders.py
        self.entity_embs = nn.Embedding(self.num_entities, self.emb_dim)
        self.mode_embeddings = nn.Embedding(len(graph.mode_weights), self.emb_dim)

        self.mode_ids = {}
        mode_id = 0
        for mode in graph.mode_weights:
            self.mode_ids[mode] = mode_id

        self.rel_ids = {}
        id_rel = 0
        for r1 in graph.relations:
            for r2 in graph.relations[r1]:
                rel = (r1, r2[1], r2[0])
                self.rel_ids[rel] = id_rel
                id_rel += 1

        self.rgcn = RGCNConv(in_channels=self.emb_dim, out_channels=self.emb_dim,
                             num_relations=len(graph.rel_edges),
                             num_bases=num_bases)

        if scatter_op == 'add':
            scatter_fn = scatter_add
        elif scatter_op == 'max':
            scatter_fn = scatter_max
        elif scatter_op == 'min':
            scatter_fn = scatter_min
        elif scatter_op == 'mean':
            scatter_fn = scatter_mean
        else:
            raise ValueError(f'Unknown scatter op {scatter_op}')

        if readout == 'sum':
            self.readout = self.sum_readout
        elif readout == 'max':
            self.readout = self.max_readout
        elif readout == 'mlp':
            self.readout = MLPReadout(self.emb_dim, scatter_fn)
        elif readout == 'targetmlp':
            self.readout = TargetMLPReadout(self.emb_dim, scatter_fn)
        else:
            raise ValueError(f'Unknown readout function {readout}')

        self.dropout = nn.Dropout(dropout)
        self.weight_decay = weight_decay

    def sum_readout(self, embs, batch_idx, *args, **kwargs):
        return scatter_add(embs, batch_idx, dim=0)

    def max_readout(self, embs, batch_idx, *args, **kwargs):
        out, argmax = scatter_max(embs, batch_idx, dim=0)
        return out

    def forward(self, formula, queries, target_nodes):
        query_graph = RGCNQueryDataset.get_query_graph(formula, queries,
                                                       self.rel_ids, self.mode_ids)
        anchor_ids, var_ids, edge_index, edge_type, batch_idx = query_graph
        targets = torch.tensor(target_nodes, dtype=torch.long)

        device = next(self.parameters()).device
        anchor_ids = anchor_ids.to(device)
        var_ids = var_ids.to(device)
        edge_index = edge_index.to(device)
        edge_type = edge_type.to(device)
        batch_idx = batch_idx.to(device)
        targets = targets.to(device)

        return self.forward_train(anchor_ids, var_ids, edge_index, edge_type,
                                  batch_idx, targets)

    def forward_train(self, anchor_ids, var_ids, edge_index, edge_type, batch_idx, targets):
        batch_size = anchor_ids.shape[0]
        n_anchors, n_vars = anchor_ids.shape[1], var_ids.shape[1]
        n_nodes = n_anchors + n_vars

        anchor_embs = self.entity_embs(anchor_ids)
        var_embs = self.mode_embeddings(var_ids)
        x = torch.cat((anchor_embs, var_embs), dim=1).reshape(-1, self.emb_dim)

        x = F.relu(self.rgcn(x, edge_index, edge_type))
        x = self.dropout(self.rgcn(x, edge_index, edge_type))
        x = self.readout(x, batch_idx, batch_size, n_nodes, n_anchors)

        target_embeds = self.entity_embs(targets)
        scores = F.cosine_similarity(x, target_embeds, dim=1)

        return scores

    def margin_loss(self, anchor_ids, var_ids, edge_index, edge_type, batch_idx,
                    targets, neg_targets, margin=1):
        device = next(self.parameters()).device
        anchor_ids = anchor_ids.to(device)
        var_ids = var_ids.to(device)
        edge_index = edge_index.to(device)
        edge_type = edge_type.to(device)
        batch_idx = batch_idx.to(device)
        targets = targets.to(device)
        neg_targets = neg_targets.to(device)
        affs = self.forward_train(anchor_ids, var_ids, edge_index, edge_type, batch_idx, targets)
        neg_affs = self.forward_train(anchor_ids, var_ids, edge_index, edge_type, batch_idx, neg_targets)
        loss = margin - (affs - neg_affs)
        loss = torch.clamp(loss, min=0)
        loss = loss.mean()

        if isinstance(self.readout, nn.Module):
            l2_reg = 0
            for param in self.readout.parameters():
                l2_reg += torch.norm(param)

            loss += self.weight_decay * l2_reg

        return loss


class MLPReadout(nn.Module):
    def __init__(self, dim, scatter_fn):
        super(MLPReadout, self).__init__()
        self.layers = nn.Sequential(nn.Linear(in_features=dim, out_features=dim),
                                    nn.ReLU(),
                                    nn.Linear(in_features=dim, out_features=dim),
                                    nn.ReLU())
        self.scatter_fn = scatter_fn

    def forward(self, embs, batch_idx, *args, **kwargs):
        x = self.layers(embs)
        x = self.scatter_fn(x, batch_idx, dim=0)

        # If scatter_fn is max or min, values and indices are returned
        if isinstance(x, tuple):
            x = x[0]

        return x


class TargetMLPReadout(nn.Module):
    def __init__(self, dim, scatter_fn):
        super(TargetMLPReadout, self).__init__()
        self.layers = nn.Sequential(nn.Linear(in_features=2*dim, out_features=dim),
                                    nn.ReLU(),
                                    nn.Linear(in_features=dim, out_features=dim),
                                    nn.ReLU())
        self.scatter_fn = scatter_fn

    def forward(self, embs, batch_idx, batch_size, num_nodes, num_anchors):
        device = embs.device

        non_target_idx = torch.ones(num_nodes, dtype=torch.uint8)
        non_target_idx[num_anchors] = 0
        non_target_idx.to(device)

        batch_idx = batch_idx.reshape(batch_size, -1)
        batch_idx = batch_idx[:, non_target_idx].reshape(-1)

        embs = embs.reshape(batch_size, num_nodes, -1)
        non_targets = embs[:, non_target_idx]
        targets = embs[:, 1 - non_target_idx].expand_as(non_targets)

        x = torch.cat((targets, non_targets), dim=-1)
        x = x.reshape(batch_size * (num_nodes - 1), -1).contiguous()

        x = self.layers(x)
        x = self.scatter_fn(x, batch_idx, dim=0)

        # If scatter_fn is max or min, values and indices are returned
        if isinstance(x, tuple):
            x = x[0]

        return x


if __name__ == '__main__':
    batch_size = 64
    num_nodes = 4
    num_anchors = 2
    emb_dim = 16
    embs = torch.rand(batch_size * num_nodes, emb_dim)
    batch_idx = torch.arange(start=0, end=batch_size).unsqueeze(dim=-1)
    batch_idx = batch_idx.expand(-1, num_nodes).reshape(-1)
    model = TargetMLPReadout(emb_dim)
    out = model(embs, batch_idx, batch_size, num_nodes, num_anchors)
    print(out.shape)
