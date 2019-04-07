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


from torch_geometric.nn import RGCNConv
from torch_scatter import scatter_add, scatter_max
import torch.nn.functional as F

class RGCNEncoderDecoder(nn.Module):
    def __init__(self, graph, enc, readout='sum'):
        super(RGCNEncoderDecoder, self).__init__()
        self.enc = enc
        self.graph = graph
        self.emb_dim = graph.feature_dims[next(iter(graph.feature_dims))]
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
                             num_relations=len(graph.rel_edges), num_bases=10)

        if readout == 'sum':
            self.readout = self.sum_readout
        elif readout == 'max':
            self.readout = self.max_readout
        elif readout == 'mlp':
            self.readout = MLPReadout(self.emb_dim)
        elif readout == 'targetmlp':
            self.readout = TargetMLPReadout(self.emb_dim)
        else:
            raise ValueError(f'Unknown readout function {readout}')

    def sum_readout(self, embs, batch_idx, *args, **kwargs):
        return scatter_add(embs, batch_idx, dim=0)

    def max_readout(self, embs, batch_idx, *args, **kwargs):
        out, argmax = scatter_max(embs, batch_idx, dim=0)
        return out

    def forward(self, formula, queries, target_nodes,
                anchor_ids=None, var_ids=None, q_graphs=None):

        if anchor_ids is None or var_ids is None or q_graphs is None:
            anchor_ids, var_ids, q_graphs = RGCNQueryDataset.get_query_graph(formula,
                                                                          queries,
                                                                          self.rel_ids,
                                                                          self.mode_ids)
        device = next(self.parameters()).device
        var_ids = var_ids.to(device)
        q_graphs = q_graphs.to(device)

        batch_size, n_anchors = anchor_ids.shape
        n_vars = var_ids.shape[0]
        num_nodes = n_anchors + n_vars

        x = torch.empty(batch_size, num_nodes, self.emb_dim).to(var_ids.device)
        for i, anchor_mode in enumerate(formula.anchor_modes):
            x[:, i] = self.enc(anchor_ids[:, i], anchor_mode).t()
        x[:, n_anchors:] = self.mode_embeddings(var_ids)
        x = x.reshape(-1, self.emb_dim)
        q_graphs.x = x

        out = F.relu(self.rgcn(q_graphs.x, q_graphs.edge_index, q_graphs.edge_type))
        out = self.rgcn(out, q_graphs.edge_index, q_graphs.edge_type)
        out = self.readout(out, q_graphs.batch, batch_size, num_nodes, n_anchors)

        target_embeds = self.enc(target_nodes, formula.target_mode).t()
        scores = F.cosine_similarity(out, target_embeds, dim=1)

        return scores

    def margin_loss(self, formula, queries, anchor_ids=None, var_ids=None, q_graphs=None,
                    hard_negatives=False, margin=1):
        if not "inter" in formula.query_type and hard_negatives:
            raise Exception("Hard negative examples can only be used with intersection queries")
        elif hard_negatives:
            neg_nodes = [random.choice(query.hard_neg_samples) for query in queries]
        elif formula.query_type == "1-chain":
            neg_nodes = [random.choice(self.graph.full_lists[formula.target_mode]) for _ in queries]
        else:
            neg_nodes = [random.choice(query.neg_samples) for query in queries]

        affs = self.forward(formula, queries, [query.target_node for query in queries],
                            anchor_ids, var_ids, q_graphs)
        neg_affs = self.forward(formula, queries, neg_nodes,
                                anchor_ids, var_ids, q_graphs)
        loss = margin - (affs - neg_affs)
        loss = torch.clamp(loss, min=0)
        loss = loss.mean()
        return loss


class MLPReadout(nn.Module):
    def __init__(self, dim):
        super(MLPReadout, self).__init__()
        self.layers = nn.Sequential(nn.Linear(in_features=dim, out_features=dim),
                                    nn.ReLU(),
                                    nn.Linear(in_features=dim, out_features=dim),
                                    nn.ReLU())

    def forward(self, embs, batch_idx, *args, **kwargs):
        x = self.layers(embs)
        return scatter_add(x, batch_idx, dim=0)


class TargetMLPReadout(nn.Module):
    def __init__(self, dim):
        super(TargetMLPReadout, self).__init__()
        self.layers = nn.Sequential(nn.Linear(in_features=2*dim, out_features=dim),
                                    nn.ReLU(),
                                    nn.Linear(in_features=dim, out_features=dim),
                                    nn.ReLU())

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
        return scatter_add(x, batch_idx, dim=0)


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
