import torch
import torch.nn as nn

import random
from netquery.graph import _reverse_relation

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


from torch_geometric.data import Data, Batch
from torch_geometric.nn import RGCNConv
from torch_scatter import scatter_add, scatter_mean, scatter_min, scatter_max, scatter_std
import torch.nn.functional as F

class RGCNEncoderDecoder(nn.Module):
    query_edge_indices = {'1-chain': [[0],
                                      [1]],
                          '2-chain': [[0, 2],
                                      [2, 1]],
                          '3-chain': [[0, 3, 2],
                                      [3, 2, 1]],
                          '2-inter': [[0, 1],
                                      [2, 2]],
                          '3-inter': [[0, 1, 2],
                                      [3, 3, 3]],
                          '3-inter_chain': [[0, 1, 3],
                                            [2, 3, 2]],
                          '3-chain_inter': [[0, 1, 3],
                                            [3, 3, 2]]}

    query_edge_label_idx = {'1-chain': [0],
                            '2-chain': [1, 0],
                            '3-chain': [2, 1, 0],
                            '2-inter': [0, 1],
                            '3-inter': [0, 1, 2],
                            '3-inter_chain': [0, 2, 1],
                            '3-chain_inter': [1, 2, 0]}

    variable_node_idx = {'1-chain': [0],
                         '2-chain': [0, 2],
                         '3-chain': [0, 2, 4],
                         '2-inter': [0],
                         '3-inter': [0],
                         '3-chain_inter': [0, 2],
                         '3-inter_chain': [0, 3]}

    def __init__(self, graph, enc):
        super(RGCNEncoderDecoder, self).__init__()
        self.enc = enc
        self.graph = graph

        self.emb_dim = graph.feature_dims[next(iter(graph.feature_dims))]
        self.mode_embeddings = {}
        for mode in graph.mode_weights:
            embeddings = torch.empty(self.emb_dim, dtype=torch.float32)
            embeddings = embeddings.uniform_().unsqueeze(dim=0)
            self.mode_embeddings[mode] = nn.Parameter(embeddings)
            self.register_parameter("emb-" + mode, self.mode_embeddings[mode])

        self.rel_ids = {}
        id_rel = 0
        for r1 in graph.relations:
            for r2 in graph.relations[r1]:
                rel = (r1, r2[1], r2[0])
                self.rel_ids[rel] = id_rel
                id_rel += 1

        # TODO: hparam num_bases
        # TODO: hparam num_layers
        self.rgcn = RGCNConv(in_channels=self.emb_dim, out_channels=self.emb_dim,
                             num_relations=len(self.rel_ids), num_bases=10)

    def forward(self, formula, queries, target_nodes):
        q_graphs = self.get_query_graphs(formula, queries)

        out = F.relu(self.rgcn(q_graphs.x, q_graphs.edge_index, q_graphs.edge_type))
        # TODO: hparam readout function
        out = scatter_add(out, q_graphs.batch, dim=0)

        target_embeds = self.enc(target_nodes, formula.target_mode).t()
        scores = F.cosine_similarity(out, target_embeds, dim=1)

        return scores

    def get_query_graphs(self, formula, queries):
        device = next(self.parameters()).device
        batch_size = len(queries)
        n_anchors = len(formula.anchor_modes)

        x = torch.empty(batch_size, n_anchors, self.emb_dim).to(device)
        # First rows of x contain embeddings of all anchor nodes
        for i, anchor_mode in enumerate(formula.anchor_modes):
            anchors = [q.anchor_nodes[i] for q in queries]
            x[:, i] = self.enc(anchors, anchor_mode).t()

        # The rest of the rows contain generic mode embeddings for variable nodes
        var_nodes = formula.get_variable_nodes()
        var_idx = RGCNEncoderDecoder.variable_node_idx[formula.query_type]
        var_embs = torch.stack(
            [self.mode_embeddings[var_nodes[i]] for i in var_idx], dim=1)
        x = torch.cat((x, var_embs.expand(batch_size, -1, -1)), dim=1)

        x = x.reshape(-1, self.emb_dim)

        edge_index = RGCNEncoderDecoder.query_edge_indices[formula.query_type]
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        rels = formula.get_rels()
        rel_idx = RGCNEncoderDecoder.query_edge_label_idx[formula.query_type]
        edge_type = [self.rel_ids[_reverse_relation(rels[i])] for i in rel_idx]
        edge_type = torch.tensor(edge_type, dtype=torch.long)

        edge_data = Data(edge_index=edge_index)
        edge_data.edge_type = edge_type
        graph = Batch.from_data_list([edge_data for i in range(batch_size)])
        graph.x = x

        return graph.to(device)

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
