from collections import defaultdict, OrderedDict
import pickle as pickle
from multiprocessing import Process
from collections import Counter
from multiprocessing import cpu_count
from torch.utils.data import Dataset, DataLoader
from .graph import _reverse_relation
import numpy as np
from torch_geometric.data import Data, Batch
import torch
import os.path as osp
import random
import os

from netquery.graph import Graph, Query, _reverse_edge


def load_graph(data_dir, embed_dim):
    rels, adj_lists, node_maps = pickle.load(open(data_dir+"/graph_data.pkl", "rb"))
    node_mode_counts = {mode: len(node_maps[mode]) for mode in node_maps}
    num_nodes = sum(node_mode_counts.values())

    new_node_maps = torch.ones(num_nodes + 1, dtype=torch.long).fill_(-1)
    for m, id_list in node_maps.items():
        for i, n in enumerate(id_list):
            assert new_node_maps[n] == -1
            new_node_maps[n] = i

    node_maps = new_node_maps
    feature_dims = {m : embed_dim for m in rels}
    feature_modules = {m : torch.nn.Embedding(node_mode_counts[m] + 1, embed_dim) for m in rels}
    for mode in rels:
        feature_modules[mode].weight.data.normal_(0, 1./embed_dim)

    features = lambda nodes, mode: feature_modules[mode](node_maps[nodes])
    graph = Graph(features, feature_dims, rels, adj_lists)
    return graph, feature_modules, node_maps


def sample_new_clean(data_dir):
    graph_loader = lambda : load_graph(data_dir, 10)[0]
    sample_clean_test(graph_loader, data_dir)


def clean_test(data_dir):
    test_edges = pickle.load(open(osp.join(data_dir, 'test_edges.pkl'), "rb"))
    val_edges = pickle.load(open(osp.join(data_dir, 'val_edges.pkl'), "rb"))
    deleted_edges = set([q[0][1] for q in test_edges] + [_reverse_edge(q[0][1]) for q in test_edges] +
                [q[0][1] for q in val_edges] + [_reverse_edge(q[0][1]) for q in val_edges])

    for i in range(2,4):
        for kind in ["val", "test"]:
            if kind == "val":
                to_keep = 1000
            else:
                to_keep = 10000
            test_queries = load_queries_by_type(data_dir+"/{:s}_queries_{:d}.pkl".format(kind, i), keep_graph=True)
            print("Loaded", i, kind)
            for query_type in test_queries:
                test_queries[query_type] = [q for q in test_queries[query_type] if len(q.get_edges().intersection(deleted_edges)) > 0]
                test_queries[query_type] = test_queries[query_type][:to_keep]

            print(f'Done making {i:d}-{kind} queries:')
            for q_type in test_queries:
                print(f'\t{q_type}: {len(test_queries[q_type])}')

            test_queries = [q.serialize() for queries in list(test_queries.values()) for q in queries]
            pickle.dump(test_queries, open(data_dir+"/{:s}_queries_{:d}.pkl".format(kind, i), "wb"), protocol=pickle.HIGHEST_PROTOCOL)


def make_train_test_edge_data(data_dir):
    print("Loading graph...")
    graph, _, _ = load_graph(data_dir, 10)
    print("Getting all edges...")
    edges = graph.get_all_edges()
    split_point = int(0.1*len(edges))
    val_test_edges_all = edges[:split_point]
    val_test_edges = []
    print("Getting test negative samples...")
    val_test_edge_negsamples = []
    non_neg_edges = set()

    for e in val_test_edges_all:
        neg_samples = graph.get_negative_edge_samples(e, 100)
        # In some special cases there might not be any valid negative samples,
        # for instance for the edge (topic, rdf:type, class): since all
        # topics are of type Topic (a class), there are no entities of type
        # topic *not* related to the class Topic through the type relationship.
        if len(neg_samples) > 0:
            val_test_edges.append(e)
            val_test_edge_negsamples.append(neg_samples)
        elif e[1] not in non_neg_edges:
            non_neg_edges.add(e[1])
            print('Omitting edges of type', e[1])

    val_test_edge_negsamples = [graph.get_negative_edge_samples(e, 100) for e in val_test_edges]
    print("Making and storing test queries.")
    val_test_edge_queries = [Query(("1-chain", val_test_edges[i]), val_test_edge_negsamples[i], None, 100, True) for i in range(len(val_test_edges))]
    val_split_point = int(0.1*len(val_test_edge_queries))
    val_queries = val_test_edge_queries[:val_split_point]
    test_queries = val_test_edge_queries[val_split_point:]
    pickle.dump([q.serialize() for q in val_queries], open(data_dir+"/val_edges.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in test_queries], open(data_dir+"/test_edges.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    print("Removing test edges...")
    graph.remove_edges(val_test_edges)
    print("Making and storing train queries.")
    train_edges = graph.get_all_edges()
    train_queries = [Query(("1-chain", e), None, None, keep_graph=True) for e in train_edges]
    pickle.dump([q.serialize() for q in train_queries], open(data_dir+"/train_edges.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)


def _discard_negatives(file_name, prob=0.9):
    """Discard all but one negative sample for each query, with probability
    prob"""
    queries = pickle.load(open(file_name, "rb"))
    queries = [q if random.random() > prob else (q[0], [random.choice(list(q[1]))], None if q[2] is None else [random.choice(list(q[2]))]) for q in queries]
    pickle.dump(queries, open(file_name, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    print("Finished", file_name)


def discard_negatives(data_dir):
    _discard_negatives(data_dir + "/val_edges.pkl")
    _discard_negatives(data_dir + "/test_edges.pkl")
    for i in range(2, 4):
        _discard_negatives(data_dir + "/val_queries_{:d}.pkl".format(i))
        _discard_negatives(data_dir + "/test_queries_{:d}.pkl".format(i))


def print_query_stats(queries):
    counts = Counter()
    for q in queries:
        q_type = q.formula.query_type
        counts[q_type] += 1

    for q_type in counts:
        print(f'\t{q_type}: {counts[q_type]}')


def make_train_test_query_data(data_dir):
    graph, _, _ = load_graph(data_dir, 10)
    num_samples = 1e6
    num_workers = cpu_count()
    samples_per_worker = num_samples // num_workers
    queries_2, queries_3 = parallel_sample(graph, num_workers, samples_per_worker, data_dir, test=False)

    print('Done making training queries:')
    print_query_stats(queries_2)
    print_query_stats(queries_3)

    pickle.dump([q.serialize() for q in queries_2], open(data_dir + "/train_queries_2.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in queries_3], open(data_dir + "/train_queries_3.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)


def load_queries(data_file, keep_graph=False):
    raw_info = pickle.load(open(data_file, "rb"))
    return [Query.deserialize(info, keep_graph=keep_graph) for info in raw_info]


def load_queries_by_formula(data_file):
    raw_info = pickle.load(open(data_file, "rb"))
    queries = defaultdict(lambda : defaultdict(list))
    for raw_query in raw_info:
        query = Query.deserialize(raw_query)
        queries[query.formula.query_type][query.formula].append(query)
    return queries


def load_queries_by_type(data_file, keep_graph=True):
    raw_info = pickle.load(open(data_file, "rb"))
    queries = defaultdict(list)
    for raw_query in raw_info:
        query = Query.deserialize(raw_query, keep_graph=keep_graph)
        queries[query.formula.query_type].append(query)
    return queries


def load_test_queries_by_formula(data_file):
    raw_info = pickle.load(open(data_file, "rb"))
    queries = {"full_neg" : defaultdict(lambda : defaultdict(list)), 
            "one_neg" : defaultdict(lambda : defaultdict(list))}
    for raw_query in raw_info:
        neg_type = "full_neg" if len(raw_query[1]) > 1 else "one_neg"
        query = Query.deserialize(raw_query)
        queries[neg_type][query.formula.query_type][query.formula].append(query)
    return queries


def sample_clean_test(graph_loader, data_dir):
    train_graph = graph_loader()
    test_graph = graph_loader()
    test_edges = load_queries(data_dir + "/test_edges.pkl")
    val_edges = load_queries(data_dir + "/val_edges.pkl")
    train_graph.remove_edges([(q.target_node, q.formula.rels[0], q.anchor_nodes[0]) for q in test_edges+val_edges])
    test_queries_2 = test_graph.sample_test_queries(train_graph, ["2-chain", "2-inter"], 9000, 1)
    test_queries_2.extend(test_graph.sample_test_queries(train_graph, ["2-chain", "2-inter"], 1000, 1000))
    val_queries_2 = test_graph.sample_test_queries(train_graph, ["2-chain", "2-inter"], 10, 900)
    val_queries_2.extend(test_graph.sample_test_queries(train_graph, ["2-chain", "2-inter"], 100, 1000))
    val_queries_2 = list(set(val_queries_2)-set(test_queries_2))
    print(len(val_queries_2))
    test_queries_3 = test_graph.sample_test_queries(train_graph, ["3-chain", "3-inter", "3-inter_chain", "3-chain_inter"], 9000, 1)
    test_queries_3.extend(test_graph.sample_test_queries(train_graph, ["3-chain", "3-inter", "3-inter_chain", "3-chain_inter"], 1000, 1000))
    val_queries_3 = test_graph.sample_test_queries(train_graph, ["3-chain", "3-inter", "3-inter_chain", "3-chain_inter"], 900, 1)
    val_queries_3.extend(test_graph.sample_test_queries(train_graph, ["3-chain", "3-inter", "3-inter_chain", "3-chain_inter"], 100, 1000))
    val_queries_3 = list(set(val_queries_3)-set(test_queries_3))
    print(len(val_queries_3))
    pickle.dump([q.serialize() for q in test_queries_2], open(data_dir + "/test_queries_2.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in test_queries_3], open(data_dir + "/test_queries_3.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in val_queries_2], open(data_dir + "/val_queries_2.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in val_queries_3], open(data_dir + "/val_queries_3.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

        
def parallel_sample_worker(pid, num_samples, graph, data_dir, is_test, test_edges):
    if not is_test:
        graph.remove_edges([(q.target_node, q.formula.rels[0], q.anchor_nodes[0]) for q in test_edges])
    print("Running worker", pid)
    queries_2 = graph.sample_queries(2, num_samples, 100 if is_test else 1, verbose=True)
    queries_3 = graph.sample_queries(3, num_samples, 100 if is_test else 1, verbose=True)
    print("Done running worker, now saving data", pid)
    pickle.dump([q.serialize() for q in queries_2], open(data_dir + "/queries_2-{:d}.pkl".format(pid), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in queries_3], open(data_dir + "/queries_3-{:d}.pkl".format(pid), "wb"), protocol=pickle.HIGHEST_PROTOCOL)


def parallel_sample(graph, num_workers, samples_per_worker, data_dir, test=False, start_ind=None):
    if not test:
        print("Loading test/val data..")
        test_edges = load_queries(data_dir + "/test_edges.pkl")
        val_edges = load_queries(data_dir + "/val_edges.pkl")
    else:
        test_edges = []
        val_edges = []
    proc_range = list(range(num_workers)) if start_ind is None else list(range(start_ind, num_workers+start_ind))
    procs = [Process(target=parallel_sample_worker, args=[i, samples_per_worker, graph, data_dir, test, val_edges+test_edges]) for i in proc_range]
    for p in procs:
        p.start()
    for p in procs:
        p.join() 
    queries_2 = []
    queries_3 = []
    for i in range(num_workers):
        queries_2_file = osp.join(data_dir, "queries_2-{:d}.pkl".format(i))
        new_queries_2 = load_queries(queries_2_file, keep_graph=True)
        os.remove(queries_2_file)
        queries_2.extend(new_queries_2)

        queries_3_file = osp.join(data_dir, "queries_3-{:d}.pkl".format(i))
        new_queries_3 = load_queries(queries_3_file, keep_graph=True)
        os.remove(queries_3_file)
        queries_3.extend(new_queries_3)

    return queries_2, queries_3


class QueryDataset(Dataset):
    """A dataset for queries of a specific type, e.g. 1-chain.
    The dataset contains queries for formulas of different types, e.g.
    200 queries of type (('protein', '0', 'protein')),
    500 queries of type (('protein', '0', 'function')).
    (note that these queries are of type 1-chain).

    Args:
        queries (dict): maps formulas (graph.Formula) to query instances
            (list of graph.Query?)
    """
    def __init__(self, queries, *args, **kwargs):
        self.queries = queries
        self.num_formula_queries = OrderedDict()
        for form, form_queries in queries.items():
            self.num_formula_queries[form] = len(form_queries)
        self.num_queries = sum(self.num_formula_queries.values())
        self.max_num_queries = max(self.num_formula_queries.values())

    def __len__(self):
        return self.max_num_queries

    def __getitem__(self, index):
        return index

    def collate_fn(self, idx_list):
        # Select a formula type (e.g. ('protein', '0', 'protein'))
        # with probability proportional to the number of queries of that
        # formula type
        counts = np.array(list(self.num_formula_queries.values()))
        probs = counts / float(self.num_queries)
        formula_index = np.argmax(np.random.multinomial(1, probs))
        formula = list(self.num_formula_queries.keys())[formula_index]

        n = self.num_formula_queries[formula]
        # Assume sorted idx_list
        min_idx, max_idx = idx_list[0], idx_list[-1]

        start = min_idx % n
        end = min((max_idx + 1) % n, n)
        end = n if end <= start else end
        queries = self.queries[formula][start:end]

        return formula, queries


class RGCNQueryDataset(QueryDataset):
    """A dataset for queries of a specific type, e.g. 1-chain.
    The dataset contains queries for formulas of different types, e.g.
    200 queries of type (('protein', '0', 'protein')),
    500 queries of type (('protein', '0', 'function')).
    (note that these queries are of type 1-chain).

    Args:
        queries (dict): maps formulas (graph.Formula) to query instances
            (list of graph.Query?)
    """
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

    query_diameters = {'1-chain': 1,
                       '2-chain': 2,
                       '3-chain': 3,
                       '2-inter': 1,
                       '3-inter': 1,
                       '3-inter_chain': 2,
                       '3-chain_inter': 2}

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

    def __init__(self, queries, enc_dec):
        super(RGCNQueryDataset, self).__init__(queries)
        self.mode_ids = enc_dec.mode_ids
        self.rel_ids = enc_dec.rel_ids

    def collate_fn(self, idx_list):
        formula, queries = super(RGCNQueryDataset, self).collate_fn(idx_list)
        graph_data = RGCNQueryDataset.get_query_graph(formula, queries,
                                                      self.rel_ids,
                                                      self.mode_ids)
        anchor_ids, var_ids, graph = graph_data
        return formula, queries, anchor_ids, var_ids, graph

    @staticmethod
    def get_query_graph(formula, queries, rel_ids, mode_ids):
        batch_size = len(queries)
        n_anchors = len(formula.anchor_modes)

        anchor_ids = np.empty([batch_size, n_anchors]).astype(np.int)
        # First rows of x contain embeddings of all anchor nodes
        for i, anchor_mode in enumerate(formula.anchor_modes):
            anchors = [q.anchor_nodes[i] for q in queries]
            anchor_ids[:, i] = anchors

        # The rest of the rows contain generic mode embeddings for variables
        all_nodes = formula.get_nodes()
        var_idx = RGCNQueryDataset.variable_node_idx[formula.query_type]
        var_ids = np.array([mode_ids[all_nodes[i]] for i in var_idx],
                            dtype=np.int)

        edge_index = RGCNQueryDataset.query_edge_indices[formula.query_type]
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        rels = formula.get_rels()
        rel_idx = RGCNQueryDataset.query_edge_label_idx[formula.query_type]
        edge_type = [rel_ids[_reverse_relation(rels[i])] for i in rel_idx]
        edge_type = torch.tensor(edge_type, dtype=torch.long)

        edge_data = Data(edge_index=edge_index)
        edge_data.edge_type = edge_type
        edge_data.num_nodes = n_anchors + len(var_idx)
        graph = Batch.from_data_list([edge_data for i in range(batch_size)])

        return (torch.tensor(anchor_ids, dtype=torch.long),
                torch.tensor(var_ids, dtype=torch.long),
                graph)


def make_data_iterator(data_loader):
    iterator = iter(data_loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(data_loader)
            continue


def get_queries_iterator(queries, batch_size, enc_dec=None):
    dataset = RGCNQueryDataset(queries, enc_dec)
    loader = DataLoader(dataset, batch_size, shuffle=False,
                        collate_fn=dataset.collate_fn)
    return make_data_iterator(loader)


if __name__ == '__main__':
    queries = {('protein','0','protein'): ['a' + str(i) for i in range(10)],
               ('protein', '0', 'function'): ['b' + str(i) for i in range(20)],
               ('function', '0', 'function'): ['c' + str(i) for i in range(30)]}

    iterator = get_queries_iterator(queries, batch_size=4)

    for i in range(50):
        batch = next(iterator)
        print(batch)
