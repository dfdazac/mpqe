import pickle as pickle
import torch
import random
import os.path as osp
from multiprocessing import cpu_count
from collections import Counter

from netquery.data_utils import parallel_sample, load_queries_by_type, sample_clean_test
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
    val_test_edges = edges[:split_point]
    print("Getting negative samples...")
    val_test_edge_negsamples = [graph.get_negative_edge_samples(e, 100) for e in val_test_edges]
    print("Making and storing test queries.")
    val_test_edge_queries = [Query(("1-chain", val_test_edges[i]), val_test_edge_negsamples[i], None, 100, True) for i in range(split_point)]
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


if __name__ == "__main__":
    #make_train_test_query_data("/dfs/scratch0/nqe-bio/")
    #make_train_test_edge_data("/dfs/scratch0/nqe-bio/")
    sample_new_clean("/dfs/scratch0/nqe-bio/")
    #clean_test()
