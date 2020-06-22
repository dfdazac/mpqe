from functools import reduce
from collections import defaultdict
from sacred import Experiment
from torch_geometric.datasets import Entities
import pandas as pd
import os
import os.path as osp
import rdflib as rdf
import pickle as pkl

import mpqe.data_utils as utils

ex = Experiment()


# noinspection PyUnusedLocal
@ex.config
def config():
    name = 'Bio2'


class RDVPDataset(Entities):
    """A wrapper around the Entities dataset that bypasses the process
    method, used to download the datasets."""
    def __init__(self, root, name):
        assert name in ['AIFB', 'AM', 'MUTAG', 'BGS']
        self.name = name.lower()
        super(Entities, self).__init__(root, name)

    def process(self):
        pass


def extract_entity_types(types_folder):
    """Build a dictionary mapping entity types to entities, by reading CSV
    files from a directory. Each CSV file corresponds to a type."""
    node_maps = {}

    for file in os.listdir(types_folder):
        if file.endswith('.csv'):
            ent_type = file.split('.')[0]
            file_path = osp.join(types_folder, file)
            node_maps[ent_type] = set(open(file_path).read().splitlines())

    # Assert that all sets of typed entities are disjoint
    typed_entities = sum(map(len, node_maps.values()))
    unique_entities = len(reduce(lambda x, y: x.union(y), node_maps.values()))
    assert typed_entities == unique_entities

    return node_maps


def get_triples_file(directory):
    """Look for an .nt file in the given directory. Note: only the first file
    found (if any) is returned."""
    filename = None
    for file in os.listdir(directory):
        if file.endswith('.nt'):
            filename = file
            break

    if filename is None:
        raise FileNotFoundError(f'No .nt files found in {directory}')

    return filename


def get_entity_type(node_maps, entity):
    """Given an entity ID (int), retrieve its type."""
    ent_type = None
    for t in node_maps:
        if entity in node_maps[t]:
            ent_type = t
            break

    return ent_type


@ex.command(unobserved=True)
def download_graph(name):
    RDVPDataset(name, name)


@ex.command(unobserved=True)
def preprocess_graph(name):
    """Read RDF triples and a list of csv files mapping entities to their
    types, and store a subgraph containing only entities of known types.
    """
    raw_dir = osp.join(name, 'raw')
    out_dir = osp.join(name, 'processed')

    # Load graph
    nt_file = get_triples_file(raw_dir)
    graph = rdf.Graph()
    print(f'Loading graph from {osp.join(raw_dir, nt_file)}...')
    graph.parse(osp.join(raw_dir, nt_file), format='nt')

    # Extract entities of predefined types, listed in .csv files
    type_entities = extract_entity_types(out_dir)

    entity_ids_path = osp.join(out_dir, 'entity_ids.pkl')
    if osp.exists(entity_ids_path):
        print(f'Loaded entity_ids from {entity_ids_path}...')
        entity_ids = pkl.load(open(entity_ids_path, 'rb'))
    else:
        entity_ids = defaultdict(lambda: len(entity_ids))

    relations = set()

    rels = defaultdict(set)
    adj_lists = dict()
    node_maps = defaultdict(set)

    print('Extracting triples...')
    for subj, pred, obj in graph.triples((None, None, None)):
        subj = str(subj)
        obj = str(obj)
        subj_type = get_entity_type(type_entities, subj)
        obj_type = get_entity_type(type_entities, obj)

        # Add only triples involving extracted entities (i.e. with known type)
        if all([subj_type, obj_type]):
            # Remove period from relations as they cannot occur in parameter
            # names in PyTorch
            rel = str(pred).replace('.', '')
            relations.add(rel)

            try:
                subj_id = entity_ids[subj]
                obj_id = entity_ids[obj]
            except KeyError:
                raise ValueError(f'Could not map entity to an ID.\n'
                                 f'Consider deleting {entity_ids_path}'
                                 f'to create map from scratch.')

            node_maps[subj_type].add(subj_id)
            node_maps[obj_type].add(obj_id)

            # Add edge and its inverse
            rels[subj_type].add((obj_type, rel))
            rels[obj_type].add((subj_type, rel))

            triple = (subj_type, rel, obj_type)
            triple_inv = (obj_type, rel, subj_type)

            if triple not in adj_lists:
                adj_lists[triple] = defaultdict(set)
            if triple_inv not in adj_lists:
                adj_lists[triple_inv] = defaultdict(set)

            adj_lists[triple][subj_id].add(obj_id)
            adj_lists[triple_inv][obj_id].add(subj_id)

    # Convert rels to dict of list
    rels = {ent_type: list(rels[ent_type]) for ent_type in rels.keys()}
    # Convert node_maps to dict of list
    node_maps = {ent_type: list(node_maps[ent_type]) for ent_type in node_maps}
    # Lock dictionary with entity IDs
    entity_ids = dict(entity_ids)
    pkl.dump(entity_ids, open(osp.join(out_dir, 'entity_ids.pkl'), 'wb'),
             protocol=pkl.HIGHEST_PROTOCOL)

    # Save to disk
    graph_data = (rels, adj_lists, node_maps)
    graph_path = osp.join(out_dir, 'graph_data.pkl')
    file = open(graph_path, 'wb')
    pkl.dump(graph_data, file, protocol=pkl.HIGHEST_PROTOCOL)

    num_edges = 0
    for triple in adj_lists:
        for ent_id in adj_lists[triple]:
            num_edges += len(adj_lists[triple][ent_id])

    num_entities = sum(map(len, node_maps.values()))

    print(f'Saved graph data to {graph_path} with statistics:')
    print(f'  {num_entities:d} entities')
    print(f'  {len(node_maps):d} entity types')
    print(f'  {num_edges:d} edges')
    print(f'  {len(relations):d} edge types')

    # Save label data for classification
    if name == 'AM':
        label_header = 'label_cateogory'
        nodes_header = 'proxy'
    elif name == 'AIFB':
        label_header = 'label_affiliation'
        nodes_header = 'person'
    elif name == 'MUTAG':
        label_header = 'label_mutagenic'
        nodes_header = 'bond'
    elif name == 'BGS':
        label_header = 'label_lithogenesis'
        nodes_header = 'rock'

    labels_df = pd.read_csv(osp.join(raw_dir, 'completeDataset.tsv'), sep='\t')
    labels_set = set(labels_df[label_header].values.tolist())
    labels_dict = {lab: i for i, lab in enumerate(list(labels_set))}

    train_labels_df = pd.read_csv(osp.join(raw_dir, 'trainingSet.tsv'), sep='\t')
    train_labels = {}
    for node, label in zip(train_labels_df[nodes_header].values,
                           train_labels_df[label_header].values):
        assert node in entity_ids
        train_labels[entity_ids[node]] = labels_dict[label]

    pkl.dump(train_labels, open(osp.join(out_dir, 'train_labels.pkl'), 'wb'),
             protocol=pkl.HIGHEST_PROTOCOL)

    test_labels_df = pd.read_csv(osp.join(raw_dir, 'testSet.tsv'), sep='\t')
    test_labels = {}
    for node, label in zip(test_labels_df[nodes_header].values,
                           test_labels_df[label_header].values):
        assert node in entity_ids
        test_labels[entity_ids[node]] = labels_dict[label]

    pkl.dump(test_labels, open(osp.join(out_dir, 'test_labels.pkl'), 'wb'),
             protocol=pkl.HIGHEST_PROTOCOL)


@ex.command(unobserved=True)
def make_queries(name):
    data_dir = osp.join(name, 'processed')
    utils.make_train_test_edge_data(data_dir)
    utils.make_train_queries(data_dir)
    utils.make_test_queries(data_dir)
    utils.clean_test_queries(data_dir)
    utils.discard_negatives(data_dir)


if __name__ == '__main__':
    ex.run_commandline()
