from functools import reduce
from collections import defaultdict
from sacred import Experiment
import os
import os.path as osp
import rdflib as rdf
import pickle

import netquery.bio.data_utils as utils

ex = Experiment()


@ex.config
def config():
    data_dir = '../bio/aifb/'


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
def preprocess_graph(data_dir):
    """Read RDF triples and a list of csv files mapping entities to their
    types, and store a subgraph containing only entities of known types.
    """

    # Load graph
    nt_file = get_triples_file(data_dir)
    graph = rdf.Graph()
    print(f'Loading graph from {nt_file}...')
    graph.parse(osp.join(data_dir, nt_file), format='nt')

    # Extract entities of predefined types, listed in .csv files
    type_entities = extract_entity_types(data_dir)

    entity_ids = defaultdict(lambda: len(entity_ids))
    relations = set()

    rels = defaultdict(set)
    adj_lists = dict()
    node_maps = defaultdict(set)

    print('Extracting triples...')
    for subj, pred, obj in graph.triples((None, None, None)):
        subj_type = get_entity_type(type_entities, str(subj))
        obj_type = get_entity_type(type_entities, str(obj))

        # Add only triples involving extracted entities (i.e. with known type)
        if all([subj_type, obj_type]):
            # Remove period from relations as they cannot occur in parameter
            # names in PyTorch
            rel = str(pred).replace('.', '')
            relations.add(rel)
            subj_id = entity_ids[subj]
            obj_id = entity_ids[obj]

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

    # Save to disk
    graph_data = (rels, adj_lists, node_maps)
    graph_path = osp.join(data_dir, 'graph_data.pkl')
    file = open(graph_path, 'wb')
    pickle.dump(graph_data, file, protocol=pickle.HIGHEST_PROTOCOL)

    num_edges = 0
    for triple in adj_lists:
        for ent_id in adj_lists[triple]:
            num_edges += len(adj_lists[triple][ent_id])

    num_entities = sum(map(len, node_maps.values()))

    print(f'Saved graph to {graph_path} with statistics:')
    print(f'  {num_entities:d} entities')
    print(f'  {len(node_maps):d} entity types')
    print(f'  {num_edges:d} edges')
    print(f'  {len(relations):d} edge types')


@ex.command(unobserved=True)
def make_queries(data_dir):
    utils.make_train_test_edge_data(data_dir)
    utils.make_train_test_query_data(data_dir)
    utils.sample_new_clean(data_dir)
    utils.clean_test(data_dir)
    utils.discard_negatives(data_dir)


if __name__ == '__main__':
    ex.run_commandline()
