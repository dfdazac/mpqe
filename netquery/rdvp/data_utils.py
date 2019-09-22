from functools import reduce
from collections import defaultdict
from sacred import Experiment
import os
import os.path as osp
import rdflib as rdf
import pickle

import netquery.bio.data_utils as utils

ex = Experiment()

# TODO: Consider using a regular Python CLI
@ex.config
def config():
    data_dir = './'
    name = 'aifbc'


def extract_entity_types(types_folder):
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


def get_entity_type(node_maps, entity):
    ent_type = None
    for t in node_maps:
        if entity in node_maps[t]:
            ent_type = t
            break

    return ent_type


def defaultdict_set_factory():
    return defaultdict(set)


@ex.command(unobserved=True)
def extract_graph_data(data_dir, name):
    # Load graph
    data_path = osp.join(data_dir, name)
    graph = rdf.Graph()
    graph.parse(osp.join(data_path, f'{name}.nt'), format='nt')

    # Extract entities of predefined types, listed in .csv files
    type_entities = extract_entity_types(data_path)

    entity_ids = defaultdict(lambda: len(entity_ids))
    relations = set()

    rels = defaultdict(set)
    adj_lists = defaultdict(defaultdict_set_factory)
    node_maps = defaultdict(set)

    print('Extracting triples...')
    for subj, pred, obj in graph.triples((None, None, None)):
        subj_type = get_entity_type(type_entities, str(subj))
        obj_type = get_entity_type(type_entities, str(obj))

        # Add only triples involving extracted entities (i.e. with known type)
        if all([subj_type, obj_type]):
            rel = str(pred)
            relations.add(rel)
            subj_id = entity_ids[subj]
            obj_id = entity_ids[obj]

            node_maps[subj_type].add(subj_id)
            node_maps[obj_type].add(obj_id)

            # Add edge and its inverse
            rels[subj_type].add((obj_type, rel))
            rels[obj_type].add((subj_type, rel))
            adj_lists[(subj_type, rel, obj_type)][subj_id].add(obj_id)
            adj_lists[(obj_type, rel, subj_type)][obj_id].add(subj_id)

    # Convert rels to dict of list
    rels = {ent_type: list(rels[ent_type]) for ent_type in rels.keys()}
    # Convert node_maps to dict of list
    node_maps = {ent_type: list(node_maps[ent_type]) for ent_type in node_maps}

    # Save to disk
    graph_data = (rels, adj_lists, node_maps)
    graph_path = osp.join(data_path, 'graph_data.pkl')
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
def make_train_test_edges(data_dir, name):
    data_path = osp.join(data_dir, name)
    utils.make_train_test_edge_data(data_path)


@ex.command(unobserved=True)
def make_train_queries(data_dir, name):
    data_path = osp.join(data_dir, name)
    utils.make_train_test_query_data(data_path)


@ex.command(unobserved=True)
def make_test_queries(data_dir, name):
    data_path = osp.join(data_dir, name)
    utils.sample_new_clean(data_path)


@ex.command(unobserved=True)
def clean_test_queries(data_dir, name):
    data_path = osp.join(data_dir, name)
    utils.clean_test(data_path)


if __name__ == '__main__':
    ex.run_commandline()
