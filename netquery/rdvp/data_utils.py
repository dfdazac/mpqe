from functools import reduce
from collections import defaultdict, namedtuple
from torch_geometric.datasets import Entities
from sacred import Experiment
import os
import os.path as osp
import gzip
import rdflib as rdf
import pickle


class RDVPDataset(Entities):
    def __init__(self, root, name):
        assert name in ['AIFB', 'AM', 'MUTAG', 'BGS']
        self.name = name.lower()
        super(Entities, self).__init__(root, name)

    def process(self):
        pass


ex = Experiment()

# TODO: Consider using a regular Python CLI
@ex.config
def config():
    data_dir = './'
    name = 'AIFB'


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


@ex.command(unobserved=True)
def extract_graph_data(data_dir, name):
    # Load graph
    data_path = osp.join(data_dir, name)
    dataset = RDVPDataset(data_path, name)
    graph_file, *_ = dataset.raw_paths
    graph = rdf.Graph()
    with gzip.open(graph_file, 'rb') as f:
        graph.parse(file=f, format='nt')

    # Extract entities of predefined types, listed in .csv files
    type_entities = extract_entity_types(osp.join(data_path, 'processed'))

    entity_ids = defaultdict(lambda: len(entity_ids))
    relations = set()

    rels = defaultdict(set)
    adj_lists = defaultdict(lambda: defaultdict(set))
    node_maps = defaultdict(set)

    print('Extracting triples...')
    for subj, pred, obj in graph.triples((None, None, None)):
        subj_type = get_entity_type(type_entities, str(subj))
        obj_type = get_entity_type(type_entities, str(obj))

        # Add only triples involving extracted entities (i.e. with known type)
        if all([subj_type, obj_type]):
            rel = str(pred)

            if name == 'AIFB':
                # Discard redundant relations
                if rel == 'http://swrc.ontoware.org/ontology#publication':
                    rel = 'http://swrc.ontoware.org/ontology#author'
                    subj, obj = obj, subj
                    subj_type, obj_type = obj_type, subj_type

            relations.add(rel)
            subj_id = entity_ids[subj]
            obj_id = entity_ids[obj]

            node_maps[subj_type].add(subj_id)
            node_maps[obj_type].add(obj_id)

            rels[subj_type].add((obj_type, rel))
            triple = (subj_type, rel, obj_type)
            adj_lists[triple][subj_id].add(obj_id)

    # Convert rels to dict of list
    rels = {ent_type: list(rels[ent_type]) for ent_type in rels.keys()}
    # Convert adj_lists to dict of dict
    adj_lists = {rel: dict(adj_lists[rel]) for rel in adj_lists.keys()}
    # Convert node_maps to dict of list
    node_maps = {ent_type: list(node_maps[ent_type]) for ent_type in node_maps}

    # Save to disk
    graph_data = (rels, adj_lists, node_maps)
    graph_path = osp.join(data_path, 'processed', 'graph_data.pkl')
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


if __name__ == '__main__':
    ex.run_commandline()
