from collections import defaultdict, namedtuple
from torch_geometric.datasets import Entities
from sacred import Experiment
import os.path as osp
import gzip
import rdflib as rdf
from urllib import parse
import pickle

ex = Experiment()


@ex.config
def config():
    data_dir = './'
    name = 'AIFB'


def get_entity_type(entity_uri: str):
    ent_type = 'literal'
    if entity_uri.startswith('http://'):
        if entity_uri.endswith('instance'):
            uri = parse.urlparse(entity_uri)
            ent_type = uri.scheme + '://' + uri.netloc + osp.dirname(uri.path)
        else:
            ent_type = entity_uri

    return ent_type


@ex.command(unobserved=True)
def extract_graph_data(data_dir, name):
    # Load graph
    data_path = osp.join(data_dir, name)
    dataset = Entities(data_path, name)

    graph_file, *_ = dataset.raw_paths
    rdf_graph = rdf.Graph()
    with gzip.open(graph_file, 'rb') as f:
        rdf_graph.parse(file=f, format='nt')

    entity_ids = defaultdict(lambda: len(entity_ids))
    entity_types = dict()
    relations = set()

    rels = defaultdict(set)
    adj_lists = defaultdict(lambda: defaultdict(set))
    node_maps = defaultdict(set)

    Entity = namedtuple('Entity', ['id', 'ent_type'])

    print('Processing graph...')
    for subj, pred, obj in rdf_graph.triples((None, None, None)):
        rel = str(pred)
        relations.add(rel)

        entities = []
        for entity in (subj, obj):
            # Determine entity ID and type
            id = entity_ids[entity]
            if id not in entity_types:
                entity_types[id] = get_entity_type(str(entity))
            ent_type = entity_types[id]
            entities.append(Entity(id, ent_type))

            node_maps[ent_type].add(id)

        subject_ent, object_ent = entities
        rels[subject_ent.ent_type].add((object_ent.ent_type, rel))
        triple = (subject_ent.ent_type, rel, object_ent.ent_type)
        adj_lists[triple][subject_ent.id].add(object_ent.id)

    # Convert rels to dict of list
    rels = {ent_type: list(rels[ent_type]) for ent_type in rels.keys()}
    # Convert adj_lists to dict of dict
    adj_lists = {rel: dict(adj_lists[rel]) for rel in adj_lists.keys()}
    # Convert node_maps to dict of list
    node_maps = {ent_type: list(node_maps[ent_type]) for ent_type in node_maps}

    # Save to disk
    graph = (rels, adj_lists, node_maps)
    graph_path = osp.join(data_path, 'processed', 'graph_data.pkl')
    file = open(graph_path, 'wb')
    pickle.dump(graph, file, protocol=pickle.HIGHEST_PROTOCOL)

    num_edges = 0
    for triple in adj_lists:
        for ent_id in adj_lists[triple]:
            num_edges += len(adj_lists[triple][ent_id])

    print(f'Saved graph to {graph_path} with statistics:')
    print(f'  {len(entity_types):d} entities')
    print(f'  {len(node_maps):d} entity types')
    print(f'  {num_edges:d} edges')
    print(f'  {len(relations):d} edge types')


if __name__ == '__main__':
    ex.run_commandline()
