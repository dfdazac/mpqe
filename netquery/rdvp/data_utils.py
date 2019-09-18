from collections import defaultdict, namedtuple
from torch_geometric.datasets import Entities
from sacred import Experiment
import os.path as osp
import gzip
import rdflib as rdf
from urllib import parse
import pickle


class RDVPDataset(Entities):
    def __init__(self, root, name):
        assert name in ['AIFB', 'AM', 'MUTAG', 'BGS']
        self.name = name.lower()
        super(Entities, self).__init__(root, name)

    def process(self):
        pass


DATASET_TYPES = {'AIFB': ['http://swrc.ontoware.org/ontology#Publication',
                          'http://swrc.ontoware.org/ontology#Person',
                          'http://swrc.ontoware.org/ontology#Topic',
                          'http://swrc.ontoware.org/ontology#Organization']}


ex = Experiment()

# TODO: Consider using a regular Python CLI
@ex.config
def config():
    data_dir = './'
    name = 'AIFB'


def extract_entities(graph: rdf.Graph, name):
    query_str = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT ?entity
        WHERE {{
            ?entity rdf:type <{0}>
        }}"""

    node_maps = {}
    types = DATASET_TYPES[name]
    for ent_type in types:
        node_maps[ent_type] = set()
        query = query_str.format(ent_type)
        query_results = graph.query(query)
        for row in query_results:
            node_maps[ent_type].add(row.entity)

    return node_maps


@ex.command(unobserved=True)
def extract_graph_data(data_dir, name):
    # Load graph
    data_path = osp.join(data_dir, name)
    dataset = RDVPDataset(data_path, name)

    graph_file, *_ = dataset.raw_paths
    graph = rdf.Graph()
    with gzip.open(graph_file, 'rb') as f:
        graph.parse(file=f, format='nt')

    node_maps = extract_entities(graph, name)

    entity_ids = defaultdict(lambda: len(entity_ids))
    entity_types = dict()
    relations = set()

    rels = defaultdict(set)
    adj_lists = defaultdict(lambda: defaultdict(set))

    Entity = namedtuple('Entity', ['id', 'ent_type'])

    print('Processing graph...')
    for subj, pred, obj in graph.triples((None, None, None)):
        # print('', subj, '\n', pred, '\n', obj, '\n' + '-'*100)
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
    graph_data = (rels, adj_lists, node_maps)
    graph_path = osp.join(data_path, 'processed', 'graph_data.pkl')
    file = open(graph_path, 'wb')
    pickle.dump(graph_data, file, protocol=pickle.HIGHEST_PROTOCOL)

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
