import os
import os.path as osp
import pickle as pkl
import numpy as np
from argparse import ArgumentParser
import mpqe.utils as utils
from mpqe.data_utils import load_queries_by_formula, load_test_queries_by_formula, load_graph
from mpqe.model import RGCNEncoderDecoder, QueryEncoderDecoder
from mpqe.train_helpers import train_ingredient, run_train
from sacred import Experiment
from sacred.observers import MongoObserver
import torch
from torch import optim

parser = ArgumentParser()
parser.add_argument("--model", type=str, default="qrgcn")
parser.add_argument("--embed_dim", type=int, default=128)
parser.add_argument("--data_dir", type=str, default="./rdvp/AIFB/processed/")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--shared_layers", default=False, action='store_true')
parser.add_argument("--adaptive", default=False, action='store_true')
parser.add_argument("--depth", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--max_iter", type=int, default=10000000)
parser.add_argument("--max_burn_in", type=int, default=1000000)
parser.add_argument("--val_every", type=int, default=5000)
parser.add_argument("--tol", type=float, default=0.0001)
parser.add_argument("--cuda", action='store_true')
parser.add_argument("--log_dir", type=str, default="./")
parser.add_argument("--model_dir", type=str, default="./")
parser.add_argument("--decoder", type=str, default="bilinear")
parser.add_argument("--readout", type=str, default="sum")
parser.add_argument("--inter_decoder", type=str, default="mean")
parser.add_argument("--opt", type=str, default="adam")
parser.add_argument("--dropout", type=float, default=0)
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--num_bases", type=int, default=0)
parser.add_argument("--scatter_op", type=str, default='add')
parser.add_argument("--path_weight", type=float, default=0.01)
args = parser.parse_args()

print("Loading graph data..")
graph, feature_modules, node_maps = load_graph(args.data_dir, args.embed_dim)
if args.cuda:
    graph.features = utils.cudify(feature_modules, node_maps)
out_dims = {mode:args.embed_dim for mode in graph.relations}

print("Loading edge data..")
train_queries = load_queries_by_formula(args.data_dir + "/train_edges.pkl")
val_queries = load_test_queries_by_formula(args.data_dir + "/val_edges.pkl")
test_queries = load_test_queries_by_formula(args.data_dir + "/test_edges.pkl")

print("Loading query data..")
for i in range(2, 4):
    train_queries.update(load_queries_by_formula(args.data_dir + "/train_queries_{:d}.pkl".format(i)))
    i_val_queries = load_test_queries_by_formula(args.data_dir + "/val_queries_{:d}.pkl".format(i))
    val_queries["one_neg"].update(i_val_queries["one_neg"])
    val_queries["full_neg"].update(i_val_queries["full_neg"])
    i_test_queries = load_test_queries_by_formula(args.data_dir + "/test_queries_{:d}.pkl".format(i))
    test_queries["one_neg"].update(i_test_queries["one_neg"])
    test_queries["full_neg"].update(i_test_queries["full_neg"])

enc = utils.get_encoder(args.depth, graph, out_dims, feature_modules, args.cuda)

if args.model == 'qrgcn':
    enc_dec = RGCNEncoderDecoder(graph, enc, args.readout, args.scatter_op,
                                 args.dropout, args.weight_decay,
                                 args.num_layers, args.shared_layers,
                                 args.adaptive)
elif args.model == 'gqe':
    dec = utils.get_metapath_decoder(graph,
                                     enc.out_dims if args.depth > 0 else out_dims,
                                     args.decoder)
    inter_dec = utils.get_intersection_decoder(graph, out_dims, args.inter_decoder)
    enc_dec = QueryEncoderDecoder(graph, enc, dec, inter_dec)
else:
    raise ValueError(f'Unknown model {args.model}')

if args.cuda:
    enc_dec.cuda()

if args.opt == "sgd":
    optimizer = optim.SGD([p for p in enc_dec.parameters() if p.requires_grad],
                          lr=args.lr, momentum=0)
elif args.opt == "adam":
    optimizer = optim.Adam([p for p in enc_dec.parameters() if p.requires_grad],
                           lr=args.lr)

fname = "{data:s}{depth:d}-{embed_dim:d}-{lr:f}-{model}-{decoder}-{readout}."
log_file = (args.log_dir + fname + "log").format(
        data=args.data_dir.strip().split("/")[-1],
        depth=args.depth,
        embed_dim=args.embed_dim,
        lr=args.lr,
        decoder=args.decoder,
        model=args.model,
        readout=args.readout)

model_file = "model.pt"

logger = utils.setup_logging(log_file)

ex = Experiment(ingredients=[train_ingredient])
# Set up database logs
uri = os.environ.get('MLAB_URI')
database = os.environ.get('MLAB_DB')
if all([uri, database]):
    ex.observers.append(MongoObserver.create(uri, database))
else:
    print('Running without Sacred observers')


# noinspection PyUnusedLocal
@ex.config
def config():
    model = args.model
    lr = args.lr
    num_layers = args.num_layers
    shared_layers = args.shared_layers
    adaptive = args.adaptive
    readout = args.readout
    dropout = args.dropout
    weight_decay = args.weight_decay
    max_burn_in = args.max_burn_in
    num_basis = args.num_bases
    scatter_op = args.scatter_op
    opt = args.opt
    data_dir = args.data_dir
    path_weight = args.path_weight
    decoder = args.decoder

@ex.main
def main(data_dir, _run):
    exp_id = '-' + str(_run._id) if _run._id is not None else ''
    db_name = database if database is not None else ''
    folder_path = osp.join(args.log_dir, db_name, 'output' + exp_id)
    if not osp.exists(folder_path):
        os.mkdir(folder_path)
    model_path = osp.join(folder_path, model_file)

    run_train(enc_dec, optimizer, train_queries, val_queries, test_queries,
              logger, batch_size=args.batch_size, max_burn_in=args.max_burn_in,
              val_every=args.val_every, max_iter=args.max_iter,
              model_file=model_path, path_weight=args.path_weight)

    # Export embeddings for node classification
    entity_ids_path = osp.join(data_dir, 'entity_ids.pkl')
    if osp.exists(entity_ids_path):
        entity_ids = pkl.load(open(entity_ids_path, 'rb'))
        embeddings = np.zeros((len(entity_ids), 1 + args.embed_dim))

        for i, ent_id in enumerate(entity_ids.values()):
            for mode in enc_dec.graph.full_sets:
                if ent_id in enc_dec.graph.full_sets[mode]:
                    embeddings[i, 0] = ent_id
                    id_tensor = torch.tensor([ent_id])
                    emb = enc_dec.enc(id_tensor, mode).detach().cpu().numpy()
                    embeddings[i, 1:] = emb.reshape(-1)

        file_path = osp.join(folder_path, 'embeddings.npy')
        np.save(file_path, embeddings)
        print(f'Saved embeddings at {file_path}')
    else:
        print('Did not find entity_ids dictionary. Files found:')
        print(os.listdir(data_dir))


ex.run()
