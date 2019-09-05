import os
from argparse import ArgumentParser
from netquery.utils import *
from netquery.bio.data_utils import load_graph
from netquery.data_utils import load_queries_by_formula, load_test_queries_by_formula
from netquery.model import RGCNEncoderDecoder, QueryEncoderDecoder
from netquery.train_helpers import train_ingredient, run_train
from sacred import Experiment
from sacred.observers import MongoObserver

from torch import optim

parser = ArgumentParser()
parser.add_argument("--model", type=str, default="qrgcn")
parser.add_argument("--embed_dim", type=int, default=128)
parser.add_argument("--data_dir", type=str, default="./bio_data/")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--depth", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--max_iter", type=int, default=10000000)
parser.add_argument("--max_burn_in", type=int, default=1000000)
parser.add_argument("--val_every", type=int, default=5000)
parser.add_argument("--tol", type=float, default=0.0001)
parser.add_argument("--cuda", action='store_true')
parser.add_argument("--log_dir", type=str, default="./")
parser.add_argument("--model_dir", type=str, default="./")
parser.add_argument("--decoder", type=str, default="rgcn")
parser.add_argument("--readout", type=str, default="sum")
parser.add_argument("--inter_decoder", type=str, default="mean")
parser.add_argument("--opt", type=str, default="adam")
parser.add_argument("--dropout", type=float, default=0)
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--num_bases", type=int, default=0)
parser.add_argument("--scatter_op", type=str, default='add')
args = parser.parse_args()

print("Loading graph data..")
graph, feature_modules, node_maps = load_graph(args.data_dir, args.embed_dim)
if args.cuda:
    graph.features = cudify(feature_modules, node_maps)
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

enc = get_encoder(args.depth, graph, out_dims, feature_modules, args.cuda)

if args.model == 'qrgcn':
    enc_dec = RGCNEncoderDecoder(graph, enc, args.readout, args.scatter_op,
                                 args.dropout, args.weight_decay)
elif args.model == 'gqe':
    dec = get_metapath_decoder(graph,
                               enc.out_dims if args.depth > 0 else out_dims,
                               args.decoder)
    inter_dec = get_intersection_decoder(graph, out_dims, args.inter_decoder)
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

fname = "/{data:s}{depth:d}-{embed_dim:d}-{lr:f}-{model}-{readout}."
log_file = args.log_dir + fname + "log".format(
        data=args.data_dir.strip().split("/")[-1],
        depth=args.depth,
        embed_dim=args.embed_dim,
        lr=args.lr,
        decoder=args.decoder,
        model=args.model,
        readout=args.readout)
model_file = args.model_dir + fname + "pt".format(
        data=args.data_dir.strip().split("/")[-1],
        depth=args.depth,
        embed_dim=args.embed_dim,
        lr=args.lr,
        decoder=args.decoder,
        model=args.model,
        readout=args.readout)
logger = setup_logging(log_file)

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
    readout = args.readout
    dropout = args.dropout
    weight_decay = args.weight_decay
    max_burn_in = args.max_burn_in
    num_basis = args.num_bases
    scatter_op = args.scatter_op
    opt = args.opt


@ex.main
def main():
    run_train(enc_dec, optimizer, train_queries, val_queries, test_queries,
              logger, batch_size=args.batch_size, max_burn_in=args.max_burn_in,
              val_every=args.val_every, max_iter=args.max_iter,
              model_file=model_file)

    torch.save(enc_dec.state_dict(), model_file)


ex.run()
