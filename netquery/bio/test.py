from argparse import ArgumentParser
import pickle

from netquery.utils import *
from netquery.bio.data_utils import load_graph
from netquery.data_utils import load_queries_by_formula, load_test_queries_by_formula, RGCNQueryDataset
from netquery.model import RGCNEncoderDecoder
from netquery.train_helpers import run_train

from torch import optim

parser = ArgumentParser()
parser.add_argument("--embed_dim", type=int, default=128)
parser.add_argument("--data_dir", type=str, default="./bio_data/")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--depth", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--max_iter", type=int, default=100000000)
parser.add_argument("--max_burn_in", type=int, default=1000000)
parser.add_argument("--val_every", type=int, default=5000)
parser.add_argument("--tol", type=float, default=0.0001)
parser.add_argument("--cuda", action='store_true')
parser.add_argument("--log_dir", type=str, default="./")
parser.add_argument("--model_dir", type=str, default="./")
parser.add_argument("--decoder", type=str, default="bilinear")
parser.add_argument("--inter_decoder", type=str, default="mean")
parser.add_argument("--opt", type=str, default="adam")
args = parser.parse_args()

print("Loading graph data..")
graph, _, _ = load_graph(args.data_dir)

enc_dec = RGCNEncoderDecoder(graph, args.embed_dim, readout='sum')
if args.cuda:
    enc_dec.cuda()

types = ['1-chain', '2-chain', '3-chain', '2-inter', '3-inter', '3-chain_inter', '3-inter_chain']
for t in types:
    print(f'Testing {t}')
    formula, queries, target_nodes = pickle.load(open(f'{t}.p', 'rb'))
