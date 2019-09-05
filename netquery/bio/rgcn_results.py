from argparse import ArgumentParser
from netquery.utils import *
from netquery.bio.data_utils import load_graph
from netquery.model import RGCNEncoderDecoder
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

samples_per_mode = 200

parser = ArgumentParser()
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

state_dict = torch.load(f'../../output/2/0-128-0.010000-rgcn-{args.readout}.pt',
                        map_location='cpu')


print("Loading graph data..")
graph, feature_modules, node_maps = load_graph(args.data_dir, args.embed_dim)
if args.cuda:
    graph.features = cudify(feature_modules, node_maps)
out_dims = {mode:args.embed_dim for mode in graph.relations}

enc = get_encoder(args.depth, graph, out_dims, feature_modules, args.cuda)
enc_dec = RGCNEncoderDecoder(graph, enc, args.readout, args.scatter_op,
                             args.dropout, args.weight_decay)

enc_dec.load_state_dict(state_dict)
num_modes = len(enc_dec.graph.full_lists)

num_samples = num_modes * samples_per_mode
embs = np.zeros([num_samples + num_modes, enc_dec.emb_dim])
labels = np.zeros(num_samples + num_modes)
labels_str = []

start = 0
label = 0
for mode in enc_dec.graph.full_lists:
    embeddings = enc_dec.enc(enc_dec.graph.full_lists[mode][:samples_per_mode],
                             mode).t()
    embeddings = embeddings.detach().numpy()
    embs[start:start + samples_per_mode] = embeddings
    labels[start:start + samples_per_mode] = label
    labels_str.append(mode)

    start = start + samples_per_mode
    label += 1

for mode in enc_dec.mode_ids:
    mode_id = torch.tensor(enc_dec.mode_ids[mode], dtype=torch.long)
    embs[start] = enc_dec.mode_embeddings(mode_id).detach().numpy()
    labels_str.append(mode + '-t')
    start += 1

print('Fitting TSNE...')
tsne = TSNE(random_state=0)
x = tsne.fit_transform(embs)

start = 0
for i in range(5):
    color = f'C{i}'
    plt.scatter(x[start: start + samples_per_mode, 0],
                x[start: start + samples_per_mode, 1],
                color=color,
                label=labels_str[i])

    plt.plot(x[num_samples + i, 0], x[num_samples + i, 1], '^', color=color,
             label=labels_str[i + num_modes], markersize=10,
             markeredgewidth=1.5, markeredgecolor='k')

    start = start + samples_per_mode

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()

