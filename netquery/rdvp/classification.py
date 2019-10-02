import os.path as osp
import numpy as np
import pickle as pkl
from sacred import Experiment
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

ex = Experiment()


# noinspection PyUnusedLocal
@ex.config
def config():
    data_dir = 'AIFB'
    experiment_id = '33'
    random_splits = False
    num_runs = 1


def make_train_test_data(embs, train_labels, test_labels):
    emb_dim = embs.shape[1] - 1

    # Create training data
    x_train = np.zeros((len(train_labels), emb_dim))
    x_test = np.zeros((len(test_labels), emb_dim))
    y_train = np.zeros(len(train_labels), dtype=np.int)
    y_test = np.zeros(len(test_labels), dtype=np.int)

    train_count = 0
    test_count = 0
    for row in embs:
        entity_id = int(row[0])
        embedding = row[1:]
        if entity_id in train_labels:
            x_train[train_count] = embedding
            y_train[train_count] = train_labels[entity_id]
            train_count += 1
        elif entity_id in test_labels:
            x_test[test_count] = embedding
            y_test[test_count] = test_labels[entity_id]
            test_count += 1
        else:
            continue

    return x_train, y_train, x_test, y_test


@ex.automain
def run_classifier(data_dir, experiment_id, random_splits, num_runs, _log):
    data_dir = osp.join(data_dir, 'processed')
    embs_dir = osp.join(data_dir,
                        f'artifacts-{experiment_id}',
                        'embeddings.npy')

    embs = np.load(embs_dir)
    train_labels = pkl.load(open(osp.join(data_dir, 'train_labels.pkl'), 'rb'))
    test_labels = pkl.load(open(osp.join(data_dir, 'test_labels.pkl'), 'rb'))

    x_train, y_train, x_test, y_test = make_train_test_data(embs,
                                                            train_labels,
                                                            test_labels)

    scores = np.empty(num_runs)
    for i in tqdm(range(num_runs)):
        if random_splits:
            x = np.concatenate((x_train, x_test), axis=0)
            y = np.concatenate((y_train, y_test), axis=0)
            test_size = len(test_labels)/(len(test_labels) + len(train_labels))
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
            train_idx, test_idx = next(splitter.split(x, y))

            x_train, y_train = x[train_idx], y[train_idx]
            x_test, y_test = x[test_idx], y[test_idx]

        # Train classifier
        model = LogisticRegressionCV(cv=5, multi_class='auto', max_iter=300)
        model.fit(x_train, y_train)

        # Evaluate accuracy
        y_pred = model.predict(x_test)
        scores[i] = accuracy_score(y_test, y_pred) * 100

    _log.info(f'Accuracy: {scores.mean():.2f} ± {scores.std():.2f}')
