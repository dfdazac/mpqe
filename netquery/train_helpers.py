import numpy as np
from .utils import eval_auc_queries, eval_perc_queries
from .data_utils import get_queries_iterator
from .model import RGCNEncoderDecoder, QueryEncoderDecoder
import torch
from sacred import Ingredient

train_ingredient = Ingredient('train')


def check_conv(vals, window=2, tol=1e-6):
    if len(vals) < 2 * window:
        return False
    conv = np.mean(vals[-window:]) - np.mean(vals[-2*window:-window]) 
    return conv < tol


def update_loss(loss, losses, ema_loss, ema_alpha=0.01):
    losses.append(loss)
    if ema_loss is None:
        ema_loss = loss
    else:
        ema_loss = (1-ema_alpha)*ema_loss + ema_alpha*loss
    return losses, ema_loss


@train_ingredient.capture
@torch.no_grad()
def run_eval(model, queries, iteration, logger, batch_size=128, by_type=False,
             _run=None):
    model.eval()
    vals = {}
    def _print_by_rel(rel_aucs, logger):
        for rels, auc in rel_aucs.items():
            logger.info(str(rels) + "\t" + str(auc))
    for query_type in queries["one_neg"]:
        auc, rel_aucs = eval_auc_queries(queries["one_neg"][query_type], model)
        perc = eval_perc_queries(queries["full_neg"][query_type], model, batch_size)
        vals[query_type] = auc
        logger.info("{:s} val AUC: {:f} val perc {:f}; iteration: {:d}".format(query_type, auc, perc, iteration))
        _run.log_scalar(f'{query_type}_val_auc', auc, iteration)
        _run.log_scalar(f'{query_type}_val_perc', perc, iteration)
        if by_type:
            _print_by_rel(rel_aucs, logger)
        if "inter" in query_type:
            auc, rel_aucs = eval_auc_queries(queries["one_neg"][query_type], model, hard_negatives=True)
            perc = eval_perc_queries(queries["full_neg"][query_type], model, hard_negatives=True)
            logger.info("Hard-{:s} val AUC: {:f} val perc {:f}; iteration: {:d}".format(query_type, auc, perc, iteration))
            _run.log_scalar(f'hard_{query_type}_val_auc', auc, iteration)
            _run.log_scalar(f'hard_{query_type}_val_perc', perc, iteration)
            if by_type:
                _print_by_rel(rel_aucs, logger)
            vals[query_type + "hard"] = auc
    return vals


@train_ingredient.capture
def run_train(model, optimizer, train_queries, val_queries, test_queries,
              logger, max_burn_in=100000, batch_size=512, log_every=500,
              val_every=1000, tol=1e-6, max_iter=int(10e7), inter_weight=0.005,
              path_weight=0.01, model_file=None, _run=None):
    edge_conv = False
    ema_loss = None
    vals = []
    losses = []
    conv_test = None

    if isinstance(model, RGCNEncoderDecoder):
        train_iterators = {}
        for query_type in train_queries:
            queries = train_queries[query_type]
            train_iterators[query_type] = get_queries_iterator(queries,
                                                               batch_size,
                                                               model)

    for i in range(max_iter):
        
        optimizer.zero_grad()

        if isinstance(model, RGCNEncoderDecoder):
            loss = run_batch_v2(train_iterators['1-chain'], model)
        else:
            loss = run_batch(train_queries["1-chain"], model, i, batch_size)

        if not edge_conv and (check_conv(vals) or len(losses) >= max_burn_in):
            logger.info("Edge converged at iteration {:d}".format(i-1))
            logger.info("Testing at edge conv...")
            conv_test = run_eval(model, test_queries, i, logger)
            conv_test = np.mean(list(conv_test.values()))
            edge_conv = True
            losses = []
            ema_loss = None
            vals = []
            if model_file is not None:
                torch.save(model.state_dict(), model_file+"-edge_conv")
        
        if edge_conv:
            for query_type in train_queries:
                if query_type == "1-chain" and max_burn_in > 0:
                    continue
                if "inter" in query_type:
                    if isinstance(model, RGCNEncoderDecoder):
                        loss += inter_weight * run_batch_v2(train_iterators[query_type], model)
                        loss += inter_weight * run_batch_v2(train_iterators[query_type], model, hard_negatives=True)
                    else:
                        loss += inter_weight * run_batch(train_queries[query_type], model, i, batch_size)
                        loss += inter_weight * run_batch(train_queries[query_type], model, i, batch_size, hard_negatives=True)
                else:
                    if isinstance(model, RGCNEncoderDecoder):
                        loss += path_weight * run_batch_v2(train_iterators[query_type], model)
                    else:
                        loss += path_weight * run_batch(train_queries[query_type], model, i, batch_size)

            if check_conv(vals):
                    logger.info("Fully converged at iteration {:d}".format(i))
                    break

        losses, ema_loss = update_loss(loss.item(), losses, ema_loss)
        loss.backward()
        optimizer.step()
            
        if i % log_every == 0:
            logger.info("Iter: {:d}; ema_loss: {:f}".format(i, ema_loss))
            _run.log_scalar('ema_loss', ema_loss, i)
            
        if i >= val_every and i % val_every == 0:
            v = run_eval(model, val_queries, i, logger)
            if edge_conv:
                vals.append(np.mean(list(v.values())))
            else:
                vals.append(v["1-chain"])
    
    v = run_eval(model, test_queries, i, logger)
    test_avg_auc = np.mean(list(v.values()))
    logger.info("Test macro-averaged val: {:f}".format(test_avg_auc))
    _run.log_scalar('test_auc', test_avg_auc)
    logger.info("Improvement from edge conv: {:f}".format((np.mean(list(v.values()))-conv_test)/conv_test))

    torch.save(model.state_dict(), model_file)


def run_batch(train_queries, enc_dec, iter_count, batch_size, hard_negatives=False):
    num_queries = [float(len(queries)) for queries in list(train_queries.values())]
    denom = float(sum(num_queries))
    formula_index = np.argmax(np.random.multinomial(1, 
            np.array(num_queries)/denom))
    formula = list(train_queries.keys())[formula_index]
    n = len(train_queries[formula])
    start = (iter_count * batch_size) % n
    end = min(((iter_count+1) * batch_size) % n, n)
    end = n if end <= start else end
    queries = train_queries[formula][start:end]
    loss = enc_dec.margin_loss(formula, queries, hard_negatives=hard_negatives)
    return loss


def run_batch_v2(queries_iterator, enc_dec, hard_negatives=False):
    enc_dec.train()

    batch = next(queries_iterator)
    loss = enc_dec.margin_loss(*batch, hard_negatives=hard_negatives)
    return loss
