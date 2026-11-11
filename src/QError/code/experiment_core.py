import contextlib
import io
import math
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from mscn.data import make_dataset
from mscn.model import SetConv
from mscn.util import encode_data, encode_samples, normalize_labels, unnormalize_labels
import train


def encode(records,cp,target,labels=None,max_predicates=None):
    td,cd,od,jd=cp['dicts']
    bm=[np.vstack([np.unpackbits(np.frombuffer(bytes.fromhex(b),dtype=np.uint8)) for b in r['bitmap_hex']]) for r in records]
    ps,js=encode_data([r['predicates'] or [['']] for r in records],[r['joins'] or [''] for r in records],cp['column_min_max_vals'],cd,od,jd)
    return make_dataset(encode_samples([r['tables'] for r in records],bm,td),ps,js,np.zeros(len(records)) if labels is None else labels,
        max(len(target['joins']),1),len(target['predicates']) if max_predicates is None else max_predicates)


def make_model(cp):
    d=cp['dicts'];return SetConv(len(d[0])+1000,len(d[1])+len(d[2])+1,len(d[3]),256)


def normalized_labels(rows):
    with contextlib.redirect_stdout(io.StringIO()):
        return normalize_labels([max(int(r['true_cardinality']),1) for r in rows])


def vectorized_qerror(preds,targets,min_val,max_val):
    predictions=train.unnormalize_torch(preds,min_val,max_val).reshape(-1)
    actual=train.unnormalize_torch(targets,min_val,max_val).reshape(-1)
    return torch.where(predictions>actual,predictions/actual,actual/predictions).mean()


def infer_features(model, checkpoint, tables, joins, predicates, samples):
    table2vec, column2vec, op2vec, join2vec = checkpoint["dicts"]
    sample_features = encode_samples(tables, samples, table2vec)
    predicate_features, join_features = encode_data(predicates, joins, checkpoint["column_min_max_vals"],
                                                    column2vec, op2vec, join2vec)
    dataset = make_dataset(sample_features, predicate_features, join_features, np.zeros(len(tables)),
                           max(map(len, join_features)), max(map(len, predicate_features)))
    outputs = []
    started = time.monotonic()
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=1024):
            sample, predicate, join, _, sample_mask, predicate_mask, join_mask = batch
            outputs.extend(model(sample, predicate, join, sample_mask, predicate_mask, join_mask).flatten().tolist())
    if not all(math.isfinite(value) for value in outputs):
        raise ValueError("Model produced non-finite values")
    counts = unnormalize_labels(outputs, checkpoint["min_val"], checkpoint["max_val"])
    if any(value < 1 for value in counts):
        raise ValueError("Expected positive model estimates")
    return [int(value) for value in counts], time.monotonic()-started
