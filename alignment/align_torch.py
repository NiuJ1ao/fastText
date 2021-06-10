#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import argparse
import torch
from torch import nn
from torch.nn import CosineSimilarity
import torch.optim as optim
from utils_torch import *
from tqdm import tqdm
import sys

parser = argparse.ArgumentParser(description='RCSLS for supervised word alignment')

parser.add_argument("--src_emb", type=str, default='', help="Load source embeddings")
parser.add_argument("--tgt_emb", type=str, default='', help="Load target embeddings")
parser.add_argument('--center', action='store_true', help='whether to center embeddings or not')

parser.add_argument('--mode', type=str, default='baseline', help='')

parser.add_argument("--dico_train", type=str, default='', help="train dictionary")
parser.add_argument("--dico_train1", type=str, default='', help="train dictionary")
parser.add_argument("--dico_test", type=str, default='', help="validation dictionary")
parser.add_argument("--dico_test1", type=str, default='', help="validation dictionary")

parser.add_argument("--output", type=str, default='', help="where to save aligned embeddings")
parser.add_argument("--tune_dir", type=str, default='', help="where to save parameters")

parser.add_argument("--knn", type=int, default=10, help="number of nearest neighbors in RCSL/CSLS")
parser.add_argument("--nn_knn", type=int, default=1, help="number of nearest neighbors in NN/CSLS")
parser.add_argument("--maxneg", type=int, default=200000, help="Maximum number of negatives for the Extended RCSLS")
parser.add_argument("--maxsup", type=int, default=-1, help="Maximum number of training examples")
parser.add_argument("--maxload", type=int, default=200000, help="Maximum number of loaded vectors")
parser.add_argument("--seed", type=int, default=42, help="Seed")
parser.add_argument("--cycle_coef", type=float, default=0, help="Coefficient for cycle losses")

parser.add_argument("--model", type=str, default="none", help="Set of constraints: spectral or none")
parser.add_argument("--supervise", type=str, default="single")

parser.add_argument("--lr", type=float, default=1.0, help='learning rate')
parser.add_argument("--niter", type=int, default=10, help='number of iterations')
parser.add_argument("--batchsize", type=int, default=10000, help="batch size for sgd")

params = parser.parse_args()

###### SPECIFIC FUNCTIONS ######
# functions specific to RCSLS
# the rest of the functions are in utils.py

def getknn(sc, x, y, k=10):
    sidx = np.argpartition(sc.cpu().detach().numpy(), -k, axis=1)[:, -k:]
    ytopk = y[sidx.flatten(), :]
    ytopk = ytopk.reshape(sidx.shape[0], sidx.shape[1], y.shape[1])
    f = torch.sum(sc[torch.arange(sc.shape[0])[:, None], sidx])
    df = torch.matmul(ytopk.sum(1).T, x)
    return f / k, df / k, ytopk


def rcsls(X_src, Y_tgt, Z_src, Z_tgt, R, knn=10):
    # print("original impl.")
    X_trans = torch.matmul(X_src, R.T)
    f = 2 * torch.sum(X_trans * Y_tgt)
    df = 2 * torch.matmul(Y_tgt.T, X_src)
    fk0, dfk0, topk0 = getknn(torch.matmul(X_trans, Z_tgt.T), X_src, Z_tgt, knn)
    fk1, dfk1, topk1 = getknn(torch.matmul(torch.matmul(Z_src, R.T), Y_tgt.T).T, Y_tgt, Z_src, knn)
    f = f - fk0 -fk1
    df = df - dfk0 - dfk1.T
    return -f / X_src.shape[0], -df / X_src.shape[0]


def getknn_torch(sc, k=10):
    _, sidx = torch.topk(sc, k, dim=1)
    f = torch.sum(sc[torch.arange(sc.shape[0])[:, None], sidx])
    return (f / k)


def rcsls_nn(X_src, Y_tgt, Z_src, Z_tgt, flinear, blinear, cycle_loss, coef, mode="baseline",
             knn=10, pbar=None):
    # print("pytorch impl.")
    X_trans = flinear(X_src)
    X_trans_loss = 2 * torch.sum(X_trans * Y_tgt)
    X_trans_k0 = getknn_torch(torch.matmul(X_trans, Z_tgt.T))
    X_trans_k1 = getknn_torch(torch.matmul(flinear(Z_src), Y_tgt.T).T, knn)
    X_trans_loss = - X_trans_loss + X_trans_k0 + X_trans_k1
    
    if mode == "baseline":
        return X_trans_loss / X_src.shape[0] 

    Y_trans = blinear(Y_tgt)
    Y_trans_loss = 2 * torch.sum(Y_trans * X_src)
    Y_trans_k0 = getknn_torch(torch.matmul(Y_trans, Z_src.T), knn)
    Y_trans_k1 = getknn_torch(torch.matmul(blinear(Z_tgt), X_src.T).T, knn)
    Y_trans_loss = - Y_trans_loss + Y_trans_k0 + Y_trans_k1
    
    if mode == "dual":
        return (X_trans_loss + Y_trans_loss) / X_src.shape[0]
    
    X_cycle = blinear(X_trans)
    X_cycle_loss = cycle_loss(X_cycle, X_src).sum()
    Y_cycle = flinear(Y_trans)
    Y_cycle_loss = cycle_loss(Y_cycle, Y_tgt).sum()
    
    f = X_trans_loss + Y_trans_loss + coef * (X_cycle_loss + Y_cycle_loss)
    # pbar.write("X_trans_loss = %.4f - Y_trans_loss = %.4f - X_cycle_loss = %.4f - Y_cycle_loss = %.4f" % (X_trans_loss, Y_trans_loss, coef*X_cycle_loss, coef*Y_cycle_loss))
    return f / X_src.shape[0]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
###### MAIN ######
setup_seed(params.seed)

# load word embeddings
words_tgt, x_tgt = load_vectors(params.tgt_emb, maxload=params.maxload, center=params.center)
words_src, x_src = load_vectors(params.src_emb, maxload=params.maxload, center=params.center)

# load validation bilingual lexicon
src2tgt, src2tgt_lexicon_size = load_lexicon(params.dico_test, words_src, words_tgt)
tgt2src, tgt2src_lexicon_size = load_lexicon(params.dico_test1, words_tgt, words_src)

# word --> vector indices
idx_src = idx(words_src)
idx_tgt = idx(words_tgt)

# load train bilingual lexicon
src2tgt_pairs = load_pairs(params.dico_train, idx_src, idx_tgt)
if params.supervise == "single" or params.mode == "baseline":
    pairs = src2tgt_pairs
else:
    tgt2src_pairs = load_pairs(params.dico_train1, idx_tgt, idx_src)
    pairs = np.concatenate([src2tgt_pairs, np.fliplr(tgt2src_pairs)])
    if params.supervise == "union":
        pairs = np.unique(pairs, axis=0)
    elif params.supervise == "intersect":
        uniques, counts = np.unique(pairs, return_counts=True, axis=0)
        pairs = uniques[counts > 1]
    else:
        raise TypeError

# remove overlaps with testset
pairs = [(src, tgt) for src, tgt in pairs 
         if not ((src in src2tgt and tgt in src2tgt[src]) \
             or (tgt in tgt2src and src in tgt2src[tgt]))]

if params.maxsup > 0 and params.maxsup < len(pairs):
    pairs = pairs[:params.maxsup]
print("Loaded train dictionary with %d pairs" % (len(pairs)))

# selecting training vector pairs
X_src, Y_tgt = select_vectors_from_pairs(x_src, x_tgt, pairs)

# adding negatives for RCSLS
Z_src = x_src[:params.maxneg, :]
Z_tgt = x_tgt[:params.maxneg, :]

# initialization:
R = procrustes(X_src, Y_tgt)
bR = procrustes(Y_tgt, X_src)

niter, lr = params.niter, params.lr
nn_knn = params.nn_knn

flinear = nn.Linear(x_src.shape[-1], x_tgt.shape[-1], bias=False)
blinear = nn.Linear(x_tgt.shape[-1], x_src.shape[-1], bias=False)

# optimization
optimizer = optim.SGD([{'params': flinear.parameters()}, 
                       {'params': blinear.parameters()}], lr=lr)
# cycle_loss = torch.nn.MSELoss(reduction='sum')
cycle_loss = torch.nn.CosineSimilarity()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flinear.to(device)
blinear.to(device)
R, bR = R.to(device), bR.to(device)
X_src, Y_tgt = X_src.to(device), Y_tgt.to(device)
Z_src, Z_tgt = Z_src.to(device), Z_tgt.to(device)
x_src, x_tgt = x_src.to(device), x_tgt.to(device)

# initialize ffn
flinear.weight.data.copy_(R)
blinear.weight.data.copy_(bR)
nnacc = compute_nn_accuracy(flinear(x_src), x_tgt, src2tgt, lexicon_size=src2tgt_lexicon_size, knn=nn_knn)
print("[init -- Procrustes] NN: %.4f"%(nnacc))
sys.stdout.flush()
nnacc = compute_nn_accuracy(blinear(x_tgt), x_src, tgt2src, lexicon_size=tgt2src_lexicon_size, knn=nn_knn)
print("[init -- Procrustes] NN: %.4f"%(nnacc))
sys.stdout.flush()

best_valid_metric = 0
tune_dir = params.tune_dir

pbar = tqdm(range(0, niter + 1))
for it in pbar:
    indices = np.random.choice(X_src.shape[0], size=params.batchsize, replace=False)
    # f, df = rcsls(X_src[indices, :], Y_tgt[indices, :], Z_src, Z_tgt, R, params.knn)
    # R -= lr * df
    loss = rcsls_nn(X_src=X_src[indices, :], Y_tgt=Y_tgt[indices, :], Z_src=Z_src, Z_tgt=Z_tgt, 
                    flinear=flinear, blinear=blinear, cycle_loss=cycle_loss, coef=params.cycle_coef, 
                    knn=params.knn, pbar=pbar, mode=params.mode)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    pbar.set_postfix({'loss': loss.item()})
    # pbar.write("[it=%d] loss = %.4f - f = %.4f" % (it, loss, f))
    # pbar.write("[it=%d] loss = %.4f" % (it, loss))

    if (it > 0 and it % 5 == 0) or it == niter:
        fnnacc = compute_nn_accuracy(flinear(x_src), x_tgt, src2tgt, lexicon_size=src2tgt_lexicon_size, knn=nn_knn)
        fcsls = compute_csls_accuracy(flinear(x_src), x_tgt, src2tgt, lexicon_size=src2tgt_lexicon_size, device=device, knn=nn_knn)
        if params.mode == "baseline":
            pbar.write("[it=%d] NN = %.4f - CSLS = %.4f - Coverage = %.4f" % (it, fnnacc, fcsls, len(src2tgt) / src2tgt_lexicon_size))
            if fcsls > best_valid_metric:
                best_valid_metric = fcsls
                torch.save(flinear.state_dict(), tune_dir+'/best_X.t7')
        else:
            bnnacc = compute_nn_accuracy(blinear(x_tgt), x_src, tgt2src, lexicon_size=tgt2src_lexicon_size, knn=nn_knn)
            bcsls = compute_csls_accuracy(blinear(x_tgt), x_src, tgt2src, lexicon_size=tgt2src_lexicon_size, device=device, knn=nn_knn)
            pbar.write("[it=%d] NN = %.4f-%.4f - CSLS = %.4f-%.4f - Coverage = %.4f-%.4f" % (it, fnnacc, bnnacc, fcsls, bcsls, len(src2tgt) / src2tgt_lexicon_size, len(tgt2src) / tgt2src_lexicon_size))
            mean_csls = (fcsls + bcsls) / 2
            if mean_csls > best_valid_metric:
                best_valid_metric = mean_csls
                torch.save(flinear.state_dict(), tune_dir+'/best_X.t7')
                torch.save(blinear.state_dict(), tune_dir+'/best_Y.t7')


flinear.load_state_dict(torch.load(tune_dir+'/best_X.t7'))
x_trans = flinear(x_src)
nnacc = compute_nn_accuracy(x_trans, x_tgt, src2tgt, lexicon_size=src2tgt_lexicon_size, knn=nn_knn)
cslsproc = compute_csls_accuracy(x_trans, x_tgt, src2tgt, lexicon_size=src2tgt_lexicon_size, device=device, knn=nn_knn)
print("[final] NN = %.4f - CSLS = %.4f - Coverage = %.4f" % (nnacc, cslsproc, len(src2tgt) / src2tgt_lexicon_size))
if params.mode != "baseline":
    blinear.load_state_dict(torch.load(tune_dir+'/best_Y.t7'))
    x_trans = blinear(x_tgt)
    nnacc = compute_nn_accuracy(x_trans, x_src, tgt2src, lexicon_size=tgt2src_lexicon_size, knn=nn_knn)
    cslsproc = compute_csls_accuracy(x_trans, x_src, tgt2src, lexicon_size=tgt2src_lexicon_size, device=device, knn=nn_knn)
    print("[final] NN = %.4f - CSLS = %.4f - Coverage = %.4f" % (nnacc, cslsproc, len(tgt2src) / tgt2src_lexicon_size))
