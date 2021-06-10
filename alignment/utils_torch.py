#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io
import numpy as np
import torch
from tqdm import tqdm
import collections


def load_vectors(fname, maxload=200000, norm=True, center=False, verbose=True):
    if verbose:
        print("Loading vectors from %s" % fname)
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    if maxload > 0:
        n = min(n, maxload)
    x = torch.zeros([n, d])
    words = []
    for i, line in enumerate(fin):
        if i >= n:
            break
        tokens = line.rstrip().split(' ')
        words.append(tokens[0])
        v = np.array(tokens[1:], dtype=float)
        v = torch.tensor(v)
        x[i, :] = v
    if norm:
        x /= torch.linalg.norm(x, dim=1)[:, np.newaxis] + 1e-8
    if center:
        x -= x.mean(dim=0)[np.newaxis, :]
        x /= torch.linalg.norm(x, dim=1)[:, np.newaxis] + 1e-8
    if verbose:
        print("%d word vectors loaded" % (len(words)))
    return words, x


def idx(words):
    w2i = {}
    for i, w in enumerate(words):
        if w not in w2i:
            w2i[w] = i
    return w2i


def save_vectors(fname, x, words):
    n, d = x.shape
    fout = io.open(fname, 'w', encoding='utf-8')
    fout.write(u"%d %d\n" % (n, d))
    for i in tqdm(range(n)):
        fout.write(words[i] + " " + " ".join(map(lambda a: "%.4f" % a, x[i, :])) + "\n")
    fout.close()


def save_matrix(fname, x):
    n, d = x.shape
    fout = io.open(fname, 'w', encoding='utf-8')
    fout.write(u"%d %d\n" % (n, d))
    for i in tqdm(range(n)):
        fout.write(" ".join(map(lambda a: "%.4f" % a, x[i, :])) + "\n")
    fout.close()


def procrustes(X_src, Y_tgt):
    U, s, V = torch.linalg.svd(torch.matmul(Y_tgt.T, X_src))
    return torch.matmul(U, V)


def select_vectors_from_pairs(x_src, y_tgt, pairs):
    n = len(pairs)
    d = x_src.shape[1]
    x = torch.zeros([n, d])
    y = torch.zeros([n, d])
    for k, ij in enumerate(pairs):
        i, j = ij
        x[k, :] = x_src[i, :]
        y[k, :] = y_tgt[j, :]
    return x, y


def load_lexicon(filename, words_src, words_tgt, verbose=True):
    f = io.open(filename, 'r', encoding='utf-8')
    lexicon = collections.defaultdict(set)
    idx_src , idx_tgt = idx(words_src), idx(words_tgt)
    vocab = set()
    for line in f:
        word_src, word_tgt = line.split()
        if word_src in idx_src and word_tgt in idx_tgt:
            lexicon[idx_src[word_src]].add(idx_tgt[word_tgt])
        vocab.add(word_src)
    if verbose:
        coverage = len(lexicon) / float(len(vocab))
        print("Coverage of source vocab: %.4f" % (coverage))
    return lexicon, float(len(vocab))


def load_pairs(filename, idx_src, idx_tgt, verbose=True):
    f = io.open(filename, 'r', encoding='utf-8')
    pairs = []
    tot = 0
    for line in f:
        a, b = line.rstrip().split(' ')
        tot += 1
        if a in idx_src and b in idx_tgt:
            pairs.append((idx_src[a], idx_tgt[b]))
    if verbose:
        coverage = (1.0 * len(pairs)) / tot
        print("Found pairs for training: %d - Total pairs in file: %d - Coverage of pairs: %.4f" % (len(pairs), tot, coverage))
    return pairs


def compute_nn_accuracy(x_src, x_tgt, lexicon, bsz=100, lexicon_size=-1, knn=1):
    if lexicon_size < 0:
        lexicon_size = len(lexicon)
    idx_src = list(lexicon.keys())
    acc = 0.0
    x_src /= torch.linalg.norm(x_src, dim=1)[:, np.newaxis] + 1e-8
    x_tgt /= torch.linalg.norm(x_tgt, dim=1)[:, np.newaxis] + 1e-8
    for i in range(0, len(idx_src), bsz):
        e = min(i + bsz, len(idx_src))
        scores = torch.mm(x_tgt, x_src[idx_src[i:e]].T)
        # pred = scores.argmax(dim=0)
        # pred = pred.cpu().numpy()
        _, pred = scores.topk(knn, dim=0)
        pred = pred.T.cpu().numpy()
        for j in range(i, e):
            for k in pred[j - i]:
                if k in lexicon[idx_src[j]]:
                    acc += 1.0
                    break
    return acc / lexicon_size


def compute_csls_accuracy(x_src, x_tgt, lexicon, lexicon_size=-1, k=10, bsz=1024, device=None, knn=1):
    if lexicon_size < 0:
        lexicon_size = len(lexicon)
    idx_src = list(lexicon.keys())

    x_src /= torch.linalg.norm(x_src, dim=1)[:, np.newaxis] + 1e-8
    x_tgt /= torch.linalg.norm(x_tgt, dim=1)[:, np.newaxis] + 1e-8

    sr = x_src[list(idx_src)]
    sc = torch.matmul(sr, x_tgt.T)
    similarities = 2 * sc
    sc2 = torch.zeros(x_tgt.shape[0], device=device)
    for i in range(0, x_tgt.shape[0], bsz):
        j = min(i + bsz, x_tgt.shape[0])
        sc_batch = torch.matmul(x_tgt[i:j, :], x_src.T)
        dotprod, _ = torch.topk(sc_batch, k, dim=1)
        sc2[i:j] = torch.mean(dotprod, dim=1)
    similarities -= sc2[np.newaxis, :]

    # nn = torch.argmax(similarities, dim=1).tolist()
    _, nn = torch.topk(similarities, knn, dim=1)
    correct = 0.0
    for k in range(0, len(lexicon)):
        for neigh in nn[k]:
            if neigh.item() in lexicon[idx_src[k]]:
                correct += 1.0
                break
    return correct / lexicon_size
