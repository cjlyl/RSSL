import numpy as np
import numpy
import collections
from handle_embeddings import *
try:
    import cupy
except ImportError:
    cupy = None


def supports_cupy():
    return cupy is not None


def get_cupy():
    return cupy


def get_array_module(x):
    if cupy is not None:
        return cupy.get_array_module(x)
    else:
        return numpy


def asnumpy(x):
    if cupy is not None:
        return cupy.asnumpy(x)
    else:
        return numpy.asarray(x)



def topk_mean(m, k, inplace=False):  # TODO Assuming that axis is 1
    xp = get_array_module(m)
    n = m.shape[0]
    ans = xp.zeros(n, dtype=m.dtype)
    if k <= 0:
        return ans
    if not inplace:
        m = xp.array(m)
    ind0 = xp.arange(n)
    ind1 = xp.empty(n, dtype=int)
    minimum = m.min()
    for i in range(k):
        m.argmax(axis=1, out=ind1)
        ans += m[ind0, ind1]
        m[ind0, ind1] = minimum
    return ans / k


def CSLS(src_embs,tgt_embs, bsz=1024):
    BATCH_SIZE = 500
    neighborhood = 10
    translation = collections.defaultdict(int)
    csls_10k = 10000
    normalize(src_embs,['unit', 'center', 'unit'])
    normalize(tgt_embs,['unit', 'center', 'unit'])
    xp = get_cupy()
    src_embs = xp.asarray(src_embs)
    tgt_embs = xp.asarray(tgt_embs)
    src_size = src_embs.shape[0]
    score = {}
    scores = []
    knn_sim_bwd_tgt = xp.zeros(tgt_embs.shape[0])
    for i in range(0, tgt_embs.shape[0], BATCH_SIZE):
        j = min(i + BATCH_SIZE, tgt_embs.shape[0])
        knn_sim_bwd_tgt[i:j] = topk_mean(tgt_embs[i:j].dot(src_embs.T), k=neighborhood, inplace=True)
    for i in range(0, src_size, BATCH_SIZE):
        sc = []
        j = min(i + BATCH_SIZE, src_size)
        similarities = 2 * src_embs[i:j].dot(tgt_embs.T) - knn_sim_bwd_tgt # Equivalent to the real CSLS scores for NN
        sim_cos = src_embs[i:j].dot(tgt_embs.T)
        nn = similarities.argmax(axis=1).tolist()
        for m, n in enumerate(nn):
            score[m + i] = n
            if m+i < csls_10k:
                sc.append(sim_cos[m][n])
        if i + BATCH_SIZE < csls_10k:
            # print(np.mean(sc))
            scores.append(np.mean(sc))
    return score,np.mean(scores)

def get_mapped_embs(src_embs, tgt_embs, W):
    avg_src = np.mean(src_embs, axis=0)
    avg_tgt = np.mean(tgt_embs, axis=0)
    src_embs -= avg_src
    src_embs_ = np.dot(src_embs,W)
    mapped_src_embs = src_embs_ + avg_tgt
    return mapped_src_embs

def get_isometric_words(we1,we2,transform_src,transform_tgt,src2src,tgt2tgt):
    xp = get_cupy()
    transform_src = xp.asarray(transform_src)
    transform_tgt = xp.asarray(transform_tgt)
    src_emb = xp.asarray(we1.vectors)
    tgt_emb = xp.asarray(we2.vectors)
    src_emb_test = xp.asarray(we1.testVectors)
    tgt_emb_test = xp.asarray(we2.testVectors)
    transformed_src = get_mapped_embs(src_emb, tgt_emb_test, transform_src)
    transformed_tgt = get_mapped_embs(tgt_emb, src_emb_test, transform_tgt)
    src2trans,csls_src = CSLS(transformed_src, tgt_emb_test)
    tgt2trans,csls_tgt = CSLS(transformed_tgt, src_emb_test)
    trans_dic = []
    lex = {}
    for A1, B1 in src2trans.items():
        pair = []
        A2 = src2src[A1]
        B2 = tgt2tgt[B1]
        if B2 in tgt2trans.keys() and A2 == tgt2trans[B2]:
            pair.append(we1.id2words[A1])
            pair.append(we2.id2words[B1])
            trans_dic.append(pair)
            lex[we1.id2words[A1]] = we2.id2words[B1]
    return trans_dic,csls_src,csls_tgt,lex
