from tqdm import tqdm
import numpy as np
from metrics_module.eer import compute_eer
from metrics_module.dcf import *

from typing import Dict, List, Optional
def cosine_score(
    trials : List,
    index_mapping: Dict, # {'spk_i':embedding_i (n,Dim)}
    eval_vectors: Dict
    ):
    all_scores = []
    all_labels = []
    target_scores = []
    nontarget_scores = []
    for item in trials:
        all_labels.append(int(item[0])) # trials <label> <spk1> <spk2>
        enroll_vector = eval_vectors[index_mapping[item[1]]] # n,Dim
        test_vector = eval_vectors[index_mapping[item[2]]] # n,Dim
        dim = len(enroll_vector)
        score = enroll_vector.dot(test_vector.T)
        norm = np.linalg.norm(enroll_vector) * np.linalg.norm(test_vector)
        score = dim * score / norm
        all_scores.append(score)
        if item[0] == "1":
            target_scores.append(score)
        else:
            nontarget_scores.append(score)
    eer, th = compute_eer(target_scores, nontarget_scores)

    c_miss = 1
    c_fa = 1
    fnrs, fprs, thresholds = ComputeErrorRates(all_scores, all_labels)
    mindcf_easy, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.01, c_miss, c_fa)
    mindcf_hard, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.001, c_miss, c_fa)

    return eer, th, mindcf_easy, mindcf_hard
def multi_cosine_score():
    pass
def PLDA_score(trials, index_mapping, eval_vectors, plda_analyzer):
    all_scores = []
    all_labels = []
    target_scores = []
    nontarget_scores= []
    for item in trials:
        all_labels.append(int(item[0]))
        enroll_vector = eval_vectors[index_mapping[item[1]]]
        test_vector = eval_vectors[index_mapping[item[2]]]
        score = plda_analyzer.NLScore(enroll_vector, test_vector)
        all_scores.append(score)
        if item[0] == "1":
            target_scores.append(score)
        else:
            nontarget_scores.append(score)
    eer, th = compute_eer(target_scores, nontarget_scores)

    c_miss = 1
    c_fa = 1
    fnrs, fprs, thresholds = ComputeErrorRates(all_scores, all_labels)
    mindcf_easy, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.01, c_miss, c_fa)
    mindcf_hard, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.001, c_miss, c_fa)

    return eer, th, mindcf_easy, mindcf_hard
