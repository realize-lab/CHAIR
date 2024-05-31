"""
Credit: recall@k implementation from : https://github.com/leftthomas/CGD
"""

import torch
from typing import List


def recall(query_features, query_labels, rank: List[int], gallery_features=None, gallery_labels=None, ret_num: bool=False):
    num_querys = len(query_labels)
    gallery_features = query_features if gallery_features is None else gallery_features

    cosine_matrix = query_features @ gallery_features.t()

    if gallery_labels is None:
        cosine_matrix.fill_diagonal_(-float('inf'))
        gallery_labels = query_labels
    else:
        pass


    idx = cosine_matrix.topk(k=rank[-1], dim=-1, largest=True)[1]

    recall_list = []
    num_recall_list = []

    print(f"Shapes: \n query_labels: {query_labels.shape} \n idx: {idx.shape} \n gallery_labels: {gallery_labels.shape}")
    
    for r in rank:
        correct = (gallery_labels[idx[:, 0:r]] == query_labels.unsqueeze(dim=-1)).float()
        print(f"correct: {correct.shape}")
        sum_correct = correct.sum(dim=1).clip(max=1)
        recall_list.append(sum_correct.sum().item() / num_querys)
        num_recall_list.append(correct.sum().item() / (num_querys * r))

    if ret_num:
        return recall_list, num_recall_list
        
    return recall_list
