import numpy as np
from collections import defaultdict
import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def recallk(output, target, k = 10):
    avg_recallk = np.array([])
    eps = 1e-7
    users = list(output['user'].unique())
    output_dict = defaultdict(list)
    target_dict = defaultdict(list)

    for row in output.iterrows():
        output_dict[row[1]['user']].append(row[1]['item'])
    for row in target.iterrows():
        target_dict[row[1]['user']].append(row[1]['item'])
    
    for user in users:
        inter = len(set(output_dict[user][:k]) & set(target_dict[user]))
        div = len(target_dict[user]) + eps
        avg_recallk = np.append(avg_recallk, (inter / div))
    return avg_recallk.mean()