import torch
import bottleneck as bn
import numpy as np
from scipy.sparse import csr_matrix


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


def recall_at_k_batch(X_pred, heldout_batch, k=10):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)  # 큰 요소 k개의 인덱스가 앞에 오도록 해서 인덱스를 뽑아냄
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)  # (500, 8607) 모양의 0으로 채워진 배열 만듦
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True  # k개를 true로 바꿔줌

    X_true_binary = (heldout_batch > 0)  # 바깥에 뺴놓은 데이터에 대해서만 True가 매겨짐
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(np.float32)  # 둘 다 and인 것을 골라내서 갯수를 세어줌
    
    n_hidden = X_true_binary.sum(axis=1)
    n_hidden[n_hidden==0] = 1  # 0인 값을 충분히 작은 숫자로 바꿔줌 (n_hidden이 0이면 어차피 분자는 0)
    recall = tmp / np.minimum(k, n_hidden) 
    recall = recall.mean(axis=0).item()
    print(f'recall: {recall}')
    
    return recall

