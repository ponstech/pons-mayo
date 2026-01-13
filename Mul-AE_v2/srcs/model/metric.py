import torch
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np


def accuracy(output, target):
    with torch.no_grad():
        pred = output
        assert len(pred) == len(target)
        correct = 0
        # correct += torch.sum(pred == target).item()
        correct += (pred == target).sum()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


# def F1(true, pred):
def F1(pred, true):
    pred = pred.reshape(-1, 1)
    true = true.reshape(-1, 1)
    F1s = []
    precisions = []
    recalls = []
    if true.shape != pred.shape:
        print("Two array must have exactly the same dimension!!")
        return []
    for ix in range(true.shape[1]):
        F1s.append(f1_score(true[:, ix], pred[:, ix], zero_division=0))
        precisions.append(precision_score(true[:, ix], pred[:, ix],zero_division=0))
        recalls.append(recall_score(true[:, ix], pred[:, ix], zero_division=0))
    f1 = np.array(F1s, dtype=np.float32)
    precision = np.array(precisions, dtype=np.float32)
    recall = np.array(recalls, dtype=np.float32)
    return f1, np.mean(f1), precision, np.mean(precision), recall, np.mean(recall)