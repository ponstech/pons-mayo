import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import random


def _normalize_features(features: np.ndarray, zscore: bool = True, l2: bool = False) -> np.ndarray:
	X = features.astype(np.float32)
	if zscore:
		mu = X.mean(axis=0, keepdims=True)
		sigma = X.std(axis=0, keepdims=True) + 1e-8
		X = (X - mu) / sigma
	if l2:
		norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
		X = X / norm
	return X


def graph_cosine_topk(features: np.ndarray, labels: np.ndarray, df: pd.DataFrame, top_k: int = 7, normalize_zscore: bool = True, normalize_l2: bool = False) -> Data:
	num_nodes = features.shape[0]

	features = _normalize_features(features, zscore=normalize_zscore, l2=normalize_l2)
	feature_tensor = torch.tensor(features, dtype=torch.float32)
	labels_tensor = torch.tensor(labels, dtype=torch.long)

	norms = np.linalg.norm(features, axis=1, keepdims=True)
	norms[norms == 0] = 1e-9
	cos_sim = (features @ features.T) / (norms @ norms.T)

	adj = np.zeros((num_nodes, num_nodes), dtype=np.int32)
	for i in range(num_nodes):
		cos_sim[i, i] = -np.inf
		neighbors = np.argsort(cos_sim[i])[-top_k:]
		adj[i, neighbors] = 1
		for j in neighbors:
			adj[j, i] = 1

	edge_index = np.array(np.nonzero(adj))
	edge_index = torch.tensor(edge_index, dtype=torch.long)

	return Data(x=feature_tensor, edge_index=edge_index, y=labels_tensor)


def graph_sparse_random(features_df: pd.DataFrame, labels: np.ndarray, max_neighbors: int = 10, seed: int = 42) -> Data:
	"""Build a sparse random neighbor graph (no patient ids required)."""
	random.seed(seed)
	node_features = torch.tensor(features_df.drop(columns=['Label', 'Image']).values, dtype=torch.float32)
	y = torch.tensor(labels, dtype=torch.long)

	num_nodes = node_features.shape[0]
	edge_index = []
	for i in range(num_nodes):
		# sample up to max_neighbors distinct nodes (excluding self)
		candidates = [j for j in range(num_nodes) if j != i]
		if not candidates:
			continue
		chosen = random.sample(candidates, min(max_neighbors, len(candidates)))
		for j in chosen:
			edge_index.append([i, j])
			edge_index.append([j, i])  # make undirected

	edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2,0), dtype=torch.long)
	return Data(x=node_features, edge_index=edge_index, y=y)
