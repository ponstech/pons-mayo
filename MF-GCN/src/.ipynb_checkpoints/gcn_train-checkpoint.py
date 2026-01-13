import os
from typing import Dict, Tuple, Optional
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

from utils.GN_models import (
	GCNNClassifier,
	GCNN_Dot_Product,
	GCNN_Concat_Attention,
	GCNN_Prod_Res,
	GATClassifier,
	GraphSAGEClassifier,
	GNClassifier,
)


MODEL_ZOO = {
	"GCNNClassifier": GCNNClassifier,
	"GCNN_Dot_Product": GCNN_Dot_Product,
	"GCNN_Concat_Attention": GCNN_Concat_Attention,
	"GCNN_Prod_Res": GCNN_Prod_Res,
	"GATClassifier": GATClassifier,
	"GraphSAGEClassifier": GraphSAGEClassifier,
	"GNClassifier": GNClassifier,
}


def compute_homophily(graph) -> float:
	src, dst = graph.edge_index
	if graph.num_edges == 0:
		return 0.0
	return (graph.y[src] == graph.y[dst]).float().mean().item()


def save_graph_image(graph, out_path: str, title_prefix: str = "Graph"):
	fig, ax = plt.subplots(figsize=(6, 5))
	G = to_networkx(graph, to_undirected=True)
	nodes = G.number_of_nodes()
	h = compute_homophily(graph)
	node_colors = ["tab:blue" if int(graph.y[i].item()) == 0 else "tab:red" for i in range(nodes)]
	nx.draw(G, node_size=20, node_color=node_colors, with_labels=False, ax=ax)
	ax.set_title(f"{title_prefix} | N={nodes} | homophily={h:.3f}")
	os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
	fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
	plt.close(fig)


def aggregate_predictions_by_patient(logits: torch.Tensor, patient_ids: torch.Tensor, threshold: float = 0.8):
	from collections import defaultdict
	patient_logits = defaultdict(list)
	for i in range(len(logits)):
		pid = int(patient_ids[i].item())
		patient_logits[pid].append(logits[i])

	avg_logits, preds = {}, {}
	for pid, lst in patient_logits.items():
		tensor = torch.stack(lst)
		probs = torch.softmax(tensor, dim=1)
		mask = (probs.max(dim=1).values >= threshold)
		filtered = tensor if mask.sum() == 0 else tensor[mask]
		mean_logit = filtered.mean(dim=0)
		avg_logits[pid] = mean_logit
		preds[pid] = int(mean_logit.argmax().item())
	return avg_logits, preds


def build_model(name: str, in_channels: int, hidden_channels: int, num_classes: int, device: torch.device):
	Model = MODEL_ZOO[name]
	if name in ("GCNN_Dot_Product", "GCNN_Prod_Res"):
		model = Model(in_channels, in_channels, num_classes).to(device)
	else:
		model = Model(in_channels, hidden_channels, num_classes).to(device)
	return model


def train_gcn(model_name: str, train_graph, val_graph, hidden_channels: int, lr: float, weight_decay: float, epochs: int, patience: int, aggregation_threshold: float, save_dir: str, device: torch.device, plot_path: Optional[str] = None) -> str:
	os.makedirs(save_dir, exist_ok=True)
	in_channels = train_graph.x.shape[1]
	model = build_model(model_name, in_channels, hidden_channels, num_classes=2, device=device)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
	criterion = torch.nn.CrossEntropyLoss()

	best_val = 0.0
	counter = 0
	best_path = os.path.join(save_dir, f"best_{model_name}.pth")

	train_graph = train_graph.to(device)
	val_graph = val_graph.to(device)
	train_losses: list[float] = []
	val_accs: list[float] = []

	for epoch in range(epochs):
		model.train()
		optimizer.zero_grad()
		out = model(train_graph.x, train_graph.edge_index)
		loss = criterion(out, train_graph.y)
		loss.backward()
		optimizer.step()

		model.eval()
		with torch.no_grad():
			vout = model(val_graph.x, val_graph.edge_index)
			val_acc = (vout.argmax(1) == val_graph.y).float().mean().item()

		print(f"Epoch {epoch+1}/{epochs} | Train Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f}")
		train_losses.append(loss.item())
		val_accs.append(val_acc)

		if val_acc > best_val:
			best_val = val_acc
			counter = 0
			torch.save(model.state_dict(), best_path)
			print(f"Saved best GCN ({model_name}) with Val Acc: {best_val:.4f}")
		else:
			counter += 1
			if counter >= patience:
				print("Early stopping GCN training")
				break

	# Save loss/val plot if requested
	if plot_path:
		os.makedirs(os.path.dirname(plot_path), exist_ok=True)
		fig2, ax2 = plt.subplots(figsize=(6,4))
		ax2.plot(train_losses, label="Train Loss")
		ax2.plot(val_accs, label="Val Acc")
		ax2.set_xlabel("Epochs")
		ax2.set_ylabel("Loss/Acc")
		ax2.set_title("GCN Train Loss / Val Acc")
		ax2.legend()
		fig2.savefig(plot_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
		plt.close(fig2)

	return best_path


def evaluate_gcn(model_name: str, model_path: str, graph, aggregation_threshold: float, device: torch.device) -> Dict:
	in_channels = graph.x.shape[1]
	model = build_model(model_name, in_channels, hidden_channels=256, num_classes=2, device=device)
	model.load_state_dict(torch.load(model_path, map_location=device))
	model.eval()

	with torch.no_grad():
		out = model(graph.x.to(device), graph.edge_index.to(device))
		pred = out.argmax(1).cpu().numpy()
		true = graph.y.cpu().numpy()

	acc = accuracy_score(true, pred)
	tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
	return {
		"accuracy": acc,
		"tn": int(tn),
		"fp": int(fp),
		"fn": int(fn),
		"tp": int(tp),
	}
