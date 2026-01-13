import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd


def plot_confusion_matrix(y_true, y_pred, out_path: str, class_names=("BENIGN", "MALIGNANT"), normalize=True, title: str | None = None, show_counts: bool = True):
	cm_raw = confusion_matrix(y_true, y_pred)
	if normalize:
		cm = cm_raw.astype("float") / cm_raw.sum(axis=1, keepdims=True)
	else:
		cm = cm_raw

	plt.figure(figsize=(5, 4))
	fmt = ".2f" if normalize else "d"
	if show_counts and normalize:
		# Build annotations with count and percent
		annot = np.empty_like(cm).astype(object)
		for i in range(cm.shape[0]):
			for j in range(cm.shape[1]):
				annot[i, j] = f"{cm_raw[i, j]}\n({cm[i, j]*100:.1f}%)"
		sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
	else:
		sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", xticklabels=class_names, yticklabels=class_names)

	plt.xlabel("Predicted")
	plt.ylabel("True")
	plt.title(title if title is not None else ("Confusion Matrix" + (" (Normalized)" if normalize else "")))
	os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
	plt.tight_layout()
	plt.savefig(out_path, dpi=150)
	plt.close()


def save_metrics_csv(metrics: dict, out_path: str):
	os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
	df = pd.DataFrame([metrics])
	df.to_csv(out_path, index=False)
