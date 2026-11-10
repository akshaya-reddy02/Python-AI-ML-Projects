from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "creditcard.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.20
EPOCHS = 30
HIDDEN_CHANNELS = 32
K_NEIGHBORS = 5


class FraudGraphSAGE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels=32):
        super().__init__()
        self.conv1 = SAGEConv(input_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.output = torch.nn.Linear(hidden_channels, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return self.output(x)


def build_similarity_edges(features, k=5):
    """Build a sparse graph using nearest neighbours in standardized features."""
    from sklearn.neighbors import NearestNeighbors

    n_rows = features.shape[0]
    if n_rows < 2:
        raise ValueError("At least two transactions are needed to build a graph.")

    k = min(k, n_rows - 1)
    neighbors = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", n_jobs=2)
    neighbors.fit(features)
    indices = neighbors.kneighbors(return_distance=False)

    # Remove each node's self-edge; create bidirectional edges.
    sources = np.repeat(np.arange(n_rows), k)
    targets = np.concatenate([row[row != i][:k] for i, row in enumerate(indices)])

    # Ensure exactly k targets per source even if self is not the first neighbour.
    if len(targets) != len(sources):
        pairs = []
        for i, row in enumerate(indices):
            valid = [int(j) for j in row if int(j) != i][:k]
            pairs.extend((i, j) for j in valid)
        sources = np.array([p[0] for p in pairs], dtype=np.int64)
        targets = np.array([p[1] for p in pairs], dtype=np.int64)

    edge_index = np.vstack([sources, targets])
    reverse_edges = edge_index[::-1]
    edge_index = np.concatenate([edge_index, reverse_edges], axis=1)

    # Remove duplicate edges and return PyTorch Geometric's [2, edges] format.
    edge_index = np.unique(edge_index.T, axis=0).T
    return torch.tensor(edge_index, dtype=torch.long)


def main():
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    print("Loading and preparing transaction data...")
    df = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)

    y = df["Class"].astype(np.int64).to_numpy()
    X = df.drop(columns=["Class"]).astype(np.float32).to_numpy()

    train_indices, test_indices = train_test_split(
        np.arange(len(y)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Fit scaling only on training data to avoid fitting the scaler on test rows.
    scaler = StandardScaler()
    scaler.fit(X[train_indices])
    X_scaled = scaler.transform(X).astype(np.float32)

    print("Building a transaction-similarity graph...")
    edge_index = build_similarity_edges(X_scaled, K_NEIGHBORS)

    x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)

    train_mask = torch.zeros(len(y), dtype=torch.bool)
    test_mask = torch.zeros(len(y), dtype=torch.bool)
    train_mask[train_indices] = True
    test_mask[test_indices] = True

    model = FraudGraphSAGE(
        input_channels=data.num_node_features,
        hidden_channels=HIDDEN_CHANNELS,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

    train_labels = y[train_indices]
    class_counts = np.bincount(train_labels, minlength=2)
    class_weights = class_counts.sum() / (
        2.0 * np.maximum(class_counts, 1)
    )
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

    print(f"Transactions/nodes: {data.num_nodes:,}")
    print(f"Graph edges: {data.num_edges:,}")
    print(f"Training nodes: {len(train_indices):,}")
    print(f"Test nodes: {len(test_indices):,}")
    print("Training GraphSAGE...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(
            logits[train_mask],
            data.y[train_mask],
            weight=weight_tensor,
        )
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:02d}/{EPOCHS} - loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(
            model(data.x, data.edge_index), dim=1
        )[:, 1].cpu().numpy()

    y_test = y[test_indices]
    test_probabilities = probabilities[test_indices]

    average_precision = float(
        average_precision_score(y_test, test_probabilities)
    )
    roc_auc = float(roc_auc_score(y_test, test_probabilities))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "fraud_gnn_weights.pt"
    scaler_path = MODEL_DIR / "gnn_scaler.joblib"
    report_path = REPORT_DIR / "gnn_evaluation.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_channels": int(data.num_node_features),
            "hidden_channels": HIDDEN_CHANNELS,
            "epochs": EPOCHS,
            "k_neighbors": K_NEIGHBORS,
            "random_state": RANDOM_STATE,
        },
        model_path,
    )

    import joblib
    joblib.dump(scaler, scaler_path)

    report = {
        "model": "GraphSAGE",
        "graph_method": "k-nearest-neighbour transaction similarity",
        "warning": (
            "This dataset has no real customer, device, or location IDs. "
            "Edges represent feature similarity, not proven fraud relationships. "
            "This transductive experiment includes test nodes in graph construction; "
            "results are exploratory and are not a deployment-ready estimate."
        ),
        "rows_after_duplicate_removal": int(len(df)),
        "fraud_rows": int(y.sum()),
        "train_rows": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
        "nodes": int(data.num_nodes),
        "directed_edges_after_symmetrization": int(data.num_edges),
        "epochs": EPOCHS,
        "average_precision": average_precision,
        "roc_auc": roc_auc,
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nGNN training completed.")
    print(f"Average Precision: {average_precision:.6f}")
    print(f"ROC-AUC: {roc_auc:.6f}")
    print(f"Model weights: {model_path}")
    print(f"Scaler: {scaler_path}")
    print(f"Evaluation report: {report_path}")


if __name__ == "__main__":
    main()