
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xgboost as xgb

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "creditcard.csv"
REPORT_PATH = ROOT / "reports" / "blend_evaluation.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
EPOCHS = 30
HIDDEN_CHANNELS = 32
K_NEIGHBORS = 5
BLEND_WEIGHT_XGB = 0.70
THRESHOLD = 0.50


class FraudGraphSAGE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels=32):
        super().__init__()
        self.conv1 = SAGEConv(input_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.output = torch.nn.Linear(hidden_channels, 2)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.25, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return self.output(x)


def build_similarity_edges(features, k=5):
    n_rows = len(features)
    if n_rows < 2:
        raise ValueError("At least two transactions are required.")

    k = min(k, n_rows - 1)
    neighbors = NearestNeighbors(
        n_neighbors=k + 1,
        algorithm="brute",
        n_jobs=2,
    )
    neighbors.fit(features)
    indices = neighbors.kneighbors(return_distance=False)

    pairs = []
    for i, row in enumerate(indices):
        valid = [int(j) for j in row if int(j) != i][:k]
        pairs.extend((i, j) for j in valid)
        pairs.extend((j, i) for j in valid)

    edge_index = np.unique(
        np.asarray(pairs, dtype=np.int64), axis=0
    ).T

    return torch.tensor(edge_index, dtype=torch.long)


def calculate_metrics(y_true, probabilities):
    predictions = (probabilities >= THRESHOLD).astype(int)

    return {
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "precision_at_0_50": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall_at_0_50": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1_at_0_50": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
    }


def main():
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    print("Loading and deduplicating dataset...")
    df = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)

    y = df["Class"].astype(np.int64).to_numpy()
    X = df.drop(columns=["Class"]).astype(np.float32).to_numpy()

    train_indices, test_indices = train_test_split(
        np.arange(len(y)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    # Model 1: XGBoost
    print("\nTraining XGBoost...")
    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())

    xgb_model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        scale_pos_weight=negatives / max(positives, 1),
        random_state=RANDOM_STATE,
        n_jobs=4,
    )
    xgb_model.fit(X_train, y_train)
    xgb_probabilities = xgb_model.predict_proba(X_test)[:, 1]

    # Fit the scaler using training rows only.
    print("\nScaling features using training data only...")
    scaler = StandardScaler()
    scaler.fit(X_train)
    scaled_features = scaler.transform(X).astype(np.float32)

    # Model 2: GraphSAGE
    # Test nodes are present in the feature-only graph, but their labels
    # are never used in the training loss.
    print("\nBuilding transaction-similarity graph...")
    edge_index = build_similarity_edges(
        scaled_features, K_NEIGHBORS
    )

    graph = Data(
        x=torch.tensor(scaled_features, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.long),
    )

    train_mask = torch.zeros(len(y), dtype=torch.bool)
    test_mask = torch.zeros(len(y), dtype=torch.bool)
    train_mask[train_indices] = True
    test_mask[test_indices] = True

    gnn_model = FraudGraphSAGE(
        input_channels=graph.num_node_features,
        hidden_channels=HIDDEN_CHANNELS,
    )
    optimizer = torch.optim.Adam(
        gnn_model.parameters(),
        lr=0.005,
        weight_decay=1e-4,
    )

    class_counts = np.bincount(y_train, minlength=2)
    class_weights = class_counts.sum() / (
        2.0 * np.maximum(class_counts, 1)
    )
    weight_tensor = torch.tensor(
        class_weights, dtype=torch.float32
    )

    print(f"Training rows: {len(train_indices):,}")
    print(f"Test rows: {len(test_indices):,}")
    print(f"Graph nodes: {graph.num_nodes:,}")
    print(f"Graph edges: {graph.num_edges:,}")
    print("Training GraphSAGE...")

    for epoch in range(1, EPOCHS + 1):
        gnn_model.train()
        optimizer.zero_grad()

        logits = gnn_model(graph.x, graph.edge_index)
        loss = F.cross_entropy(
            logits[train_mask],
            graph.y[train_mask],
            weight=weight_tensor,
        )
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:02d}/{EPOCHS} "
                f"- loss: {loss.item():.6f}"
            )

    gnn_model.eval()
    with torch.no_grad():
        all_gnn_probabilities = torch.softmax(
            gnn_model(graph.x, graph.edge_index), dim=1
        )[:, 1].cpu().numpy()

    gnn_probabilities = all_gnn_probabilities[test_indices]

    # Fixed 70/30 blend; weights are not tuned on the test set.
    blended_probabilities = (
        BLEND_WEIGHT_XGB * xgb_probabilities
        + (1.0 - BLEND_WEIGHT_XGB) * gnn_probabilities
    )

    results = {
        "xgboost": calculate_metrics(y_test, xgb_probabilities),
        "gnn": calculate_metrics(y_test, gnn_probabilities),
        "blend": calculate_metrics(y_test, blended_probabilities),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "experiment": "XGBoost and GraphSAGE weighted probability comparison",
        "dataset_rows_after_deduplication": int(len(df)),
        "train_rows": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
        "fraud_test_rows": int(y_test.sum()),
        "random_state": RANDOM_STATE,
        "gnn_epochs": EPOCHS,
        "decision_threshold": THRESHOLD,
        "xgboost_blend_weight": BLEND_WEIGHT_XGB,
        "gnn_blend_weight": 1.0 - BLEND_WEIGHT_XGB,
        "graph_method": "k-nearest-neighbour transaction feature similarity",
        "limitations": [
            "The dataset has no real customer, device, or location IDs.",
            "Graph edges represent feature similarity, not proven fraud relationships.",
            "The graph includes test nodes and their features, but not their labels in training loss; this is a transductive experiment.",
            "The 70/30 blend is a baseline, not a proven optimal combination.",
            "Results are exploratory and do not establish production performance."
        ],
        "metrics": results,
    }

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nBLENDING EVALUATION COMPLETED")
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.6f}")

    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
