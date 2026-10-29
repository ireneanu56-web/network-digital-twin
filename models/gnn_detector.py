from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from models.preprocess import RANDOM_SEED


TOPOLOGY_NODES = ["h1", "h2", "h3", "h4", "s1", "s2", "s3", "s4"]
TOPOLOGY_EDGES = [
    ("h1", "s1"),
    ("s1", "s2"),
    ("s1", "s3"),
    ("s2", "s4"),
    ("s3", "s4"),
    ("s4", "h4"),
    ("h2", "s2"),
    ("h3", "s3"),
]


def normalized_adjacency():
    node_indices = {name: index for index, name in enumerate(TOPOLOGY_NODES)}
    adjacency = torch.eye(len(TOPOLOGY_NODES), dtype=torch.float32)
    for left, right in TOPOLOGY_EDGES:
        first, second = node_indices[left], node_indices[right]
        adjacency[first, second] = 1.0
        adjacency[second, first] = 1.0
    degree = adjacency.sum(dim=1).clamp(min=1.0)
    inverse_sqrt_degree = degree.rsqrt()
    return inverse_sqrt_degree[:, None] * adjacency * inverse_sqrt_degree[None, :]


class TopologyFaultGNN(nn.Module):
    """Classify network-wide telemetry using the known Mininet graph structure."""

    def __init__(self, feature_count, class_count, hidden_size=16):
        super().__init__()
        self.register_buffer("adjacency", normalized_adjacency())
        self.node_embedding = nn.Embedding(len(TOPOLOGY_NODES), hidden_size)
        self.context_projection = nn.Linear(feature_count, hidden_size)
        self.self_layer_1 = nn.Linear(hidden_size, hidden_size)
        self.neighbor_layer_1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.self_layer_2 = nn.Linear(hidden_size, hidden_size)
        self.neighbor_layer_2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, class_count),
        )

    def forward(self, telemetry):
        batch_size = telemetry.shape[0]
        context = self.context_projection(telemetry)
        node_ids = torch.arange(len(TOPOLOGY_NODES), device=telemetry.device)
        nodes = context[:, None, :] + self.node_embedding(node_ids)[None, :, :]

        neighbors = torch.einsum("ij,bjh->bih", self.adjacency, nodes)
        nodes = functional.relu(
            self.self_layer_1(nodes) + self.neighbor_layer_1(neighbors)
        )
        neighbors = torch.einsum("ij,bjh->bih", self.adjacency, nodes)
        nodes = functional.relu(
            self.self_layer_2(nodes) + self.neighbor_layer_2(neighbors)
        )
        graph_embedding = nodes.mean(dim=1)
        return self.classifier(torch.cat((context, graph_embedding), dim=1))


def train_gnn(data: dict, model_path: Path, epochs=120):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    model = TopologyFaultGNN(
        feature_count=data["X_train"].shape[1],
        class_count=len(data["labels"]),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
    loss_function = nn.CrossEntropyLoss()
    train_features = torch.tensor(data["X_train"], dtype=torch.float32)
    train_labels = torch.tensor(data["y_train"], dtype=torch.long)
    val_features = torch.tensor(data["X_val"], dtype=torch.float32)
    val_labels = torch.tensor(data["y_val"], dtype=torch.long)

    best_loss = float("inf")
    best_state = None
    patience = 15
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = loss_function(model(train_features), train_labels)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss = loss_function(
                model(val_features), val_labels
            ).item()
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}/{epochs} - "
                f"loss: {loss.item():.4f} - val_loss: {validation_loss:.4f}"
            )
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("GNN training did not produce a valid model state")
    model.load_state_dict(best_state)
    model.eval()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_count": int(data["X_train"].shape[1]),
            "class_count": int(len(data["labels"])),
            "labels": [str(label) for label in data["labels"]],
            "nodes": TOPOLOGY_NODES,
            "edges": TOPOLOGY_EDGES,
        },
        model_path,
    )
    return model


def load_gnn(model_path: Path):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = TopologyFaultGNN(
        feature_count=checkpoint["feature_count"],
        class_count=checkpoint["class_count"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def predict_gnn(model, features):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(features, dtype=torch.float32))
    return logits.argmax(dim=1).cpu().numpy()
