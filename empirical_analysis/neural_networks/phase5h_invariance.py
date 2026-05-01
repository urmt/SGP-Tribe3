#!/usr/bin/env python3
"""
PHASE 5H: INVARIANCE OF RANKING UNDER OPERATOR TRANSFORMATIONS (OPTIMIZED)
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from scipy.stats import spearmanr

np.random.seed(42)
torch.manual_seed(42)

OUTPUT_DIR = "empirical_analysis/neural_networks/phase5h"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OPERATORS = {
    "identity": lambda x: x,
    "tanh_0.5": lambda x: np.tanh(0.5 * x),
    "tanh_1.0": lambda x: np.tanh(1.0 * x),
    "tanh_2.0": lambda x: np.tanh(2.0 * x),
    "sigmoid": lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500))),
    "softplus": lambda x: np.log1p(np.exp(np.clip(x, -500, 500))),
    "relu": lambda x: np.maximum(0, x),
    "scaled_0.5": lambda x: 0.5 * x,
    "scaled_2.0": lambda x: 2.0 * x
}

def compute_fertility_fast(x):
    eps_values = np.logspace(-3, -1, 8)
    det_values = []
    for eps in eps_values:
        dist = np.abs(x[:, None] - x[None, :])
        R = (dist < eps).astype(int)
        np.fill_diagonal(R, 0)
        
        N = R.shape[0]
        diag_counts = 0
        for k in range(-N+1, N):
            diag = np.diagonal(R, offset=k)
            length = 0
            for val in diag:
                if val == 1:
                    length += 1
                else:
                    if length >= 2:
                        diag_counts += length
                    length = 0
            if length >= 2:
                diag_counts += length
        
        total_rec = np.sum(R)
        if total_rec == 0:
            det = 0.0
        else:
            det = diag_counts / total_rec
        det_values.append(det)
    
    return np.trapz(det_values, eps_values)

class MLP(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
    
    def forward(self, x):
        self.activations = [x]
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
            self.activations.append(x)
        return x

print("Loading MNIST...")
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)

X_train = train_ds.data.float().view(-1, 784).numpy()[:4000] / 255.0
y_train = train_ds.targets.numpy()[:4000]
X_test = test_ds.data.float().view(-1, 784).numpy()[:1000] / 255.0
y_test = test_ds.targets.numpy()[:1000]

X_t = torch.FloatTensor(X_train)
y_t = torch.LongTensor(y_train)
X_v = torch.FloatTensor(X_test)
y_v = torch.LongTensor(y_test)
train_ds = TensorDataset(X_t, y_t)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

NUM_MODELS = 8

print(f"\nTraining {NUM_MODELS} models...")
model_activations = []

for seed in range(NUM_MODELS):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    model = MLP([784, 64, 32, 10])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(15):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        _ = model(X_v)
        hidden_acts = model.activations[1].numpy()
        act_flat = hidden_acts.mean(axis=0)
        
        preds = model(X_v).argmax(dim=1).numpy()
        acc = (preds == y_v.numpy()).mean()
    
    model_activations.append({
        "seed": seed,
        "activations": act_flat,
        "accuracy": acc
    })
    print(f"  Seed {seed}: accuracy={acc:.3f}")

print(f"\nComputing Fertility for {len(OPERATORS)} operators...")

F_results = {op: {} for op in OPERATORS.keys()}
accuracy_map = {}

for i, model_data in enumerate(model_activations):
    model_name = f"model_seed{model_data['seed']}"
    accuracy_map[model_name] = model_data['accuracy']
    x_flat = model_data['activations']
    
    for op_name, op_fn in OPERATORS.items():
        x_transformed = op_fn(x_flat)
        F = compute_fertility_fast(x_transformed)
        F_results[op_name][model_name] = F

rankings = {}
for op_name, model_dict in F_results.items():
    sorted_models = sorted(model_dict.items(), key=lambda x: x[1], reverse=True)
    rankings[op_name] = [m[0] for m in sorted_models]

ops = list(OPERATORS.keys())
n_ops = len(ops)
corr_matrix = np.zeros((n_ops, n_ops))

for i in range(n_ops):
    for j in range(n_ops):
        rank_i = rankings[ops[i]]
        rank_j = rankings[ops[j]]
        rank_map_i = {m: idx for idx, m in enumerate(rank_i)}
        rank_map_j = {m: idx for idx, m in enumerate(rank_j)}
        common_models = list(rank_map_i.keys())
        r_i = [rank_map_i[m] for m in common_models]
        r_j = [rank_map_j[m] for m in common_models]
        corr, _ = spearmanr(r_i, r_j)
        corr_matrix[i, j] = corr

print("\n" + "="*60)
print("CORRELATION MATRIX")
print("="*60)
print(np.round(corr_matrix, 2))

min_corr = np.min(corr_matrix[np.triu_indices(n_ops, k=1)])
max_corr = np.max(corr_matrix[np.triu_indices(n_ops, k=1)])
mean_corr = np.mean(corr_matrix[np.triu_indices(n_ops, k=1)])

print(f"\nMin: {min_corr:.3f}, Max: {max_corr:.3f}, Mean: {mean_corr:.3f}")

if min_corr > 0.8:
    case = "CASE 1: STRONG INVARIANCE"
elif min_corr > 0.5:
    case = "CASE 2: PARTIAL INVARIANCE"
else:
    case = "CASE 3: NO INVARIANCE"

print(f"\n{case}")

output = {
    "operators": ops,
    "correlation_matrix": corr_matrix.tolist(),
    "accuracy_map": accuracy_map,
    "min_correlation": min_corr,
    "max_correlation": max_corr,
    "mean_correlation": mean_corr,
    "case": case
}

with open(os.path.join(OUTPUT_DIR, "invariance_results.json"), "w") as f:
    json.dump(output, f, indent=2)

print(f"\nPhase 5H complete.")