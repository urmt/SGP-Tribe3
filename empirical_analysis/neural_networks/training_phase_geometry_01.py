#!/usr/bin/env python3
"""
SFH-SGP_TRAINING_PHASE_GEOMETRY_01
Track Φ(S, O) across training epochs
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import pairwise_distances
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

epsilon = 1e-2
l_min = 2
eps_list = np.logspace(-3, -1, 8)

def recurrence_matrix(x, epsilon):
    x = x.reshape(-1, 1)
    D = pairwise_distances(x)
    R = (D < epsilon).astype(int)
    np.fill_diagonal(R, 0)
    return R

def compute_det(R, l_min=2):
    N = R.shape[0]
    diag_counts = 0
    total_rec = np.sum(R)
    for k in range(-N+1, N):
        diag = np.diagonal(R, offset=k)
        length = 0
        for val in diag:
            if val == 1:
                length += 1
            else:
                if length >= l_min:
                    diag_counts += length
                length = 0
        if length >= l_min:
            diag_counts += length
    if total_rec == 0:
        return 0.0
    return diag_counts / total_rec

def recurrence_rate(R):
    N = R.shape[0]
    return np.sum(R) / (N * N - N) if N > 1 else 0

def compute_alpha(x, eps_list):
    R_vals = []
    for eps in eps_list:
        R = recurrence_matrix(x, eps)
        R_vals.append(recurrence_rate(R))
    log_eps = np.log(eps_list)
    log_R = np.log(np.array(R_vals) + 1e-10)
    valid = np.isfinite(log_R)
    if valid.sum() > 2:
        alpha = np.polyfit(log_eps[valid], log_R[valid], 1)[0]
    else:
        alpha = np.nan
    return alpha

def apply_operator(x, operator):
    if operator == 'identity':
        return x
    elif operator == 'normalization':
        return (x - np.mean(x)) / (np.std(x) + 1e-8)
    elif operator == 'tanh':
        return np.tanh(x)
    return x

def compute_phi(x, operator):
    x_op = apply_operator(x, operator)
    if len(x_op) < 100:
        return np.nan, np.nan
    alpha = compute_alpha(x_op, eps_list)
    R = recurrence_matrix(x_op, epsilon)
    det = compute_det(R, l_min)
    return alpha, det

print("Loading MNIST...")
from torchvision import datasets, transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)

X_train = train_ds.data.float().view(-1, 784).numpy()[:6000] / 255.0
y_train = train_ds.targets.numpy()[:6000]
X_test = test_ds.data.float().view(-1, 784).numpy()[:2000] / 255.0
y_test = test_ds.targets.numpy()[:2000]
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

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

model = MLP([784, 64, 32, 10])

X_t = torch.FloatTensor(X_train)
y_t = torch.LongTensor(y_train)
X_v = torch.FloatTensor(X_test)
y_v = torch.LongTensor(y_test)

train_ds = TensorDataset(X_t, y_t)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

snapshots = [1, 3, 5, 10, 20, 30, 50]
snapshot_states = {}
results = []

print("\n" + "="*60)
print("TRAINING WITH SNAPSHOTS")
print("="*60)

for epoch in range(1, 51):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    if epoch in snapshots:
        model.eval()
        with torch.no_grad():
            preds = model(X_v).argmax(dim=1).numpy()
            acc = (preds == y_v.numpy()).mean()
        
        print(f"Epoch {epoch}: accuracy={acc:.3f}")
        
        with torch.no_grad():
            _ = model(X_v)
        
        layer_names = ['input', 'hidden_0', 'hidden_1', 'output']
        operators = ['identity', 'tanh', 'normalization']
        
        for layer_idx, act in enumerate(model.activations):
            act_np = act.numpy()
            if act_np.ndim > 1:
                act_flat = act_np.mean(axis=1)
            else:
                act_flat = act_np
            
            for op in operators:
                alpha, det = compute_phi(act_flat, op)
                results.append({
                    'epoch': epoch,
                    'accuracy': acc,
                    'layer': layer_names[layer_idx],
                    'layer_idx': layer_idx,
                    'operator': op,
                    'alpha': alpha,
                    'det': det
                })
                print(f"  {layer_names[layer_idx]:10} | {op:15} | α={alpha:.3f}, DET={det:.3f}")

df = pd.DataFrame(results)
df.to_csv('training_phi_trajectory.csv', index=False)

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)

print("\n--- Φ evolution with training ---")
for layer in ['input', 'hidden_0', 'hidden_1', 'output']:
    print(f"\nLayer: {layer}")
    for op in ['identity', 'tanh', 'normalization']:
        layer_df = df[(df['layer'] == layer) & (df['operator'] == op)]
        first_alpha = layer_df[layer_df['epoch'] == 1]['alpha'].values[0]
        last_alpha = layer_df[layer_df['epoch'] == layer_df['epoch'].max()]['alpha'].values[0]
        first_det = layer_df[layer_df['epoch'] == 1]['det'].values[0]
        last_det = layer_df[layer_df['epoch'] == layer_df['epoch'].max()]['det'].values[0]
        print(f"  {op:15}: α {first_alpha:.3f}→{last_alpha:.3f} (Δ={last_alpha-first_alpha:+.3f}), DET {first_det:.3f}→{last_det:.3f} (Δ={last_det-first_det:+.3f})")

print("\n--- Correlation with accuracy ---")
for op in ['identity', 'tanh', 'normalization']:
    op_df = df[df['operator'] == op]
    alpha_corr = np.corrcoef(op_df['accuracy'], op_df['alpha'])[0,1]
    det_corr = np.corrcoef(op_df['accuracy'], op_df['det'])[0,1]
    print(f"{op:15}: α correlation with accuracy = {alpha_corr:.3f}, DET correlation = {det_corr:.3f}")

print("\n--- Layer trajectory stability ---")
final_epoch = df['epoch'].max()
for layer in ['hidden_0', 'hidden_1']:
    layer_df = df[(df['layer'] == layer) & (df['operator'] == 'identity')]
    epoch_vars = layer_df.groupby('epoch')[['alpha', 'det']].var()
    late_var = epoch_vars.loc[20:].mean()
    early_var = epoch_vars.loc[:5].mean()
    print(f"{layer}: early variance = {early_var.mean():.6f}, late variance = {late_var.mean():.6f}")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)

print("\nCASE ANALYSIS:")
for op in ['identity', 'tanh']:
    op_df = df[df['operator'] == op]
    alpha_change = op_df.groupby('epoch')['alpha'].mean().diff().sum()
    det_change = op_df.groupby('epoch')['det'].mean().diff().sum()
    
    if abs(alpha_change) > 0.1 or abs(det_change) > 0.1:
        case = "CASE 1: Φ evolves with accuracy → Φ reflects learning"
    else:
        case = "CASE 2: Φ static across epochs → structural artifact"
    print(f"{op}: α total change = {alpha_change:+.3f}, DET total change = {det_change:+.3f}")
    print(f"  → {case}")

print("\nResults saved to: training_phi_trajectory.csv")
