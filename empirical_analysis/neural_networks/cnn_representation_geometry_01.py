#!/usr/bin/env python3
"""
SFH-SGP_CNN_REPRESENTATION_GEOMETRY_01
Test Φ generalization to CNN representations
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

def apply_operator(x, op_name, k=1.0):
    if op_name == 'identity':
        return x
    elif op_name == 'tanh':
        return np.tanh(k * x)
    elif op_name == 'clipped':
        return np.clip(x, -k, k)
    return x

def compute_phi(x, op_name, k=1.0):
    x_op = apply_operator(x, op_name, k)
    if len(x_op) < 100:
        return np.nan, np.nan
    alpha = compute_alpha(x_op, eps_list)
    R = recurrence_matrix(x_op, epsilon)
    det = compute_det(R, l_min)
    return alpha, det

print("Loading MNIST...")
from torchvision import datasets, transforms
transform = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)

X_train = train_ds.data.float().unsqueeze(1).numpy()[:6000] / 255.0
y_train = train_ds.targets.numpy()[:6000]
X_test = test_ds.data.float().unsqueeze(1).numpy()[:2000] / 255.0
y_test = test_ds.targets.numpy()[:2000]
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = None
        self.fc2 = nn.Linear(64, 10)
        self._fc1_init = False
    
    def forward(self, x):
        self.activations = []
        
        x = torch.relu(self.conv1(x))
        self.activations.append(x)
        
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        self.activations.append(x)
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        if not self._fc1_init:
            self.fc1 = nn.Linear(x.size(1), 64)
            self._fc1_init = True
        
        x = torch.relu(self.fc1(x))
        self.activations.append(x)
        
        x = self.fc2(x)
        self.activations.append(x)
        
        return x

print("\nTraining CNN...")
model = CNN()
X_t = torch.FloatTensor(X_train)
y_t = torch.LongTensor(y_train)
X_v = torch.FloatTensor(X_test)
y_v = torch.LongTensor(y_test)

train_ds = TensorDataset(X_t, y_t)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

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
    preds = model(X_v).argmax(dim=1).numpy()
    acc = (preds == y_v.numpy()).mean()
print(f"CNN Accuracy: {acc:.3f}")

def get_layer_activations(model, X):
    model.eval()
    with torch.no_grad():
        _ = model(X)
    return model.activations

acts = get_layer_activations(model, X_v)

layer_names = ['input', 'conv1', 'conv2', 'fc1', 'output']
operators = [('identity', 1.0), ('tanh', 1.0), ('tanh', 2.0), ('clipped', 2.0)]

results = []

print("\n" + "="*60)
print("CNN REPRESENTATION GEOMETRY")
print("="*60)

for layer_idx, act in enumerate(acts):
    act_np = act.numpy()
    
    if act_np.ndim == 4:
        act_flat = act_np.mean(axis=(2, 3))
    elif act_np.ndim == 2:
        act_flat = act_np.mean(axis=1)
    else:
        act_flat = act_np.flatten()
    
    for op_name, k in operators:
        alpha, det = compute_phi(act_flat, op_name, k)
        results.append({
            'layer': layer_names[layer_idx],
            'layer_idx': layer_idx,
            'operator': op_name,
            'k': k,
            'alpha': alpha,
            'det': det
        })
        print(f"{layer_names[layer_idx]:8} | {op_name:10} k={k:.1f} | α={alpha:.3f}, DET={det:.3f}")

df = pd.DataFrame(results)
df.to_csv('cnn_layer_phi.csv', index=False)

print("\n" + "="*60)
print("COMPARISON WITH MLP")
print("="*60)

print("\n--- CNN Layer trajectories (tanh) ---")
tanh_df = df[df['operator'] == 'tanh']
for layer in layer_names:
    layer_df = tanh_df[tanh_df['layer'] == layer]
    if len(layer_df) > 0:
        print(f"{layer}: α={layer_df['alpha'].values[0]:.3f}, DET={layer_df['det'].values[0]:.3f}")

print("\n--- MLP reference (from previous experiment) ---")
print("Layer     | α      | DET")
print("input     | 0.943 | 0.263")
print("hidden_0  | 0.909 | 0.109")
print("hidden_1  | 0.767 | 0.247")
print("output    | 0.998 | 0.041")

print("\n--- CNN vs MLP comparison (tanh, hidden/fc layers) ---")
print("CNN conv1 vs MLP hidden_0:")
print("  CNN: Higher DET likely due to local feature structure")
print("CNN fc1 vs MLP hidden_1:")
print("  Similar trajectory expected")

print("\n--- Operator sensitivity in CNN ---")
print("\nWhich operators maximize layer differentiation?")
for op_name, k in operators:
    op_df = df[(df['operator'] == op_name) & (df['k'] == k)]
    if len(op_df) > 0:
        variance = op_df[['alpha', 'det']].var().mean()
        print(f"{op_name:10} k={k:.1f}: layer variance = {variance:.4f}")

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)

print("\n1) Layer separation:")
for op_name, k in [('tanh', 1.0), ('clipped', 2.0)]:
    op_df = df[(df['operator'] == op_name) & (df['k'] == k)]
    variances = op_df.groupby('layer_idx')[['alpha', 'det']].mean().var()
    print(f"  {op_name}: layer variance = {variances.mean():.4f}")

print("\n2) Trajectory (depth direction):")
tanh_cnn = df[df['operator'] == 'tanh']
for layer in ['conv1', 'conv2', 'fc1']:
    layer_df = tanh_cnn[tanh_cnn['layer'] == layer]
    if len(layer_df) > 0:
        alpha = layer_df['alpha'].values[0]
        det = layer_df['det'].values[0]
        if layer == 'conv1':
            direction = "early"
        elif layer == 'fc1':
            direction = "late"
        else:
            direction = "middle"
        print(f"  {layer} ({direction}): α={alpha:.3f}, DET={det:.3f}")

print("\n3) Operator sensitivity:")
for op_name, k in operators:
    op_df = df[(df['operator'] == op_name) & (df['k'] == k)]
    print(f"  {op_name} k={k}: avg DET = {op_df['det'].mean():.3f}")

print("\n4) CNN vs MLP structural difference:")
conv1_det = df[(df['layer'] == 'conv1') & (df['operator'] == 'tanh')]['det'].values[0]
print(f"  CNN conv1 DET = {conv1_det:.3f}")
print(f"  MLP hidden_0 DET = 0.109")
print(f"  Delta: {conv1_det - 0.109:+.3f}")

print("\nResults saved to: cnn_layer_phi.csv")