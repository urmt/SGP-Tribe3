#!/usr/bin/env python3
"""
SFH-SGP_CNN_REPRESENTATION_GEOMETRY_01 - FAST VERSION
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
    return diag_counts / total_rec if total_rec > 0 else 0

def recurrence_rate(R):
    N = R.shape[0]
    return np.sum(R) / (N * N - N) if N > 1 else 0

def compute_alpha(x, eps_list):
    R_vals = [recurrence_rate(recurrence_matrix(x, eps)) for eps in eps_list]
    log_R = np.log(np.array(R_vals) + 1e-10)
    valid = np.isfinite(log_R)
    if valid.sum() > 2:
        return np.polyfit(np.log(eps_list)[valid], log_R[valid], 1)[0]
    return np.nan

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
    return compute_alpha(x_op, eps_list), compute_det(recurrence_matrix(x_op, epsilon), l_min)

print("Loading MNIST...")
from torchvision import datasets, transforms
train_ds = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
test_ds = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())

X_train = train_ds.data.float()[:4000].unsqueeze(1).numpy() / 255.0
y_train = train_ds.targets.numpy()[:4000]
X_test = test_ds.data.float()[:1000].unsqueeze(1).numpy() / 255.0
y_test = test_ds.targets.numpy()[:1000]

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

model = CNN()
X_t, y_t = torch.FloatTensor(X_train), torch.LongTensor(y_train)
X_v, y_v = torch.FloatTensor(X_test), torch.LongTensor(y_test)

train_loader = DataLoader(TensorDataset(X_t, y_t), batch_size=256, shuffle=True)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training CNN...")
for epoch in range(8):
    model.train()
    for bx, by in train_loader:
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(model(bx), by)
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    acc = (model(X_v).argmax(1).numpy() == y_v.numpy()).mean()
print(f"Accuracy: {acc:.3f}")

model.eval()
with torch.no_grad():
    acts = model.activations

layer_names = ['input', 'conv1', 'conv2', 'fc1', 'output']
operators = [('identity', 1.0), ('tanh', 1.0), ('clipped', 2.0)]
results = []

for li, act in enumerate(acts):
    an = act.numpy()
    af = an.mean(axis=(2, 3)) if an.ndim == 4 else an.mean(axis=1)
    for op, k in operators:
        alpha, det = compute_phi(af, op, k)
        results.append({'layer': layer_names[li], 'operator': op, 'alpha': alpha, 'det': det})
        print(f"{layer_names[li]:8} | {op:10} | α={alpha:.3f}, DET={det:.3f}")

df = pd.DataFrame(results)
df.to_csv('cnn_layer_phi.csv', index=False)

print("\n--- CNN Layer Trajectories ---")
tanh_df = df[df['operator'] == 'tanh']
for _, row in tanh_df.iterrows():
    print(f"{row['layer']}: α={row['alpha']:.3f}, DET={row['det']:.3f}")

print("\n--- Comparison with MLP ---")
print("CNN fc1 vs MLP hidden_1 (similar depth): Check DET magnitude")

conv1_det = tanh_df[tanh_df['layer'] == 'conv1']['det'].values[0]
print(f"\nCNN conv1 DET: {conv1_det:.3f}")
print("MLP hidden_0 DET: 0.109")
print(f"Difference: {conv1_det - 0.109:+.3f}")

print("\nResults saved: cnn_layer_phi.csv")