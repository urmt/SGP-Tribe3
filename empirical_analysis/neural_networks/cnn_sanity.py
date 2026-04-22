#!/usr/bin/env python3
"""
SFH-SGP_CNN_MINIMAL_SANITY_TEST_01
Minimal test: does Φ separate CNN layers?
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import pairwise_distances
from torchvision import datasets, transforms

np.random.seed(42)
torch.manual_seed(42)

E = 1e-2

def get_rm(x):
    x = np.asarray(x).flatten()
    D = pairwise_distances(x.reshape(-1,1))
    R = (D < E).astype(int)
    np.fill_diagonal(R, 0)
    return R

def get_det(R):
    N = R.shape[0]
    dc = 0
    for k in range(-N+1, N):
        d = np.diagonal(R, k)
        l = 0
        for v in d:
            if v == 1:
                l += 1
            elif l >= 2:
                dc += l
                l = 0
        if l >= 2:
            dc += l
    return dc / np.sum(R) if np.sum(R) > 0 else 0

def get_rr(R):
    return np.sum(R) / (R.shape[0] ** 2)

def get_alpha(x):
    es = np.logspace(-3, -1, 6)
    rrs = [get_rr(get_rm(e)) for e in es]
    v = np.isfinite(np.log(rrs))
    return np.polyfit(np.log(es)[v], np.log(rrs)[v], 1)[0] if v.sum() > 2 else np.nan

def apply_op(x, op_name):
    x = np.asarray(x).flatten()
    if op_name == 'identity':
        return x
    elif op_name == 'tanh':
        return np.tanh(x)
    return x

print("Loading MNIST (subset)...")
train_ds = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
test_ds = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())

X_train = torch.stack([train_ds[i][0] for i in range(2000)])
y_train = torch.tensor([train_ds[i][1] for i in range(2000)])
X_test = torch.stack([test_ds[i][0] for i in range(500)])
y_test = torch.tensor([test_ds[i][1] for i in range(500)])

class MinimalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(8 * 13 * 13, 10)
        
    def forward(self, x):
        self.activations = []
        self.activations.append(x)
        
        x = self.pool(torch.relu(self.conv(x)))
        self.activations.append(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        self.activations.append(x)
        return x

print("Training (3 epochs)...")
model = MinimalCNN()
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)
opt = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):
    model.train()
    for bx, by in loader:
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model(bx), by)
        loss.backward()
        opt.step()

model.eval()
with torch.no_grad():
    acc = (model(X_test).argmax(1) == y_test).float().mean()
print(f"Accuracy: {acc:.3f}")

print("\n" + "="*50)
print("RESULTS")
print("="*50)

layers = ['input', 'conv1', 'output']
ops = ['identity', 'tanh']

print(f"\n{'Layer':<10} {'Operator':<10} {'alpha':<10} {'DET':<10}")
print("-" * 40)

for i, act in enumerate(model.activations):
    a = act.detach().numpy()
    if a.ndim == 4:
        a_flat = a.mean(axis=(2, 3))
    else:
        a_flat = a
    
    for op in ops:
        af = apply_op(a_flat, op)
        if len(af) < 100:
            continue
        al = get_alpha(af)
        dt = get_det(get_rm(af))
        print(f"{layers[i]:<10} {op:<10} {al:<10.3f} {dt:<10.3f}")

print("\n" + "="*50)
print("ANALYSIS")
print("="*50)

id_vals, tanh_vals = [], []
for i, act in enumerate(model.activations):
    a = act.detach().numpy()
    if a.ndim == 4:
        a_flat = a.mean(axis=(2, 3))
    else:
        a_flat = a
    
    for op, vals in [('identity', id_vals), ('tanh', tanh_vals)]:
        af = apply_op(a_flat, op)
        if len(af) >= 100:
            al = get_alpha(af)
            dt = get_det(get_rm(af))
            vals.append(np.sqrt(al**2 + dt**2))

id_spread = np.std(id_vals) if len(id_vals) > 1 else 0
tanh_spread = np.std(tanh_vals) if len(tanh_vals) > 1 else 0

print(f"\nIdentity Φ spread across layers: {id_spread:.4f}")
print(f"Tanh Φ spread across layers: {tanh_spread:.4f}")

print("\n" + "="*50)
print("VERDICT")
print("="*50)

if tanh_spread > id_spread * 1.5:
    print("\n✓ SUCCESS: tanh shows clear layer separation")
    print("  Φ likely generalizes to CNN representations")
elif tanh_spread > 0.05:
    print("\n~ MODERATE: tanh shows some layer separation")
    print("  Φ may generalize with architecture effects")
else:
    print("\n✗ FAILURE: no clear layer separation with tanh")
    print("  Φ may be architecture-dependent")