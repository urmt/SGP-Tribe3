#!/usr/bin/env python3
"""SFH-SGP CNN Minimal Test"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import pairwise_distances
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

E = 1e-2

def rm(x):
    D = pairwise_distances(x.reshape(-1,1))
    R = (D < E).astype(int)
    np.fill_diagonal(R, 0)
    return R

def det(R):
    N = R.shape[0]
    dc = 0
    for k in range(-N+1,N):
        d = np.diagonal(R,k)
        l = 0
        for v in d:
            if v==1: l+=1
            elif l>=2: dc+=l; l=0
        if l>=2: dc+=l
    return dc/np.sum(R) if np.sum(R)>0 else 0

def rr(R):
    return np.sum(R)/(R.shape[0]**2)

def alpha(x):
    es = np.logspace(-3, -1, 6)
    rrs = [rr(rm(e)) for e in es]
    valid = np.isfinite(np.log(rrs))
    return np.polyfit(np.log(es)[valid], np.log(rrs)[valid], 1)[0] if valid.sum()>2 else np.nan

def op(x, n, k=1):
    if n == 'id': return x
    if n == 'tanh': return np.tanh(k*x)
    if n == 'clip': return np.clip(x, -k, k)
    return x

print("Loading data...")
from torchvision import datasets
tr = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
te = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
Xtr = tr.data.float()[:3000] / 255.0
ytr = tr.targets.numpy()[:3000]
Xte = te.data.float()[:800] / 255.0
yte = te.targets.numpy()[:800]

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc1 = nn.Linear(16*5*5, 32)
        self.fc2 = nn.Linear(32, 10)
        
    def forward(self, x):
        self.acts = []
        self.acts.append(x)
        x = self.pool(torch.relu(self.conv1(x)))
        self.acts.append(x)
        x = self.pool(torch.relu(self.conv2(x)))
        self.acts.append(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        self.acts.append(x)
        x = self.fc2(x)
        self.acts.append(x)
        return x

model = CNN()
loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=128, shuffle=True)
opt = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training...")
for epoch in range(5):
    model.train()
    for bx, by in loader:
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model(bx), by)
        loss.backward()
        opt.step()

model.eval()
with torch.no_grad():
    acc = (model(Xte).argmax(1) == yte).float().mean()
print(f"Accuracy: {acc:.3f}")

layers = ['in', 'conv1', 'conv2', 'fc', 'out']
ops = [('id', 1), ('tanh', 1), ('clip', 2)]

print("\nResults:")
for i, act in enumerate(model.acts):
    a = act.numpy()
    if a.ndim == 4:
        a = a.mean(axis=(2,3))
    else:
        a = a.mean(axis=1)
    for opn, ok in ops:
        af = op(a, opn, ok)
        if len(af) < 100:
            continue
        al = alpha(af)
        dt = det(rm(af))
        print(f"{layers[i]:6} | {opn:5} | alpha={al:.3f}, DET={dt:.3f}")