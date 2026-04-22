#!/usr/bin/env python3
"""SFH-SGP CNN TEST"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import pairwise_distances
from torchvision import datasets, transforms
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

E = 1e-2

def get_rm(x):
    D = pairwise_distances(x.reshape(-1,1))
    R = (D < E).astype(int)
    np.fill_diagonal(R, 0)
    return R

def get_det(R):
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

def get_rr(R):
    return np.sum(R)/(R.shape[0]**2)

def get_alpha(x):
    es = np.logspace(-3, -1, 6)
    rrs = [get_rr(get_rm(e)) for e in es]
    v = np.isfinite(np.log(rrs))
    return np.polyfit(np.log(es)[v], np.log(rrs)[v], 1)[0] if v.sum()>2 else np.nan

def apply_op(x, n, k=1):
    if n == 'id': return x
    if n == 'tanh': return np.tanh(k*x)
    if n == 'clip': return np.clip(x, -k, k)
    return x

print("Loading...")
tr = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
te = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
Xtr = tr.data[:3000].float() / 255.0
ytr = tr.targets[:3000]
Xte = te.data[:800].float() / 255.0
yte = te.targets[:800]

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 8, 3, padding=1)
        self.p = nn.MaxPool2d(2)
        self.c2 = nn.Conv2d(8, 16, 3, padding=1)
        self.f1 = nn.Linear(16*5*5, 32)
        self.f2 = nn.Linear(32, 10)
        
    def forward(self, x):
        self.acts = []
        self.acts.append(x)
        x = self.p(torch.relu(self.c1(x)))
        self.acts.append(x)
        x = self.p(torch.relu(self.c2(x)))
        self.acts.append(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.f1(x))
        self.acts.append(x)
        x = self.f2(x)
        self.acts.append(x)
        return x

model = CNN()
loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=128, shuffle=True)
opt = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training...")
for ep in range(5):
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
        af = apply_op(a, opn, ok)
        if len(af) < 100:
            continue
        al = get_alpha(af)
        dt = get_det(get_rm(af))
        print(f"{layers[i]:6} | {opn:5} | alpha={al:.3f}, DET={dt:.3f}")