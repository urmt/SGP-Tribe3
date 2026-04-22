#!/usr/bin/env python3
"""SFH-SGP_CNN_MINIMAL"""
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import pairwise_distances

np.random.seed(42)
torch.manual_seed(42)

eps = 1e-2

def rm(x):
    x = x.reshape(-1,1)
    return (pairwise_distances(x) < eps).astype(int)

def det(R):
    N = R.shape[0]
    dc = tr = 0.0
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
    es = np.logspace(-3,-1,6)
    rrs = [rr(rm(e)) for e in es]
    valid = np.isfinite(np.log(rrs))
    return np.polyfit(np.log(es)[valid], np.log(rrs)[valid], 1)[0] if valid.sum()>2 else np.nan

def op(x,n,k=1):
    if n=='id': return x
    if n=='tanh': return np.tanh(k*x)
    if n=='clip': return np.clip(x,-k,k)
    return x

print("Load...")
from torchvision import datasets
tr = datasets.MNIST('./data',train=True,download=True,transform=lambda x:torch.tensor(x.float().unsqueeze(0)/255))
te = datasets.MNIST('./data',train=False,download=True,transform=lambda x:torch.tensor(x.float().unsqueeze(0)/255))
Xtr,ytr = tr.data.float()[:3000]/255, tr.targets.numpy()[:3000]
Xte,yte = te.data.float()[:800]/255, te.targets.numpy()[:800]

class M(nn.Module):
    def __init__(super__=None):
        super().__init__()
        self.c1=nn.Conv2d(1,8,3,padding=1)
        self.p=nn.MaxPool2d(2)
        self.c2=nn.Conv2d(1,16,3,padding=1)
        self.f1=self.f2=None
        
    def forward(s,x):
        s.a=[x]
        x=s.p(torch.relu(s.c1(x))); s.a+=[x]
        x=s.p(torch.relu(s.c2(x))); s.a+=[x]
        x=x.view(x.size(0),-1)
        if s.f1 is None: s.f1=nn.Linear(x.size(1),32); s.f2=nn.Linear(32,10)
        x=torch.relu(s.f1(x)); s.a+=[x]
        x=s.f2(x); s.a+=[x]
        return x

m=M()
dl=DataLoader(TensorDataset(Xtr,ytr),batch_size=128,shuffle=True)
o=nn.optim.Adam(m.parameters(),lr=.001)
for e in range(5):
    for bx,by in dl: o.zero_grad(); loss=nn.CrossEntropyLoss()(m(bx),by); loss.backward(); o.step()

m.eval(); print(f"Acc: {(m(Xte).argmax(1)==yte).float().mean():.3f}")

ls=['in','conv1','conv2','fc','out']
for i,a in enumerate(m.a):
    an=a.numpy()
    if an.ndim==4: an=an.mean(axis=(2,3))
    else: an=an.mean(axis=1)
    for on,ok in [('id',1),('tanh',1),('clip',2)]:
        af=op(an,on,ok)
        if len(af)<100: continue
        a_,d_ = alpha(af), det(rm(af))
        print(f"{ls[i]:6}|{on:5}|{a_:.3f},{d_:.3f}")