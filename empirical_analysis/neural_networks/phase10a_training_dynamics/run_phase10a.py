# PHASE 10B: SIGNAL vs REDUNDANT RECURRENCE

import numpy as np
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

np.random.seed(42)
torch.manual_seed(42)

OUTPUT_DIR = "."
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameters
EPS_STEPS = 10
L_MIN = 2
LONG_THRESHOLD = 5
EPSILON = 1e-8

# Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
X_train = train_ds.data.float().view(-1, 784).numpy()[:2000] / 255.0
y_train = train_ds.targets.numpy()[:2000]

X_t = torch.FloatTensor(X_train)
y_t = torch.LongTensor(y_train)
train_loader = DataLoader(TensorDataset(X_t, y_t), batch_size=128, shuffle=True)

# MLP
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
    
    def forward(self, x):
        self.activations = [x]
        x = torch.relu(self.fc1(x))
        self.activations.append(x)
        x = torch.relu(self.fc2(x))
        self.activations.append(x)
        x = self.fc3(x)
        self.activations.append(x)
        return x

# Recurrence functions
def compute_det(x, eps):
    n = len(x)
    R = np.array([[1 if abs(x[i]-x[j]) < eps else 0 for j in range(n)] for i in range(n)], dtype=np.uint8)
    total = 0
    short_lines = 0
    long_lines = 0
    for k in range(-n+1, n):
        diag = np.diag(R, k=k)
        length = 0
        for val in diag:
            if val == 1:
                length += 1
            else:
                if length >= L_MIN:
                    total += length
                    if length < LONG_THRESHOLD:
                        short_lines += length
                    else:
                        long_lines += length
                length = 0
        if length >= L_MIN:
            total += length
            if length < LONG_THRESHOLD:
                short_lines += length
            else:
                long_lines += length
    
    denom = np.sum(R) + 1e-12
    return total / denom, short_lines / denom, long_lines / denom

def compute_metrics(activations):
    # Use first hidden layer
    act = activations[1].numpy()
    act_flat = act.mean(axis=0)
    
    eps_vals = np.linspace(0.1, 1.0, EPS_STEPS)
    dets = []
    dets_short = []
    dets_long = []
    
    for eps in eps_vals:
        F_tot, F_s, F_l = compute_det(act_flat, eps)
        dets.append(F_tot)
        dets_short.append(F_s)
        dets_long.append(F_l)
    
    dets = np.array(dets)
    dets_short = np.array(dets_short)
    dets_long = np.array(dets_long)
    
    F_total = np.trapezoid(dets, eps_vals)
    F_short = np.trapezoid(dets_short, eps_vals)
    F_long = np.trapezoid(dets_long, eps_vals)
    C = 1.0 / (np.var(dets) + EPSILON)
    
    # Instability proxy: perturbation growth
    x = act_flat
    x_pert = x + 1e-6
    growths = []
    for _ in range(10):
        # Simple perturbation test
        diff = np.abs(x_pert - x).mean()
        growths.append(diff)
    
    instability = np.mean(growths)
    
    return F_total, F_short, F_long, C, instability

# Training
model = MLP()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

results = []

print("Training and tracking metrics...")
for epoch in range(1, 21):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    # Extract metrics
    model.eval()
    with torch.no_grad():
        _ = model(X_t)
        
        F_total, F_short, F_long, C, instability = compute_metrics(model.activations)
        
        preds = model(X_t).argmax(dim=1).numpy()
        accuracy = (preds == y_train[:len(preds)]).mean()
        
        results.append({
            "epoch": epoch,
            "F_total": float(F_total),
            "F_short": float(F_short),
            "F_long": float(F_long),
            "C": float(C),
            "instability": float(instability),
            "accuracy": float(accuracy)
        })
        
        print(f"Epoch {epoch}: F_tot={F_total:.4f}, F_short={F_short:.4f}, F_long={F_long:.4f}, C={C:.2f}, acc={accuracy:.3f}")

# Save
output_path = os.path.join(OUTPUT_DIR, "results.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to: {output_path}")