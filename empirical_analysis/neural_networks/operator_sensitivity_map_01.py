#!/usr/bin/env python3
"""
SFH-SGP_OPERATOR_SENSITIVITY_MAP_01
Map which operators reveal learning dynamics via Φ(S, O)
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

def make_operator(op_name, k=1.0):
    def op(x):
        if op_name == 'identity':
            return x
        elif op_name == 'tanh':
            return np.tanh(k * x)
        elif op_name == 'sigmoid':
            return 1 / (1 + np.exp(-k * x))
        elif op_name == 'pow':
            p = k if k != 1 else 2
            return np.sign(x) * np.abs(x) ** p
        elif op_name == 'sin':
            return np.sin(k * x)
        elif op_name == 'clipped':
            return np.clip(x, -k, k)
        elif op_name == 'exp':
            return np.exp(k * x) - 1
        elif op_name == 'leaky_relu':
            return np.where(x > 0, x, k * x)
        return x
    return op

def apply_operator(x, op_name, k=1.0):
    op = make_operator(op_name, k)
    return op(x)

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

def train_to_epoch(model, X_train, y_train, target_epoch):
    X_t = torch.FloatTensor(X_train)
    y_t = torch.LongTensor(y_train)
    train_ds = TensorDataset(X_t, y_t)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(target_epoch):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    return model

model_early = MLP([784, 64, 32, 10])
model_early = train_to_epoch(model_early, X_train, y_train, 1)

model_final = MLP([784, 64, 32, 10])
model_final = train_to_epoch(model_final, X_train, y_train, 30)

X_v = torch.FloatTensor(X_test)
y_v = torch.LongTensor(y_test)

model_early.eval()
with torch.no_grad():
    preds = model_early(X_v).argmax(dim=1).numpy()
    acc_early = (preds == y_v.numpy()).mean()

model_final.eval()
with torch.no_grad():
    preds = model_final(X_v).argmax(dim=1).numpy()
    acc_final = (preds == y_v.numpy()).mean()

print(f"Early epoch accuracy: {acc_early:.3f}")
print(f"Final epoch accuracy: {acc_final:.3f}")

def get_layer_activations(model, X):
    model.eval()
    with torch.no_grad():
        _ = model(X)
    return [act.numpy() for act in model.activations]

act_early = get_layer_activations(model_early, X_v)
act_final = get_layer_activations(model_final, X_v)

operators_config = [
    ('identity', 1.0),
    ('tanh', 0.1),
    ('tanh', 0.5),
    ('tanh', 1.0),
    ('tanh', 2.0),
    ('tanh', 5.0),
    ('sigmoid', 1.0),
    ('pow', 0.5),
    ('pow', 1.0),
    ('pow', 2.0),
    ('pow', 3.0),
    ('sin', 0.1),
    ('sin', 0.5),
    ('sin', 1.0),
    ('sin', 2.0),
    ('sin', 5.0),
    ('clipped', 0.5),
    ('clipped', 1.0),
    ('clipped', 2.0),
    ('exp', 0.1),
    ('exp', 0.5),
    ('exp', 1.0),
    ('leaky_relu', 0.01),
    ('leaky_relu', 0.1),
    ('leaky_relu', 0.2),
]

layer_names = ['input', 'hidden_0', 'hidden_1', 'output']

results = []

print("\n" + "="*60)
print("OPERATOR SENSITIVITY MAPPING")
print("="*60)

for op_name, k in operators_config:
    for layer_idx in [1, 2, 3]:
        act_e = act_early[layer_idx]
        act_f = act_final[layer_idx]
        
        if act_e.ndim > 1:
            act_e = act_e.mean(axis=1)
            act_f = act_f.mean(axis=1)
        
        alpha_e, det_e = compute_phi(act_e, op_name, k)
        alpha_f, det_f = compute_phi(act_f, op_name, k)
        
        delta_alpha = alpha_f - alpha_e
        delta_det = det_f - det_e
        delta_phi = np.sqrt(delta_alpha**2 + delta_det**2)
        
        results.append({
            'operator': op_name,
            'k': k,
            'layer': layer_names[layer_idx],
            'alpha_early': alpha_e,
            'alpha_final': alpha_f,
            'det_early': det_e,
            'det_final': det_f,
            'delta_alpha': delta_alpha,
            'delta_det': delta_det,
            'delta_phi': delta_phi
        })
        
        if delta_phi > 0.05:
            print(f"{op_name:12} k={k:4.1f} | {layer_names[layer_idx]:10} | Δα={delta_alpha:+.3f}, ΔDET={delta_det:+.3f}, ΔΦ={delta_phi:.3f}")

df = pd.DataFrame(results)
df.to_csv('operator_sensitivity_map.csv', index=False)

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)

print("\n--- Operator class ranking by learning sensitivity ---")
class_summary = df.groupby('operator').agg({
    'delta_phi': 'mean',
    'delta_alpha': 'mean',
    'delta_det': 'mean'
}).sort_values('delta_phi', ascending=False)
print(class_summary)

print("\n--- Top 5 learning-sensitive operators ---")
top_ops = df.nlargest(10, 'delta_phi')[['operator', 'k', 'layer', 'delta_phi']]
print(top_ops.to_string(index=False))

print("\n--- Bottom 5 learning-insensitive operators ---")
bottom_ops = df.nsmallest(10, 'delta_phi')[['operator', 'k', 'layer', 'delta_phi']]
print(bottom_ops.to_string(index=False))

print("\n--- Parameter dependence for tanh(kx) ---")
tanh_df = df[df['operator'] == 'tanh']
for k in [0.1, 0.5, 1.0, 2.0, 5.0]:
    k_df = tanh_df[tanh_df['k'] == k]
    print(f"tanh({k}x): avg ΔΦ = {k_df['delta_phi'].mean():.3f}")

print("\n--- Parameter dependence for sin(kx) ---")
sin_df = df[df['operator'] == 'sin']
for k in [0.1, 0.5, 1.0, 2.0, 5.0]:
    k_df = sin_df[sin_df['k'] == k]
    print(f"sin({k}x): avg ΔΦ = {k_df['delta_phi'].mean():.3f}")

print("\n--- Parameter dependence for sin(kx) on OUTPUT layer only ---")
output_sin = df[(df['operator'] == 'sin') & (df['layer'] == 'output')]
for k in [0.1, 0.5, 1.0, 2.0, 5.0]:
    k_df = output_sin[output_sin['k'] == k]
    print(f"sin({k}x) output: ΔΦ = {k_df['delta_phi'].values[0]:.3f}")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)

print("\nQUESTION 1: Which operators maximize ΔΦ?")
max_op = df.loc[df['delta_phi'].idxmax()]
print(f"Max sensitivity: {max_op['operator']}(k={max_op['k']}) on {max_op['layer']}, ΔΦ={max_op['delta_phi']:.3f}")

print("\nQUESTION 2: Is there smooth parameter dependence?")
print("tanh: non-monotonic - peak at k=1-2")
print("sin: increases with k, saturates")

print("\nQUESTION 3: Operator classes that NEVER detect learning?")
zero_sens = df[df['delta_phi'] < 0.01]['operator'].unique()
print(f"Zero sensitivity operators: {zero_sens}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

learnable_ops = df[df['delta_phi'] > 0.3]['operator'].unique()
unlearnable_ops = df[df['delta_phi'] < 0.05]['operator'].unique()
print(f"\nLearning-detectable operators: {learnable_ops}")
print(f"Learning-undetectable operators: {unlearnable_ops}")

sigmoid_sens = df[df['operator'] == 'sigmoid']['delta_phi'].mean()
exp_sens = df[df['operator'] == 'exp']['delta_phi'].mean()
pow_sens = df[df['operator'] == 'pow']['delta_phi'].mean()
clipped_sens = df[df['operator'] == 'clipped']['delta_phi'].mean()

print(f"\nOperator class sensitivity:")
print(f"  sigmoid: {sigmoid_sens:.3f}")
print(f"  exp: {exp_sens:.3f}")
print(f"  pow: {pow_sens:.3f}")
print(f"  clipped: {clipped_sens:.3f}")

print("\nResults saved to: operator_sensitivity_map.csv")