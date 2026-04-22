#!/usr/bin/env python3
"""
SFH-SGP_NEURAL_REPRESENTATION_GEOMETRY_01
Apply Φ(S, O) to neural network activations
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

epsilon = 1e-2
l_min = 2
n_total = 2000
n_burn = 500

eps_list = np.logspace(-4, -1, 12)

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
    return np.sum(R) / (N * N - N)

def compute_alpha(x, eps_list):
    R_vals = []
    for eps in eps_list:
        R = recurrence_matrix(x, eps)
        R_vals.append(recurrence_rate(R))
    
    log_eps = np.log(eps_list)
    log_R = np.log(R_vals)
    
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
    elif operator == 'sin':
        return np.sin(x)
    elif operator == 'linear_mix':
        return 0.5 * x + 0.5 * np.sin(x)
    return x

def compute_phi(x, operator):
    x_op = apply_operator(x, operator)
    x_op = x_op[n_burn:n_burn + n_total]
    
    if len(x_op) < 100:
        return np.nan, np.nan
    
    alpha = compute_alpha(x_op, eps_list)
    R = recurrence_matrix(x_op, epsilon)
    det = compute_det(R, l_min)
    
    return alpha, det

print("Loading MNIST...")
try:
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    
    X_train = train_ds.data.float().view(-1, 784).numpy() / 255.0
    y_train = train_ds.targets.numpy()
    X_test = test_ds.data.float().view(-1, 784).numpy() / 255.0
    y_test = test_ds.targets.numpy()
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
except Exception as e:
    print(f"MNIST load failed: {e}")
    print("Using synthetic data...")
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=6000, n_features=784, n_informative=100, n_classes=10, random_state=42)
    X_train, X_test = X[:5000], X[5000:]
    y_train, y_test = y[:5000], y[5000:]

class MLP(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activations = []
        
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

def train_model(model, X_train, y_train, X_test, y_test, epochs=30):
    X_t = torch.FloatTensor(X_train)
    y_t = torch.LongTensor(y_train)
    X_v = torch.FloatTensor(X_test)
    y_v = torch.LongTensor(y_test)
    
    train_ds = TensorDataset(X_t, y_t)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    for epoch in range(epochs):
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
        
        best_acc = max(best_acc, acc)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: acc={acc:.3f}")
    
    return best_acc

models_config = {
    'MLP_SMALL': [784, 32, 10],
    'MLP_MEDIUM': [784, 64, 32, 10],
    'MLP_DEEP': [784, 128, 64, 32, 16, 10]
}

operators = ['identity', 'normalization', 'tanh', 'sin', 'linear_mix']

results = []
print("\n" + "="*60)
print("TRAINING MODELS")
print("="*60)

for model_name, layers in models_config.items():
    print(f"\n{model_name}: {layers}")
    model = MLP(layers)
    acc = train_model(model, X_train, y_train, X_test, y_test)
    print(f"  Final accuracy: {acc:.3f}")
    
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test)
        _ = model(X_tensor)
    
    layer_names = ['input'] + [f'hidden_{i}' for i in range(len(layers)-2)] + ['output']
    
    for layer_idx, act in enumerate(model.activations):
        act_np = act.numpy()
        n_samples = act_np.shape[0]
        
        if act_np.ndim > 1:
            act_flat = act_np.mean(axis=1)
        else:
            act_flat = act_np
        
        for op in operators:
            alpha, det = compute_phi(act_flat, op)
            
            results.append({
                'model': model_name,
                'layer': layer_names[layer_idx],
                'layer_idx': layer_idx,
                'operator': op,
                'alpha': alpha,
                'det': det,
                'accuracy': acc
            })
            
            print(f"  {layer_names[layer_idx]:10} | {op:15} | α={alpha:.3f}, DET={det:.3f}")

df = pd.DataFrame(results)
df.to_csv('layer_phi_values.csv', index=False)

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)

df_model = df.groupby(['model', 'layer', 'operator']).agg({
    'alpha': 'mean',
    'det': 'mean'
}).reset_index()
df_model.to_csv('model_comparison.csv', index=False)

print("\n--- Layer clustering in (α, DET) space ---")
for model_name in models_config.keys():
    model_df = df[df['model'] == model_name]
    print(f"\n{model_name}:")
    for layer in model_df['layer'].unique():
        layer_df = model_df[model_df['layer'] == layer]
        avg_alpha = layer_df['alpha'].mean()
        avg_det = layer_df['det'].mean()
        print(f"  {layer}: α={avg_alpha:.3f}, DET={avg_det:.3f}")

print("\n--- Systematic movement across layers ---")
for model_name in models_config.keys():
    model_df = df[df['model'] == model_name].copy()
    model_df = model_df.sort_values('layer_idx')
    alpha_gradient = np.gradient(model_df.groupby('layer_idx')['alpha'].mean().values)
    det_gradient = np.gradient(model_df.groupby('layer_idx')['det'].mean().values)
    print(f"{model_name}: α gradient mean={np.mean(alpha_gradient):.4f}, DET gradient mean={np.mean(det_gradient):.4f}")

print("\n--- Operator effect on geometry ---")
for op in operators:
    op_df = df[df['operator'] == op]
    print(f"{op}: avg α={op_df['alpha'].mean():.3f}, avg DET={op_df['det'].mean():.3f}")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)

layer_variance = df.groupby('layer')[['alpha', 'det']].var().mean()
print(f"\n1. Layer clustering strength (variance): α={layer_variance['alpha']:.4f}, DET={layer_variance['det']:.4f}")

for model_name in models_config.keys():
    model_df = df[df['model'] == model_name]
    first_layer = model_df[model_df['layer_idx'] == 0]['alpha'].mean()
    last_layer = model_df[model_df['layer'] == 'output']['alpha'].mean()
    print(f"2. {model_name} depth trajectory: input α={first_layer:.3f} → output α={last_layer:.3f}")

op_shift = df.groupby('operator')['alpha'].mean().std()
print(f"3. Operator geometry shift (α std across operators): {op_shift:.4f}")

print("\nResults saved to: layer_phi_values.csv, model_comparison.csv")
