import torch
import numpy as np
import random
import os
import json

seed = 42

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

OUTPUT_DIR = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase10c_training_dynamics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)

X_train = train_ds.data.float().view(-1, 784).numpy()[:2000] / 255.0
y_train = train_ds.targets.numpy()[:2000]
X_test = test_ds.data.float().view(-1, 784).numpy() / 255.0
y_test = test_ds.targets.numpy()

X_t = torch.FloatTensor(X_train)
y_t = torch.LongTensor(y_train)
train_loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=True)

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(784, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 10)
    
    def forward(self, x):
        self.activations = [x]
        x = torch.relu(self.fc1(x))
        self.activations.append(x)
        x = torch.relu(self.fc2(x))
        self.activations.append(x)
        x = self.fc3(x)
        self.activations.append(x)
        return x

model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

print("Training Phase 10C model (20 epochs)...")

for epoch in range(1, 21):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        preds = model(X_t).argmax(dim=1).numpy()
        accuracy = (preds == y_train).mean()
    
    if epoch % 5 == 0:
        print(f"Epoch {epoch}: Train Accuracy = {accuracy:.4f}")

model.eval()

torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model.pt"))
print(f"Saved model.pt")

with torch.no_grad():
    X_full = torch.FloatTensor(X_train)
    _ = model(X_full)
    activations = model.activations[2].numpy()

np.save(os.path.join(OUTPUT_DIR, "activations.npy"), activations)
print(f"Saved activations.npy: shape = {activations.shape}")

model.eval()
with torch.no_grad():
    X_test_t = torch.FloatTensor(X_test)
    test_preds = model(X_test_t).argmax(dim=1).numpy()
    test_accuracy = (test_preds == y_test).mean()

print(f"\nFinal Test Accuracy (full test set = {len(y_test)}): {test_accuracy:.4f}")

if test_accuracy < 0.90:
    print("WARNING: Accuracy < 90%")
else:
    print("\nPHASE 10C REBUILD COMPLETE — ACTIVATIONS READY")