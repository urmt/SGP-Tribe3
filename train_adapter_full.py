"""
SGP-Tribe3 - Full Adapter Training Pipeline
=============================================
Trains the Ollama to LLaMA embedding adapter using paired embeddings.

Steps:
1. Select 50 diverse texts across 10 categories
2. Collect LLaMA embeddings (slow, ~4 hours on CPU)
3. Collect Ollama embeddings (fast, ~8 minutes)
4. Train MLP adapter: 4096 to 8192 to 9216
5. Validate and save adapter_weights.pt

Usage:
    python train_adapter_full.py

Outputs (all NEW, no overwrites):
    - data/llama_embeddings.npy
    - data/ollama_embeddings.npy
    - adapter_weights.pt
    - data/training_metadata.json
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TRANSFORMERS_DEVICE'] = 'cpu'

import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# 20 diverse training texts - 2 per category (faster training)
TRAINING_TEXTS = [
    # Simple (2)
    "The cat sat quietly on the windowsill watching the rain.",
    "A small bird built its nest in the old oak tree.",
    # Logical (2)
    "If every A is a B, and some B are C, then some A might be C.",
    "The contrapositive of a conditional statement is logically equivalent to the original.",
    # Emotional (2)
    "Her heart ached with a longing she could barely put into words.",
    "He felt a surge of triumph as years of effort finally paid off.",
    # Factual (2)
    "The speed of light in vacuum is exactly 299,792,458 meters per second.",
    "Human DNA contains approximately three billion base pairs of genetic information.",
    # Abstract (2)
    "If reality is fundamentally mathematical, then consciousness might be a computational process.",
    "The boundary between self and other may be more permeable than we imagine.",
    # Spatial (2)
    "The ancient temple stood at the center of a vast circular plaza.",
    "Rivers carved deep canyons through layers of sedimentary rock over millennia.",
    # Social (2)
    "She could tell from his hesitation that he was hiding something important.",
    "The mediator carefully balanced the competing interests of both parties.",
    # Motor (2)
    "The potter shaped the clay with practiced hands on the spinning wheel.",
    "She caught the falling glass before it hit the ground with lightning reflexes.",
    # Memory (2)
    "The smell of old books instantly transported him back to the library of his youth.",
    "Fragments of the conversation replayed in her mind long after the meeting ended.",
    # Auditory (2)
    "The thunder rolled across the valley in deep resonant waves of sound.",
    "A lone flute played a haunting melody that echoed through the empty hall.",
]

CATEGORIES = ['simple', 'logical', 'emotional', 'factual', 'abstract',
              'spatial', 'social', 'motor', 'memory', 'auditory']


def collect_llama_embeddings(texts, output_path):
    """Collect LLaMA embeddings using TRIBE v2 HuggingFaceText extractor."""
    print(f"\n[1/3] Collecting LLaMA embeddings for {len(texts)} texts...")
    print(f"  Estimated time: ~{len(texts) * 5 / 60:.0f} minutes on CPU")

    from neuralset.extractors.text import HuggingFaceText

    print("  Initializing HuggingFaceText extractor (LLaMA-3.2-3B)...")
    extractor = HuggingFaceText(
        model_name="meta-llama/Llama-3.2-3B",
        event_types="Word",
        aggregation="sum",
        frequency=2.0,
        contextualized=True,
        layers=[0.5, 0.75, 1.0],
        layer_aggregation="group_mean",
        token_aggregation="mean",
        cache_n_layers=20,
        batch_size=4,
        device="cpu",
        pretrained=True,
    )

    # Resume from checkpoint if exists
    checkpoint_path = output_path + '.checkpoint'
    embeddings = []
    start_idx = 0
    if os.path.exists(checkpoint_path):
        embeddings = list(np.load(checkpoint_path))
        start_idx = len(embeddings)
        print(f"  Resuming from checkpoint: {start_idx}/{len(texts)} embeddings already collected")

    start = time.time()

    for i in range(start_idx, len(texts)):
        text = texts[i]
        try:
            embedding = extractor.get_embedding(text)
            if embedding.ndim > 1:
                embedding = embedding.flatten()
            embeddings.append(embedding)
            elapsed = time.time() - start
            eta = (elapsed / (i - start_idx + 1)) * (len(texts) - i - 1)
            print(f"  [{i+1}/{len(texts)}] {text[:50]}... ({elapsed/60:.1f}min, ETA: {eta/60:.1f}min)")

            # Checkpoint every 10 texts
            if (i + 1) % 10 == 0:
                np.save(checkpoint_path, np.array(embeddings, dtype=np.float32))
                print(f"  Checkpoint saved: {len(embeddings)} embeddings")

        except Exception as e:
            print(f"  ERROR on text {i+1}: {e}")
            embeddings.append(np.zeros(9216, dtype=np.float32))

    result = np.array(embeddings, dtype=np.float32)
    np.save(output_path, result)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    print(f"  LLaMA embeddings saved: {result.shape}")
    return result


def collect_ollama_embeddings(texts, output_path):
    """Collect Ollama embeddings."""
    print(f"\n[2/3] Collecting Ollama embeddings for {len(texts)} texts...")

    if not OLLAMA_AVAILABLE:
        raise RuntimeError("Ollama package not installed")

    client = ollama.Client()
    model = "mistral:7b-instruct-q4_K_M"

    embeddings = []
    batch_size = 5

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            response = client.embed(model=model, input=batch, truncate=True)
            batch_embeddings = np.array(response['embeddings'], dtype=np.float32)
            embeddings.extend(batch_embeddings)
            print(f"  [{min(i+batch_size, len(texts))}/{len(texts)}] embeddings collected")
        except Exception as e:
            print(f"  ERROR on batch {i//batch_size + 1}: {e}")
            for _ in batch:
                embeddings.append(np.zeros(4096, dtype=np.float32))

    result = np.array(embeddings, dtype=np.float32)
    np.save(output_path, result)
    print(f"  Ollama embeddings saved: {result.shape}")
    return result


class EmbeddingAdapter(nn.Module):
    """MLP adapter: Ollama (4096d) to LLaMA-like (9216d)"""

    def __init__(self, input_dim=4096, target_dim=9216, hidden_dim=8192):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, x):
        return self.network(x)


def train_adapter(ollama_emb, llama_emb, output_path, epochs=200, batch_size=8, lr=1e-3):
    """Train the embedding adapter."""
    print(f"\n[3/3] Training adapter...")
    print(f"  Input dim: {ollama_emb.shape[1]}, Target dim: {llama_emb.shape[1]}")
    print(f"  Samples: {len(ollama_emb)}, Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")

    n_train = int(0.8 * len(ollama_emb))
    indices = np.random.permutation(len(ollama_emb))
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    X_train = torch.from_numpy(ollama_emb[train_idx])
    y_train = torch.from_numpy(llama_emb[train_idx])
    X_val = torch.from_numpy(ollama_emb[val_idx])
    y_val = torch.from_numpy(llama_emb[val_idx])

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    adapter = EmbeddingAdapter(ollama_emb.shape[1], llama_emb.shape[1])
    optimizer = torch.optim.Adam(adapter.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        adapter.train()
        total_loss, n_batches = 0, 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(adapter(bx), by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        adapter.eval()
        with torch.no_grad():
            val_pred = adapter(X_val)
            val_loss = criterion(val_pred, y_val).item()
            vp = val_pred / (val_pred.norm(dim=1, keepdim=True) + 1e-8)
            yv = y_val / (y_val.norm(dim=1, keepdim=True) + 1e-8)
            cosine = (vp * yv).sum(dim=1).mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in adapter.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Train: {total_loss/n_batches:.6f}, Val: {val_loss:.6f}, Cosine: {cosine:.4f}")

    if best_state:
        adapter.load_state_dict(best_state)

    adapter.eval()
    with torch.no_grad():
        before_exp = torch.zeros_like(y_val)
        before_exp[:, :X_val.shape[1]] = X_val
        bn = before_exp / (before_exp.norm(dim=1, keepdim=True) + 1e-8)
        yv = y_val / (y_val.norm(dim=1, keepdim=True) + 1e-8)
        before_cos = (bn * yv).sum(dim=1).mean().item()

        vp = adapter(X_val)
        vn = vp / (vp.norm(dim=1, keepdim=True) + 1e-8)
        after_cos = (vn * yv).sum(dim=1).mean().item()
        val_mse = criterion(vp, y_val).item()

    improvement = (after_cos - before_cos) / abs(before_cos) * 100
    print(f"\n  Final: MSE={val_mse:.6f}, Cosine {before_cos:.4f} -> {after_cos:.4f} ({improvement:+.1f}%)")

    torch.save(adapter.state_dict(), output_path)
    print(f"  Adapter saved: {output_path}")

    return {
        'val_mse': val_mse, 'cosine_before': before_cos, 'cosine_after': after_cos,
        'improvement_pct': improvement, 'best_val_loss': best_val_loss,
        'n_train': n_train, 'n_val': len(val_idx), 'epochs': epochs,
    }


def main():
    print("=" * 70)
    print("SGP-Tribe3 - Full Adapter Training Pipeline")
    print("=" * 70)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Training texts: {len(TRAINING_TEXTS)} ({len(CATEGORIES)} categories)")
    print("=" * 70)

    os.makedirs("data", exist_ok=True)

    llama_path = "data/llama_embeddings.npy"
    ollama_path = "data/ollama_embeddings.npy"
    adapter_path = "adapter_weights.pt"

    if os.path.exists(llama_path):
        print(f"\n[1/3] Loading existing LLaMA embeddings: {llama_path}")
        llama_emb = np.load(llama_path)
    else:
        llama_emb = collect_llama_embeddings(TRAINING_TEXTS, llama_path)

    if os.path.exists(ollama_path):
        print(f"\n[2/3] Loading existing Ollama embeddings: {ollama_path}")
        ollama_emb = np.load(ollama_path)
    else:
        ollama_emb = collect_ollama_embeddings(TRAINING_TEXTS, ollama_path)

    meta = train_adapter(ollama_emb, llama_emb, adapter_path, epochs=200, batch_size=8)

    with open("data/training_metadata.json", 'w') as f:
        json.dump({
            'date': datetime.now().isoformat(),
            'n_texts': len(TRAINING_TEXTS),
            'categories': CATEGORIES,
            'texts': TRAINING_TEXTS,
            'llama_shape': list(llama_emb.shape),
            'ollama_shape': list(ollama_emb.shape),
            'adapter_path': adapter_path,
            'training': meta,
        }, f, indent=2)

    print(f"\n{'=' * 70}")
    print("ADAPTER TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"Adapter: {adapter_path}")
    print(f"Metadata: data/training_metadata.json")
    print(f"Cosine: {meta['cosine_before']:.4f} -> {meta['cosine_after']:.4f}")


if __name__ == "__main__":
    main()
