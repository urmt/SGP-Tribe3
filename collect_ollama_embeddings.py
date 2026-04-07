"""
Collect Ollama embeddings for all 1022 stimuli in the stimulus bank.
Saves to data/ollama_embeddings_1022.npy
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import json
import numpy as np
import ollama
import time

def main():
    print("Loading stimulus bank...")
    with open('data/stimulus_bank.json', 'r') as f:
        bank = json.load(f)
    
    stimuli = bank['stimuli']
    texts = [s['text'] for s in stimuli]
    print(f"Total stimuli: {len(texts)}")
    
    # Check for existing embeddings
    output_path = 'data/ollama_embeddings_1022.npy'
    checkpoint_path = 'data/ollama_embeddings_1022.checkpoint.npy'
    
    if os.path.exists(output_path):
        print(f"Loading existing embeddings: {output_path}")
        embeddings = np.load(output_path)
        print(f"Loaded: {embeddings.shape}")
        return
    
    # Resume from checkpoint
    embeddings = []
    start_idx = 0
    if os.path.exists(checkpoint_path):
        embeddings = list(np.load(checkpoint_path))
        start_idx = len(embeddings)
        print(f"Resuming from checkpoint: {start_idx}/{len(texts)}")
    
    client = ollama.Client()
    model = "tinyllama"  # 2048d embeddings, much faster than mistral
    batch_size = 10  # Can batch more with tinyllama
    
    print(f"Collecting Ollama embeddings for {len(texts)} texts...")
    start_time = time.time()
    
    for i in range(start_idx, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            response = client.embed(model=model, input=batch, truncate=True)
            batch_embeddings = np.array(response['embeddings'], dtype=np.float32)
            embeddings.extend(batch_embeddings)
            
            elapsed = time.time() - start_time
            rate = len(embeddings) / elapsed
            remaining = (len(texts) - len(embeddings)) / rate
            print(f"  [{len(embeddings)}/{len(texts)}] ({elapsed/60:.1f}min, ETA: {remaining/60:.1f}min)")
            
            # Checkpoint every 100
            if len(embeddings) % 100 == 0:
                np.save(checkpoint_path, np.array(embeddings, dtype=np.float32))
                print(f"  Checkpoint saved: {len(embeddings)} embeddings")
                
        except Exception as e:
            print(f"  ERROR at batch {i}: {e}")
            for _ in batch:
                embeddings.append(np.zeros(4096, dtype=np.float32))
    
    result = np.array(embeddings, dtype=np.float32)
    np.save(output_path, result)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    print(f"\nSaved: {output_path}")
    print(f"Shape: {result.shape}")
    print(f"Mean: {result.mean():.6f}, Std: {result.std():.6f}")

if __name__ == "__main__":
    main()
