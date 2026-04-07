"""
Build PCA-informed structured projection matrix using tinyllama embeddings (2048d).

This replaces the random projection with a projection that:
1. Preserves semantic structure found via PCA on Ollama embeddings
2. Maps to fMRI vertex space (20,484 vertices) with category-aware scaling
3. Produces differentiated activations across SGP nodes

Usage:
    python build_structured_projection.py

Output:
    data/structured_projection.npy (2048 x 20484 matrix)
    data/projection_metadata.json
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
from sklearn.decomposition import PCA
import json

def main():
    """Build and save the structured projection matrix."""
    print("=" * 70)
    print("SGP-Tribe3 - Building PCA-Informed Structured Projection")
    print("=" * 70)
    
    # Load the 20 valid tinyllama embeddings
    ollama_path = 'data/ollama_embeddings_tinyllama_20.npy'
    if not os.path.exists(ollama_path):
        print(f"ERROR: {ollama_path} not found")
        print("Run: python collect_tinyllama_embeddings.py first")
        return
    
    ollama_emb = np.load(ollama_path)  # Shape: (20, 2048)
    print(f"Ollama embeddings: {ollama_emb.shape}")
    print(f"Mean: {ollama_emb.mean():.6f}, Std: {ollama_emb.std():.6f}")
    print(f"Non-zero: {np.count_nonzero(ollama_emb)} / {ollama_emb.size}")
    
    # Validate embeddings are not all zeros
    if np.all(ollama_emb == 0):
        print("ERROR: All embeddings are zero. Cannot build projection.")
        return
    
    # Run PCA to find semantic structure in the embeddings
    # With 20 samples, we can extract at most 19 meaningful components
    n_components = min(19, ollama_emb.shape[0] - 1)
    pca = PCA(n_components=n_components)
    pca.fit(ollama_emb)
    
    print(f"\nPCA Analysis:")
    print(f"  Components: {n_components}")
    print(f"  Explained variance ratio:")
    for i, evr in enumerate(pca.explained_variance_ratio_):
        bar = '#' * int(evr * 50)
        print(f"    PC{i+1:2d}: {evr:.4f} {bar}")
    print(f"  Total explained: {pca.explained_variance_ratio_.sum():.4f}")
    
    # Load stimulus bank for category information
    stimulus_bank_path = 'data/stimulus_bank.json'
    if not os.path.exists(stimulus_bank_path):
        print(f"WARNING: {stimulus_bank_path} not found. Skipping category-aware scaling.")
        categories = []
    else:
        with open(stimulus_bank_path, 'r') as f:
            bank = json.load(f)
        categories = [s['category'] for s in bank['stimuli']]
        unique_cats = sorted(set(categories))
        print(f"\nCategories ({len(unique_cats)}): {unique_cats}")
        print(f"Stimuli per category:")
        for cat in unique_cats:
            count = categories.count(cat)
            print(f"  {cat:12s}: {count}")
    
    # Build structured projection matrix
    # Strategy:
    # 1. Use PCA components to create semantically-informed projection
    # 2. Add category-aware scaling for differentiation
    # 3. Use structured random projection for remaining dimensions
    
    input_dim = 2048       # tinyllama embedding dimension
    target_dim = 20484     # fMRI vertex space
    n_pcs = n_components   # Number of PCA components
    
    print(f"\nBuilding projection matrix: ({input_dim} x {target_dim})")
    print(f"  PCA components: {n_pcs}")
    print(f"  Vertices per PC: {target_dim // n_pcs}")
    
    np.random.seed(42)  # Fixed seed for reproducibility
    
    # Initialize projection matrix
    projection = np.zeros((input_dim, target_dim), dtype=np.float32)
    
    # Step 1: PCA-informed projection
    # Each PC direction is projected to a region of fMRI space
    # Scaled by explained variance (more important PCs get stronger projection)
    vertices_per_pc = target_dim // n_pcs
    remaining_vertices = target_dim % n_pcs
    
    for i in range(n_pcs):
        # Number of vertices for this PC
        n_vertices = vertices_per_pc + (1 if i < remaining_vertices else 0)
        
        # Get the PC direction in original embedding space
        pc_direction = pca.components_[i]  # Shape: (2048,)
        
        # Scale by sqrt of explained variance
        scale = np.sqrt(pca.explained_variance_ratio_[i])
        
        # Create random basis for this PC's region
        random_basis = np.random.randn(input_dim, n_vertices).astype(np.float32)
        
        # Project PC direction through random basis, scaled by importance
        pc_projection = pc_direction[:, np.newaxis] * random_basis * scale
        
        # Add to projection matrix
        start_idx = i * vertices_per_pc + min(i, remaining_vertices)
        end_idx = start_idx + n_vertices
        projection[:, start_idx:end_idx] = pc_projection
    
    # Step 2: Category-aware scaling (if categories available)
    if categories:
        unique_cats = sorted(set(categories))
        n_cats = len(unique_cats)
        
        print(f"\nAdding category-aware scaling for {n_cats} categories...")
        
        for cat_idx, cat in enumerate(unique_cats):
            # Find stimuli in this category (from the 20 training texts)
            # Map training text categories
            training_categories = [
                'simple', 'simple',
                'logical', 'logical',
                'emotional', 'emotional',
                'factual', 'factual',
                'abstract', 'abstract',
                'spatial', 'spatial',
                'social', 'social',
                'motor', 'motor',
                'memory', 'memory',
                'auditory', 'auditory'
            ]
            
            cat_indices = [i for i, c in enumerate(training_categories) if c == cat]
            
            if cat_indices:
                # Get mean embedding for this category
                cat_mean = ollama_emb[cat_indices].mean(axis=0)
                
                # Project category mean to fMRI space
                cat_projection = cat_mean[:, np.newaxis] @ projection[cat_indices[0]:cat_indices[0]+1, :]
                
                # Add category-specific bias to enhance differentiation
                region_start = (cat_idx * target_dim) // n_cats
                region_end = ((cat_idx + 1) * target_dim) // n_cats
                cat_bias = np.random.randn(input_dim, region_end - region_start).astype(np.float32) * 0.001
                projection[:, region_start:region_end] += cat_bias
    
    # Step 3: Normalize the projection matrix
    # Normalize each column to unit norm, then scale to reasonable range
    col_norms = np.linalg.norm(projection, axis=0, keepdims=True)
    col_norms = np.maximum(col_norms, 1e-8)  # Avoid division by zero
    projection = projection / col_norms * 0.01
    
    print(f"\nProjection matrix statistics:")
    print(f"  Shape: {projection.shape}")
    print(f"  Mean: {projection.mean():.8f}")
    print(f"  Std: {projection.std():.8f}")
    print(f"  Min: {projection.min():.8f}")
    print(f"  Max: {projection.max():.8f}")
    print(f"  Non-zero: {np.count_nonzero(projection)} / {projection.size} ({np.count_nonzero(projection)/projection.size*100:.1f}%)")
    
    # Save the projection matrix
    output_path = 'data/structured_projection.npy'
    np.save(output_path, projection)
    print(f"\nSaved projection: {output_path}")
    
    # Save metadata for reproducibility and documentation
    metadata = {
        'input_dim': input_dim,
        'target_dim': target_dim,
        'n_pcs': n_pcs,
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'total_explained_variance': float(pca.explained_variance_ratio_.sum()),
        'categories': unique_cats if categories else [],
        'seed': 42,
        'method': 'pca_informed_structured_projection',
        'ollama_model': 'tinyllama',
        'ollama_embedding_dim': 2048,
        'n_training_samples': len(ollama_emb),
        'projection_stats': {
            'mean': float(projection.mean()),
            'std': float(projection.std()),
            'min': float(projection.min()),
            'max': float(projection.max()),
            'non_zero_ratio': float(np.count_nonzero(projection) / projection.size),
        }
    }
    
    metadata_path = 'data/projection_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {metadata_path}")
    
    # Test the projection with the training embeddings
    print(f"\nTesting projection with {len(ollama_emb)} Ollama embeddings...")
    projected = ollama_emb @ projection  # Shape: (20, 20484)
    
    print(f"Projected shape: {projected.shape}")
    print(f"Projected statistics:")
    print(f"  Mean: {projected.mean():.6f}")
    print(f"  Std: {projected.std():.6f}")
    print(f"  Min: {projected.min():.6f}")
    print(f"  Max: {projected.max():.6f}")
    print(f"  Range: {projected.max() - projected.min():.6f}")
    
    # Check category differentiation
    if categories:
        print(f"\nCategory differentiation (mean activation per category):")
        training_categories = [
            'simple', 'simple',
            'logical', 'logical',
            'emotional', 'emotional',
            'factual', 'factual',
            'abstract', 'abstract',
            'spatial', 'spatial',
            'social', 'social',
            'motor', 'motor',
            'memory', 'memory',
            'auditory', 'auditory'
        ]
        
        for cat in unique_cats:
            cat_indices = [i for i, c in enumerate(training_categories) if c == cat]
            if cat_indices:
                cat_projected = projected[cat_indices]
                print(f"  {cat:12s}: mean={cat_projected.mean():.6f}, std={cat_projected.std():.6f}")
    
    print(f"\n{'=' * 70}")
    print("STRUCTURED PROJECTION BUILD COMPLETE")
    print(f"{'=' * 70}")
    print(f"Ready for use in app.py text inference pipeline")

if __name__ == "__main__":
    main()
