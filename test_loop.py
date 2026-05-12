# Exact replication of the chi_nodes loop structure
import numpy as np

# Mock the compute_dynamical_chi function to return a simple value
def compute_dynamical_chi(traj_3d):
    # Return a simple non-zero value for testing
    return np.linalg.norm(traj_3d, 'fro') * 0.1

# Mock trajectories data (100 timepoints, 9 nodes)
trajectories = np.random.randn(100, 9)
n_nodes = trajectories.shape[1]

print(f"Mock trajectories shape: {trajectories.shape}")
print(f"n_nodes: {n_nodes}")
print(f"range(n_nodes): {list(range(n_nodes))}")

# Exact replication of the chi_nodes loop
chi_nodes = []
for node_idx in range(n_nodes):
    # np.random.seed(42 + node_idx)  # Commented out for deterministic test
    
    # Create 3D trajectory for this node
    base = trajectories[:, node_idx]
    print(f"Node {node_idx}: base.shape = {base.shape}")

    # --- MULTI-TAU AGGREGATION: Scale-averaged dynamical torsion ---
    taus = [1, 2, 3]  # restricted stable regime
    chi_vals = []

    for tau in taus:
        print(f"  Tau {tau}: len(base) = {len(base)}, 3*tau = {3*tau}")
        if len(base) < 3 * tau:
            print(f"    -> SKIP (len < 3*tau)")
            continue
            
        print(f"    -> PROCESS")
        traj_3d = np.column_stack([
            base[:-2*tau],
            base[tau:-tau],
            base[2*tau:]
        ])
        
        chi_tau = compute_dynamical_chi(traj_3d)
        chi_vals.append(chi_tau)
        print(f"    -> chi_tau = {chi_tau:.6f}")

    # Aggregate (robust)
    if chi_vals:
        chi = np.mean(chi_vals)
    else:
        # Fallback (should not happen with normal parameters)
        chi = 0.0
    print(f"Node {node_idx}: chi_vals = {chi_vals}, chi = {chi:.6f}")
    chi_nodes.append(chi)
    print(f"Node {node_idx}: chi_nodes now = {chi_nodes}")

print(f"\nFinal chi_nodes: {chi_nodes}")
chi_nodes = np.array(chi_nodes)
print(f"chi_nodes as array: {chi_nodes}")
print(f"chi_nodes.max(): {chi_nodes.max()}")

if chi_nodes.max() > 0:
    chi_nodes_normalized = chi_nodes / chi_nodes.max()
    print(f"chi_nodes normalized: {chi_nodes_normalized}")
else:
    print("All chi_nodes are zero!")