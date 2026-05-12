import numpy as np

# Simulate what happens in the loop
print("Debugging chi_nodes aggregation...")

# Simulate the node loop
chi_nodes = []
n_nodes = 3

for node_idx in range(n_nodes):
    # Simulate base trajectory
    base = np.random.randn(100)  # length 100
    
    # Multi-tau aggregation
    taus = [1, 2, 3]  # restricted stable regime
    chi_vals = []
    
    for tau in taus:
        if len(base) < 3 * tau:
            continue
            
        traj_3d = np.column_stack([
            base[:-2*tau],
            base[tau:-tau],
            base[2*tau:]
        ])
        
        # Simple chi computation for testing (just return norm of traj_3d for simplicity)
        chi_tau = np.linalg.norm(traj_3d, 'fro') / 1000.0  # scale down
        chi_vals.append(chi_tau)
        print(f"  Node {node_idx}, tau={tau}: chi_tau={chi_tau:.6f}")
    
    # Aggregate (robust)
    if chi_vals:
        chi = np.mean(chi_vals)
    else:
        # Fallback (should not happen with normal parameters)
        chi = 0.0
    print(f"  Node {node_idx}: chi_vals={chi_vals}, chi={chi:.6f}")
    chi_nodes.append(chi)

print(f"Final chi_nodes: {chi_nodes}")
chi_nodes = np.array(chi_nodes)
print(f"chi_nodes as array: {chi_nodes}")
print(f"chi_nodes.max(): {chi_nodes.max()}")

if chi_nodes.max() > 0:
    chi_nodes_normalized = chi_nodes / chi_nodes.max()
    print(f"chi_nodes normalized: {chi_nodes_normalized}")
else:
    print("All chi_nodes are zero!")