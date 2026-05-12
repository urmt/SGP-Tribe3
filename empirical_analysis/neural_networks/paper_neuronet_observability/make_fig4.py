import numpy as np
import matplotlib.pyplot as plt
import json

with open('/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase14_architecture_invariance/results.json') as f:
    data = json.load(f)

trained = data['trained']
random_data = data['random']

depths = [int(d) for d in trained.keys()]
trained_det = [trained[str(d)]['mean_abs_delta_DET'] for d in depths]
random_det = [random_data[str(d)]['mean_abs_delta_DET'] for d in depths]

plt.figure(figsize=(8, 6))
plt.plot(depths, trained_det, 'bo-', linewidth=2, markersize=8, label='Trained')
plt.plot(depths, random_det, 'rs--', linewidth=2, markersize=8, label='Random')
plt.xlabel('Operator Composition Depth')
plt.ylabel('Mean |ΔDET|')
plt.title('Trained vs Random Network Geometry')
plt.xticks(depths)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('/home/student/sgp-tribe3/empirical_analysis/neural_networks/paper_neuronet_observability/figure_trained_vs_random.png', dpi=150)
plt.close()
print("Saved figure_trained_vs_random.png")