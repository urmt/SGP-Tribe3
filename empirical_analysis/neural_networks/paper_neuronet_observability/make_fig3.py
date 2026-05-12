import numpy as np
import matplotlib.pyplot as plt
import json

with open('/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase13b_operator_path_scaling/results.json') as f:
    data = json.load(f)

depths = [int(d) for d in data.keys()]
mean_det = [data[str(d)]['mean_abs_delta_DET'] for d in depths]

plt.figure(figsize=(8, 6))
plt.plot(depths, mean_det, 'bo-', linewidth=2, markersize=8, label='Mean |ΔDET|')
plt.xlabel('Operator Composition Depth')
plt.ylabel('Mean |ΔDET|')
plt.title('Geometric Distortion vs Operator Depth')
plt.xticks(depths)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('/home/student/sgp-tribe3/empirical_analysis/neural_networks/paper_neuronet_observability/figure_operator_depth.png', dpi=150)
plt.close()
print("Saved figure_operator_depth.png")