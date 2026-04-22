import matplotlib.pyplot as plt

layers = ["input", "hidden_0", "hidden_1", "output"]

alpha_identity = [0.99, 0.98, 0.98, 0.99]
det_identity   = [0.25, 0.22, 0.20, 0.02]

alpha_tanh = [0.93, 0.80, 0.14, 0.03]
det_tanh   = [0.27, 0.41, 0.98, 1.00]

plt.figure(figsize=(6,5))

plt.plot(alpha_identity, det_identity, 'o--', label='identity', markersize=10, linewidth=1.5)
plt.plot(alpha_tanh, det_tanh, 's--', label='tanh', markersize=10, linewidth=1.5)

for i, label in enumerate(layers):
    plt.text(alpha_identity[i] + 0.015, det_identity[i] + 0.02, label, fontsize=9)
    plt.text(alpha_tanh[i] + 0.015, det_tanh[i] + 0.02, label, fontsize=9)

plt.xlabel(r'$\alpha$ (Recurrence Scaling)', fontsize=12)
plt.ylabel('DET (Determinism)', fontsize=12)
plt.title(r'Operator-Dependent Geometry in $(\alpha, DET)$ Space', fontsize=13)

plt.xlim(0.0, 1.05)
plt.ylim(0.0, 1.05)

plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figure1_operator_geometry.png", dpi=300)
plt.close()
print("Figure 1 saved")