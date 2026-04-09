import matplotlib.pyplot as plt

B = [4, 5, 6, 7, 8, 9, 10, 11, 12]
AUC = [0.9249, 0.9242, 0.9621, 0.9256, 0.9671, 0.9478, 0.9384, 0.9328, 0.9298]

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(B, AUC, marker='D', linestyle='-', color='#1f77b4', linewidth=2, markersize=6)
ax.axvline(x=8, color='#d62728', linestyle='--', alpha=0.8, linewidth=1.5, label='B=8 Optima')
ax.axvline(x=6, color='#ff7f0e', linestyle=':', alpha=0.8, linewidth=1.5, label='B=6 Local Optima')

ax.set_xlabel('Bottleneck Width ($B$)', fontsize=11)
ax.set_ylabel('Score-A AUC', fontsize=11)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('ablation_bottleneck.pdf', format='pdf', bbox_inches='tight')
print("Plot saved to ablation_bottleneck.pdf")
