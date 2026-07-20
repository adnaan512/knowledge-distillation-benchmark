import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create assets dir if not exists
os.makedirs("assets", exist_ok=True)

# Set style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})

# 1. Accuracy Retained (Bar Chart) - Top 2 Methods
labels = ['Response (full)', 'Feature (full)']
acc = [67.7, 66.8]
retained = [86.0, 84.9]

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(labels))
width = 0.5

bars = ax.bar(x, acc, width, color=['#3498db', '#e74c3c'])
ax.set_ylabel('Accuracy (%)', fontweight='bold')
ax.set_title('Accuracy on CIFAR-10 (2 Epochs)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontweight='bold')
ax.set_ylim(0, 100)

# Add horizontal line for teacher
ax.axhline(y=78.7, color='#2c3e50', linestyle='--', label='Teacher (78.7%)', linewidth=2)

# Annotate bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.annotate(f"{height}%\n({retained[i]}% retained)",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),  # 5 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('assets/accuracy_retained.png', bbox_inches='tight')
plt.close()

# 2. Accuracy vs Compression Ratio (Scatter/Line)
fig, ax = plt.subplots(figsize=(8, 5))

# Plot by method
methods_dict = {
    'Response': {'x': [10.5, 33.6, 57.5], 'y': [67.7, 20.2, 17.8], 'color': '#3498db', 'marker': 'o'},
    'Feature': {'x': [10.5, 33.6], 'y': [66.8, 17.4], 'color': '#e74c3c', 'marker': 's'}
}

for name, data in methods_dict.items():
    ax.plot(data['x'], data['y'], marker=data['marker'], markersize=10, linewidth=2.5, label=f"{name}-based", color=data['color'])

ax.axhline(y=78.7, color='#2c3e50', linestyle='--', label='Teacher (1x, 94MB)', linewidth=2)
ax.set_xlabel('Compression Ratio (Higher is smaller model)', fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontweight='bold')
ax.set_title('Accuracy Degradation vs. Model Compression', fontsize=14, fontweight='bold')
ax.set_ylim(0, 100)
ax.legend(loc='lower left')
plt.tight_layout()
plt.savefig('assets/accuracy_vs_compression.png', bbox_inches='tight')
plt.close()

print("Plots generated successfully in 'assets/' directory.")
