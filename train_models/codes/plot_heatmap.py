import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

csv_path = r'C:\fyp basic models\results\csv\durian_yield_mapping.csv'
df = pd.read_csv(csv_path).dropna(subset=['Fruit_ID'])
fig, ax = plt.subplots(figsize=(12, 5))
sns.set_theme(style="whitegrid")

x_grid = np.linspace(-1, 11, 400)
y_grid = np.linspace(0, 4, 200)
X, Y = np.meshgrid(x_grid, y_grid)
Z = np.zeros_like(X)

sigma_x = 0.6  
sigma_y = 0.4  

for i, row in df.iterrows():
    cx, cy = row['Map_X'], row['Map_Y']
    conf = row['Confidence']
    exponent = -(((X - cx)**2) / (2 * sigma_x**2) + ((Y - cy)**2) / (2 * sigma_y**2))
    Z += conf * np.exp(exponent)

heatmap = ax.contourf(X, Y, Z, levels=100, cmap='YlOrRd', alpha=0.85)
ax.axhline(y=2.0, color='#2c3e50', linestyle='--', linewidth=2.5, label='Robot SLAM Trajectory')

ax.scatter(
    df['Map_X'], df['Map_Y'], 
    color='black', marker='+', s=120, linewidths=2.5, 
    label='Mapped Fruit Center'
)

for i, row in df.iterrows():
    ax.text(
        row['Map_X'], row['Map_Y'] + 0.25, 
        f"Fruit ID: {int(row['Fruit_ID'])}\n(Conf: {row['Confidence']})", 
        fontsize=10, fontweight='bold', ha='center', color='black',
        bbox=dict(boxstyle='round,pad=0.3', edgecolor='#d35400', facecolor='#f39c12', alpha=0.7)
    )

cbar = fig.colorbar(heatmap, ax=ax)
cbar.set_label('Spatial Yield Intensity Index', fontsize=12, fontweight='bold')
ax.set_title('Durian Yield Density Heatmap', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Orchard X-Coordinate', fontsize=12, fontweight='bold', labelpad=10)
ax.set_ylabel('Orchard Y-Coordinate', fontsize=12, fontweight='bold', labelpad=10)
ax.set_xlim(-0.5, 10.0)
ax.set_ylim(0.5, 3.5)
ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9)

output_img = r'C:\fyp basic models\results\durian_yield_heatmap.png'
os.makedirs(os.path.dirname(output_img), exist_ok=True)
plt.savefig(output_img, dpi=300, bbox_inches='tight')

print(f"heatmap is saved : {output_img}")
plt.show()