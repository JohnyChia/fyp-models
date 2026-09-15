import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from math import pi

def generate_comparison_report(results_csv, out_csv_dir, out_diagram_dir):
    os.makedirs(out_csv_dir, exist_ok=True)
    os.makedirs(out_diagram_dir, exist_ok=True)
    
    df = pd.read_csv(results_csv)
    
    pivot_df = df.pivot(index='category', columns='version', values=['metrics/mAP50(B)', 'metrics/mAP50-95(B)'])
    
    pivot_df.columns = [f"{metric}_{version}" for metric, version in pivot_df.columns]
    pivot_df.reset_index(inplace=True)
    
    out_csv_path = os.path.join(out_csv_dir, 'model_comparison_summary.csv')
    pivot_df.to_csv(out_csv_path, index=False)
    print(f"Comparison CSV saved to {out_csv_path}")
    
    categories = df['category'].unique()
    
    plt.figure(figsize=(10, 6))
    for cat in categories:
        cat_data = df[df['category'] == cat]
        plt.plot(cat_data['version'], cat_data['metrics/mAP50(B)'], marker='o', linewidth=2.5, markersize=8, label=cat)
        
    plt.xlabel('YOLO Version', fontsize=12)
    plt.ylabel('mAP@50', fontsize=12)
    plt.title('mAP@50 Trend Across YOLO Versions', fontsize=14)
    plt.legend(title='Category')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    diagram_path_line_50 = os.path.join(out_diagram_dir, 'mAP50_trend_line_graph.png')
    plt.savefig(diagram_path_line_50, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Diagram saved to {diagram_path_line_50}")
    
    plt.figure(figsize=(10, 6))
    for cat in categories:
        cat_data = df[df['category'] == cat]
        plt.plot(cat_data['version'], cat_data['metrics/mAP50-95(B)'], marker='s', linewidth=2.5, markersize=8, label=cat)
        
    plt.xlabel('YOLO Version', fontsize=12)
    plt.ylabel('mAP@50-95', fontsize=12)
    plt.title('mAP@50-95 Trend Across YOLO Versions', fontsize=14)
    plt.legend(title='Category')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    diagram_path_line_95 = os.path.join(out_diagram_dir, 'mAP50_95_trend_line_graph.png')
    plt.savefig(diagram_path_line_95, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Diagram saved to {diagram_path_line_95}")

    metrics = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
    labels = ['Precision', 'Recall', 'mAP@50', 'mAP@50-95']
    num_vars = len(labels)
    
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]
    
    for cat in categories:
        cat_data = df[df['category'] == cat]
        
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        
        plt.xticks(angles[:-1], labels, size=12)
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2","0.4","0.6","0.8","1.0"], color="grey", size=10)
        plt.ylim(0, 1)
        
        colors = ['red', 'green', 'blue']
        
        for i, (index, row) in enumerate(cat_data.iterrows()):
            values = row[metrics].values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['version'], color=colors[i % len(colors)])
            ax.fill(angles, values, colors[i % len(colors)], alpha=0.1)
            
        plt.title(f'Performance Overview: {cat.upper()}', size=15, y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        radar_path = os.path.join(out_diagram_dir, f'radar_chart_{cat}.png')
        plt.savefig(radar_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Diagram saved to {radar_path}")

    import seaborn as sns
    plt.figure(figsize=(8, 6))
    heatmap_data_50 = df.pivot(index='category', columns='version', values='metrics/mAP50(B)')
    sns.heatmap(heatmap_data_50, annot=True, cmap='YlGnBu', fmt='.3f', linewidths=.5)
    plt.title('mAP@50 Heatmap: Categories vs Versions', fontsize=14)
    heatmap_path_50 = os.path.join(out_diagram_dir, 'mAP50_heatmap.png')
    plt.savefig(heatmap_path_50, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Diagram saved to {heatmap_path_50}")

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='version', y='metrics/mAP50(B)', data=df, palette='Set2')
    sns.swarmplot(x='version', y='metrics/mAP50(B)', data=df, color=".25", size=8)
    plt.title('mAP@50 Distribution & Variance by YOLO Version', fontsize=14)
    plt.xlabel('YOLO Version')
    plt.ylabel('mAP@50')
    boxplot_path = os.path.join(out_diagram_dir, 'mAP50_boxplot.png')
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Diagram saved to {boxplot_path}")
    
    plt.figure(figsize=(10, 6))
    pc_data = df[['version', 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']].copy()
    for col in pc_data.columns[1:]:
        pc_data[col] = (pc_data[col] - pc_data[col].min()) / (pc_data[col].max() - pc_data[col].min())
        
    pd.plotting.parallel_coordinates(pc_data, 'version', colormap='Set1', linewidth=2, alpha=0.8)
    plt.title('Parallel Coordinates: Normalized Metric Trade-offs by Version', fontsize=14)
    plt.ylabel('Normalized Score (0=Min, 1=Max)')
    plt.grid(True, linestyle='--', alpha=0.5)
    pc_path = os.path.join(out_diagram_dir, 'parallel_coordinates.png')
    plt.savefig(pc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Diagram saved to {pc_path}")

if __name__ == "__main__":
    generate_comparison_report(
        r'C:\fyp basic models\train_models\results.csv',
        r'C:\fyp basic models\train_models\results\csv',
        r'C:\fyp basic models\train_models\results\diagram'
    )
