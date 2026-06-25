import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_disease_composition(cleaned_csv, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    df = pd.read_csv(cleaned_csv)
    categories = ['Healthy', 'Algal', 'Blight', 'Colletotrichum', 'Phomopsis', 'Rhizoctonia']
    counts = df[categories].sum()
    counts = counts[counts > 0]
    
    plt.figure(figsize=(12, 8))

    colors = ['#2ca02c', '#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
    
    wedges, texts, autotexts = plt.pie(
        counts, 
        labels=counts.index, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=colors, 
        pctdistance=0.75,
        textprops={'fontsize': 12},
        wedgeprops={'edgecolor': 'white', 'linewidth': 3}
    )
    
    centre_circle = plt.Circle((0,0), 0.50, fc='white')
    plt.gca().add_artist(centre_circle)
    
    plt.title('Overall Leaf Health Distribution (Healthy vs Diseases)', fontsize=18, pad=20)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'Durian_Leaf_Disease_Composition.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    generate_disease_composition(
        r'C:\fyp basic models\results\cleaned_summary_report.csv',
        r'C:\fyp basic models\results\images'
    )