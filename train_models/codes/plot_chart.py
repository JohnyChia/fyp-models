import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_academic_comparison(raw_csv, cleaned_csv, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df_raw = pd.read_csv(raw_csv)
    df_clean = pd.read_csv(cleaned_csv)
    disease_cols = ['Algal', 'Blight', 'Colletotrichum', 'Phomopsis', 'Rhizoctonia']
    df_raw['Total_Disease'] = df_raw[disease_cols].sum(axis=1)
    df_clean['Total_Disease'] = df_clean[disease_cols].sum(axis=1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    
    ax1.scatter(range(len(df_raw)), df_raw['Total_Disease'], color='red', alpha=0.5, s=20, label='Diseased')
    ax1.scatter(range(len(df_raw)), df_raw['Healthy'], color='green', alpha=0.5, s=20, label='Healthy')
    ax1.set_title(f'Raw Data Distribution (Noise: {len(df_raw)} samples)', fontsize=14)
    ax1.set_xlabel('Sample Sequence')
    ax1.set_ylabel('Leaf Count')
    ax1.grid(True, linestyle='--')
    ax1.legend()

    ax2.scatter(range(len(df_clean)), df_clean['Total_Disease'], color='red', s=40, label='Diseased')
    ax2.scatter(range(len(df_clean)), df_clean['Healthy'], color='green', s=40, label='Healthy')
    ax2.set_title(f'Refined Data Distribution (Noise Removed: {len(df_clean)} samples)', fontsize=14)
    ax2.set_xlabel('Filtered Sample Sequence')
    ax2.grid(True, linestyle='--')
    ax2.legend()
    
    plt.suptitle('Comparison Graph: Data Preprocessing Impact', fontsize=16)

    save_path = os.path.join(save_dir, 'Comparison Dataset.png')
    plt.savefig(save_path, dpi=300)
    print(f"picture is saved : {save_path}")
    
    plt.show()

if __name__ == "__main__":
    generate_academic_comparison(
        r'C:\fyp basic models\results\summary_report.csv',
        r'C:\fyp basic models\results\cleaned_summary_report.csv',
        r'C:\fyp basic models\results\images'
    )