import os
import pandas as pd
import shutil

def export_single_patients(num_patients=5):
    """
    Extracts individual patient records (RNA-seq TSV and WSI Features)
    into standalone files so they can be uploaded to the Web Dashboard for testing.
    """
    print(f"Exporting {num_patients} individual patient records for UI testing...")
    
    # Create the output directory
    out_dir = "test_patients_for_ui"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load the main datasets
    csv_path = 'Datasets/aligned_3way_slides_MINIMAL_case_level.csv'
    tsv_path = 'Datasets/TCGA-BLCA.star_counts.tsv'
    features_dir = 'Datasets/WSI_Features'
    
    df_cohort = pd.read_csv(csv_path)
    df_rna = pd.read_csv(tsv_path, sep='\t', index_col=0)
    
    # Deduplicate RNA index to avoid Pandas crash
    df_rna = df_rna[~df_rna.index.duplicated(keep='first')]
    
    # 2. Select patients that actually have extracted WSI features
    available_patients = []
    for _, row in df_cohort.iterrows():
        pid = row['case_barcode'][:12]
        pt_path = os.path.join(features_dir, f"{pid}_features.pt")
        if os.path.exists(pt_path) and pid in df_rna.index:
            available_patients.append(pid)
            
    # Take the requested number of patients
    selected = available_patients[:num_patients]
    
    for pid in selected:
        patient_folder = os.path.join(out_dir, pid)
        os.makedirs(patient_folder, exist_ok=True)
        
        # A) Export this single patient's RNA-Seq row to a standalone TSV file
        # We keep the column headers but only save this one patient's row
        single_rna = df_rna.loc[[pid]]
        rna_out_path = os.path.join(patient_folder, f"{pid}_rna.tsv")
        single_rna.to_csv(rna_out_path, sep='\t')
        print(f"[{pid}] Saved RNA-Seq to: {rna_out_path}")
        
        # B) Copy the WSI Feature tensor
        pt_in = os.path.join(features_dir, f"{pid}_features.pt")
        pt_out = os.path.join(patient_folder, f"{pid}_features.pt")
        shutil.copy2(pt_in, pt_out)
        print(f"[{pid}] Copied WSI Features to: {pt_out}")
        
    print(f"\nDone! You can now use the files in '{out_dir}/' to upload via the website dashboard.")

if __name__ == "__main__":
    export_single_patients(5)
