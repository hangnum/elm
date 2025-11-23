"""
Organize real data directory to match the expected pipeline structure.

Input structure:
    data/
    ├── 0/           # Label 0 patients
    │   ├── 202001571_A.mat
    │   ├── 202001571_P.mat
    │   └── ...
    └── 1/           # Label 1 patients
        ├── 202002345_A.mat
        └── ...

Output structure (for each modality A, P):
    jiangmen_{modality}_CMTA/
    ├── VGG16/
    │   └── feature_extract_3_1/
    │       ├── train_data/
    │       │   ├── method0/
    │       │   │   └── {patient_id}/
    │       │   │       └── {patient_id}.mat
    │       │   └── method1/
    │       ├── test_data/
    │       └── test1_data/
    ├── train_{modality}_log.txt
    ├── test_{modality}_log.txt
    └── test1_{modality}_log.txt
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import Dict, List
import scipy.io as sio


def organize_data(
    input_dir: Path,
    output_dir: Path,
    modalities: List[str] = ['A', 'P'],
    train_ratio: float = 0.6,
    test_ratio: float = 0.2,
    test1_ratio: float = 0.2,
    seed: int = 42
):
    """
    Organize real data to match pipeline structure.
    
    Args:
        input_dir: Input directory containing 0/ and 1/ subdirectories
        output_dir: Output directory for organized data
        modalities: List of modality identifiers (e.g., ['A', 'P'])
        train_ratio: Proportion of data for training (default: 0.6)
        test_ratio: Proportion of data for test (default: 0.2)
        test1_ratio: Proportion of data for test1 (default: 0.2)
        seed: Random seed for reproducibility
    """
    import numpy as np
    np.random.seed(seed)
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    print(f"Organizing data from: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Modalities: {modalities}")
    print(f"Split ratios - Train: {train_ratio}, Test: {test_ratio}, Test1: {test1_ratio}")
    
    # Verify input directory structure
    label_dirs = [input_dir / '0', input_dir / '1']
    for label_dir in label_dirs:
        if not label_dir.exists():
            raise ValueError(f"Expected directory not found: {label_dir}")
    
    # Process each modality
    for modality in modalities:
        print(f"\n{'='*60}")
        print(f"Processing modality: {modality}")
        print(f"{'='*60}")
        
        # Collect all patient IDs for this modality from both labels
        patients_by_label = {0: [], 1: []}
        
        for label in [0, 1]:
            label_dir = input_dir / str(label)
            
            # Find all patient subdirectories
            patient_dirs = [d for d in label_dir.iterdir() if d.is_dir()]
            
            for patient_dir in patient_dirs:
                # Look for files with this modality in the patient directory
                files = list(patient_dir.glob(f'*_{modality}.mat'))
                
                for file_path in files:
                    # Extract patient ID from filename (e.g., "202001571_A.mat" -> "202001571")
                    patient_id = file_path.stem.split('_')[0]
                    patients_by_label[label].append((patient_id, file_path))
        
        print(f"Found {len(patients_by_label[0])} patients with label 0")
        print(f"Found {len(patients_by_label[1])} patients with label 1")
        
        # Split each label into train/test/test1
        splits_by_label = {0: {'train': [], 'test': [], 'test1': []},
                          1: {'train': [], 'test': [], 'test1': []}}
        
        for label in [0, 1]:
            patients = patients_by_label[label]
            n_total = len(patients)
            
            # Shuffle patients
            indices = np.random.permutation(n_total)
            
            # Calculate split sizes
            n_train = int(n_total * train_ratio)
            n_test = int(n_total * test_ratio)
            n_test1 = n_total - n_train - n_test
            
            # Split
            train_indices = indices[:n_train]
            test_indices = indices[n_train:n_train + n_test]
            test1_indices = indices[n_train + n_test:]
            
            splits_by_label[label]['train'] = [patients[i] for i in train_indices]
            splits_by_label[label]['test'] = [patients[i] for i in test_indices]
            splits_by_label[label]['test1'] = [patients[i] for i in test1_indices]
            
            print(f"  Label {label}: Train={n_train}, Test={n_test}, Test1={n_test1}")
        
        # Create directory structure
        base_path = output_dir / f'jiangmen_{modality}_CMTA'
        feature_path = base_path / 'VGG16' / 'feature_extract_3_1'
        
        # Create directories for each split and method
        split_names = {'train': 'train_data', 'test': 'test_data', 'test1': 'test1_data'}
        log_files = {}
        
        for split_key, split_folder in split_names.items():
            split_path = feature_path / split_folder
            
            # Create method directories
            for label in [0, 1]:
                method_path = split_path / f'method{label}'
                method_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize log file content
            log_files[split_key] = []
        
        # Copy files and create log entries
        for split_key, split_folder in split_names.items():
            split_path = feature_path / split_folder
            
            for label in [0, 1]:
                method_path = split_path / f'method{label}'
                patients = splits_by_label[label][split_key]
                
                for patient_id, source_file in patients:
                    # Create patient directory
                    patient_path = method_path / patient_id
                    patient_path.mkdir(parents=True, exist_ok=True)
                    
                    # Copy .mat file
                    dest_file = patient_path / f'{patient_id}.mat'
                    
                    # Check if source file has expected structure
                    try:
                        mat_data = sio.loadmat(str(source_file))
                        
                        # Save to destination (preserving the same structure)
                        shutil.copy2(source_file, dest_file)
                        
                        # Add log entry
                        log_entry = f'/fake/path/{patient_id}/image.nii\n'
                        log_files[split_key].append(log_entry)
                        
                    except Exception as e:
                        print(f"Warning: Failed to process {source_file}: {e}")
                        continue
        
        # Write log files
        for split_key in ['train', 'test', 'test1']:
            log_filename = f'{split_key}_{modality}_log.txt'
            log_path = base_path / log_filename
            
            with open(log_path, 'w', encoding='UTF-8') as f:
                f.writelines(log_files[split_key])
            
            print(f"Created {log_filename}: {len(log_files[split_key])} entries")
        
        print(f"Modality {modality} organized successfully!")
    
    print(f"\n{'='*60}")
    print("Data organization complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Organize real data for pipeline')
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory containing 0/ and 1/ subdirectories')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for organized data')
    parser.add_argument('--modalities', type=str, nargs='+', default=['A', 'P'],
                        help='Modality identifiers (default: A P)')
    parser.add_argument('--train_ratio', type=float, default=0.6,
                        help='Training set ratio (default: 0.6)')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Test set ratio (default: 0.2)')
    parser.add_argument('--test1_ratio', type=float, default=0.2,
                        help='Test1 set ratio (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.test_ratio + args.test1_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
    
    organize_data(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        modalities=args.modalities,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        test1_ratio=args.test1_ratio,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
