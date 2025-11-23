"""
Generate dummy data for testing the pipeline.

This script creates a mock directory structure and random .mat files
that simulate the expected input format.
"""

import os
import numpy as np
import scipy.io as sio
from pathlib import Path
import argparse


def generate_dummy_data(
    output_root: Path,
    data_types: list = ['CT', 'BL'],
    methods: list = ['method0', 'method1'],
    n_train: int = 50,
    n_test: int = 20,
    n_test1: int = 25,
    n_slices_range: tuple = (10, 30),
    n_features: int = 3904,
    seed: int = 42
):
    """
    Generate dummy data structure.
    
    Args:
        output_root: Root directory for output
        data_types: List of data types (e.g., ['CT', 'BL'])
        methods: List of method/class names (these become labels)
        n_train: Number of training samples per method
        n_test: Number of test samples per method
        n_test1: Number of test1 samples per method
        n_slices_range: (min, max) number of slices per patient
        n_features: Number of features (default: 3904)
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    output_root = Path(output_root)
    
    for data_type in data_types:
        print(f"\nGenerating data for {data_type}...")
        
        # Create base paths
        base_path = output_root / f'jiangmen_{data_type}_CMTA'
        feature_path = base_path / 'VGG16' / 'feature_extract_3_1'
        
        # Create dataset splits folders
        for split in ['train_data', 'test_data', 'test1_data']:
            (feature_path / split).mkdir(parents=True, exist_ok=True)
        
        # Generate data for each split
        splits_info = {
            'train_data': ('train', n_train, f'train_{data_type}_log.txt'),
            'test_data': ('test', n_test, f'test_{data_type}_log.txt'),
            'test1_data': ('test1', n_test1, f'test1_{data_type}_log.txt'),
        }
        
        for split_folder, (split_prefix, n_samples, log_filename) in splits_info.items():
            log_lines = []
            split_path = feature_path / split_folder
            
            for method in methods:
                method_path = split_path / method
                method_path.mkdir(parents=True, exist_ok=True)
                
                for i in range(n_samples):
                    # Generate patient ID
                    patient_id = f'{split_prefix}_{method}_{i:03d}'
                    patient_path = method_path / patient_id
                    patient_path.mkdir(parents=True, exist_ok=True)
                    
                    # Generate random feature map
                    n_slices = np.random.randint(n_slices_range[0], n_slices_range[1] + 1)
                    feature_map = np.random.randn(n_slices, n_features).astype(np.float64)
                    
                    # Save as .mat file
                    mat_path = patient_path / f'{patient_id}.mat'
                    sio.savemat(str(mat_path), {'feature_map': feature_map})
                    
                    # Add to log file
                    log_lines.append(f'/fake/path/{patient_id}/image.nii\n')
            
            # Write log file
            log_path = base_path / log_filename
            with open(log_path, 'w', encoding='UTF-8') as f:
                f.writelines(log_lines)
            
            print(f"  Created {split_folder}: {len(log_lines)} samples")
        
        print(f"  Data type {data_type} complete!")
    
    print(f"\nDummy data generated successfully in: {output_root}")
    print(f"Total samples per data type:")
    print(f"  - Training: {n_train * len(methods)}")
    print(f"  - Test: {n_test * len(methods)}")
    print(f"  - Test1: {n_test1 * len(methods)}")


def main():
    parser = argparse.ArgumentParser(description='Generate dummy data for pipeline testing')
    parser.add_argument('--output', type=str, default='dummy_data',
                        help='Output root directory (default: dummy_data)')
    parser.add_argument('--n_train', type=int, default=50,
                        help='Number of training samples per class (default: 50)')
    parser.add_argument('--n_test', type=int, default=20,
                        help='Number of test samples per class (default: 20)')
    parser.add_argument('--n_test1', type=int, default=25,
                        help='Number of test1 samples per class (default: 25)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    generate_dummy_data(
        output_root=Path(args.output),
        n_train=args.n_train,
        n_test=args.n_test,
        n_test1=args.n_test1,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
