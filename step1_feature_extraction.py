"""
Step 1: Feature Extraction

Extract features from .mat files and average across slices for each patient.
Fixes from original code:
- Proper label encoding (directory name -> integer)
- Efficient path construction (no O(N^3) loops)
- Correct array shapes for MATLAB compatibility
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import scipy.io as sio

from utils import setup_logging, save_mat, encode_labels, read_log_file


def extract_features_for_split(
    feature_path: Path,
    log_path: Path,
    split_name: str,
    logger
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features for a single data split (train/test/test1).
    
    Args:
        feature_path: Path to feature extraction directory
        log_path: Path to log file containing patient IDs
        split_name: Name of the split (e.g., 'train_data')
        logger: Logger instance
    
    Returns:
        Tuple of (features, labels)
        - features: numpy array of shape (n_samples, n_features)
        - labels: numpy array of shape (n_samples,) containing label strings
    """
    logger.info(f"Processing split: {split_name}")
    
    # Read patient IDs from log file
    patient_ids = read_log_file(log_path)
    logger.info(f"  Found {len(patient_ids)} patients in log file")
    
    # Get split directory
    split_path = feature_path / split_name
    if not split_path.exists():
        logger.warning(f"  Split path does not exist: {split_path}")
        return np.array([]), np.array([])
    
    # Get all methods (labels) in this split
    method_dirs = [d for d in split_path.iterdir() if d.is_dir()]
    logger.info(f"  Found {len(method_dirs)} methods: {[d.name for d in method_dirs]}")
    
    features_list = []
    labels_list = []
    
    # Process each patient
    for patient_id in patient_ids:
        # Find which method this patient belongs to
        found = False
        for method_dir in method_dirs:
            patient_path = method_dir / patient_id
            mat_file = patient_path / f"{patient_id}.mat"
            
            if mat_file.exists():
                # Load feature map
                mat_data = sio.loadmat(str(mat_file))
                feature_map = mat_data['feature_map']  # Shape: (n_slices, n_features)
                
                # Average across slices
                feature_avg = np.mean(feature_map, axis=0)  # Shape: (n_features,)
                
                features_list.append(feature_avg)
                labels_list.append(method_dir.name)  # Directory name as label
                
                found = True
                break
        
        if not found:
            logger.warning(f"  Patient {patient_id} not found in any method directory")
    
    logger.info(f"  Successfully processed {len(features_list)} patients")
    
    # Convert to arrays
    if features_list:
        features = np.array(features_list)  # Shape: (n_samples, n_features)
        labels = np.array(labels_list)      # Shape: (n_samples,)
    else:
        features = np.array([])
        labels = np.array([])
    
    return features, labels


def process_data_type(
    data_root: Path,
    data_type: str,
    logger
) -> None:
    """
    Process one data type (CT or BL).
    
    Args:
        data_root: Root directory containing data
        data_type: Data type identifier ('CT' or 'BL')
        logger: Logger instance
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing data type: {data_type}")
    logger.info(f"{'='*60}")
    
    # Setup paths
    base_path = data_root / f'jiangmen_{data_type}_CMTA'
    feature_path = base_path / 'VGG16' / 'feature_extract_3_1'
    
    if not feature_path.exists():
        logger.error(f"Feature path does not exist: {feature_path}")
        return
    
    # Define splits
    splits = {
        'train': ('train_data', f'train_{data_type}_log.txt'),
        'test': ('test_data', f'test_{data_type}_log.txt'),
        'test1': ('test1_data', f'test1_{data_type}_log.txt'),
    }
    
    all_data = {}
    all_labels_str = {}
    
    # Extract features for each split
    for split_key, (split_folder, log_file) in splits.items():
        log_path = base_path / log_file
        features, labels_str = extract_features_for_split(
            feature_path, log_path, split_folder, logger
        )
        
        if len(features) > 0:
            all_data[split_key] = features
            all_labels_str[split_key] = labels_str
    
    # Encode all labels consistently
    # Collect all unique labels from all splits
    all_labels_combined = []
    for labels in all_labels_str.values():
        all_labels_combined.extend(labels.tolist())
    
    # Create consistent encoding
    unique_labels = sorted(set(all_labels_combined))
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    
    logger.info(f"\nLabel mapping: {label_mapping}")
    
    # Encode labels for each split
    encoded_labels = {}
    for split_key, labels_str in all_labels_str.items():
        encoded = np.array([label_mapping[label] for label in labels_str])
        encoded_labels[split_key] = encoded
    
    # Prepare output dictionary (MATLAB-compatible format)
    # Note: MATLAB expects column vectors for labels, row samples × column features for data
    output_dict = {}
    
    for split_key in ['train', 'test', 'test1']:
        if split_key in all_data:
            # Features: transpose to (n_features, n_samples) for MATLAB compatibility
            # Actually, let's keep it as (n_samples, n_features) which is standard
            # The MATLAB code does train(:,i) which suggests features are columns
            # So we need to transpose: (n_features, n_samples)
            X = all_data[split_key].T  # Transpose to (n_features, n_samples)
            y = encoded_labels[split_key].reshape(1, -1)  # Row vector (1, n_samples)
            
            output_dict[f'X{split_key}'] = X
            output_dict[f'Y{split_key}'] = y
            
            logger.info(f"{split_key}: X shape={X.shape}, Y shape={y.shape}")
    
    # Save to .mat file
    output_path = base_path / f'feature_{data_type}_map.mat'
    save_mat(output_path, output_dict)
    
    logger.info(f"\nSaved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Step 1: Feature Extraction')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing data')
    parser.add_argument('--data_types', type=str, nargs='+', default=['CT', 'BL'],
                        help='Data types to process (default: CT BL)')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Log file path (default: None, logs to console)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file)
    logger.info("Starting Step 1: Feature Extraction")
    
    data_root = Path(args.data_root)
    
    # Process each data type
    for data_type in args.data_types:
        try:
            process_data_type(data_root, data_type, logger)
        except Exception as e:
            logger.error(f"Error processing {data_type}: {str(e)}", exc_info=True)
    
    logger.info("\nStep 1 complete!")


if __name__ == '__main__':
    main()
