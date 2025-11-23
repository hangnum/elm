"""
Utility functions for the feature extraction and classification pipeline.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import scipy.io as sio


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_file: Optional log file path. If None, logs only to console.
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger('pipeline')
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_mat(file_path: Path, key: Optional[str] = None) -> Any:
    """
    Load data from MATLAB .mat file.
    
    Args:
        file_path: Path to .mat file
        key: Optional specific key to load. If None, returns entire dict.
    
    Returns:
        Loaded data (dict or specific value)
    """
    data = sio.loadmat(str(file_path))
    if key:
        return data[key]
    return data


def save_mat(file_path: Path, data_dict: Dict[str, np.ndarray]) -> None:
    """
    Save data to MATLAB .mat file.
    
    Args:
        file_path: Output .mat file path
        data_dict: Dictionary of arrays to save
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(str(file_path), data_dict)


def encode_labels(labels: List[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Encode string labels to integers.
    
    Args:
        labels: List of string labels
    
    Returns:
        Tuple of (encoded_labels, label_mapping)
        - encoded_labels: numpy array of integers
        - label_mapping: dict mapping string -> int
    """
    unique_labels = sorted(set(labels))
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    encoded = np.array([label_mapping[label] for label in labels])
    return encoded, label_mapping


def read_log_file(log_path: Path) -> List[str]:
    """
    Read patient IDs from log file.
    
    Args:
        log_path: Path to log file
    
    Returns:
        List of patient IDs extracted from log file
    """
    patient_ids = []
    with open(log_path, 'r', encoding='UTF-8') as f:
        for line in f:
            # Extract patient ID from path (format: .../patient_id/...)
            line = line.strip()
            if not line:
                continue
            # Assuming format like: /path/to/patient_id/file.ext
            # Extract second-to-last component
            parts = line.replace('\\', '/').split('/')
            if len(parts) >= 2:
                patient_id = parts[-2]
                patient_ids.append(patient_id)
    return patient_ids


def ensure_dir(path: Path) -> None:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path
    """
    path.mkdir(parents=True, exist_ok=True)


def youden_index(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """
    Calculate optimal cutoff using Youden's index and corresponding AUC.
    
    Args:
        y_true: True binary labels (0/1)
        y_score: Predicted scores/probabilities
    
    Returns:
        Tuple of (optimal_cutoff, auc)
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = auc(fpr, tpr)
    
    # Youden's index = sensitivity + specificity - 1 = tpr - fpr
    youden_idx = tpr - fpr
    optimal_idx = np.argmax(youden_idx)
    optimal_cutoff = thresholds[optimal_idx]
    
    return optimal_cutoff, auc_score


def handle_inf_nan(data: np.ndarray, inf_value: float = 1.0, nan_value: float = 0.0) -> np.ndarray:
    """
    Replace Inf and NaN values in array.
    
    Args:
        data: Input array
        inf_value: Value to replace Inf with (default: 1.0)
        nan_value: Value to replace NaN with (default: 0.0)
    
    Returns:
        Array with Inf/NaN replaced
    """
    data = data.copy()
    data[np.isinf(data)] = inf_value
    data[np.isnan(data)] = nan_value
    return data


def load_config(config_path: Path) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Dictionary with configuration parameters
    """
    import yaml
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

