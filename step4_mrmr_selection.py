"""
Step 4: MRMR Feature Selection

Apply Minimum Redundancy Maximum Relevance feature selection.
Uses training set only to avoid data leakage.
"""

import argparse
import numpy as np
from pathlib import Path
from typing import List
from threading import Lock

from sklearn.metrics import mutual_info_score

from utils import setup_logging, load_mat, save_mat


def calculate_mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate mutual information between two variables.
    
    Args:
        x: First variable (1D array)
        y: Second variable (1D array)
    
    Returns:
        Mutual information value
    """
    from sklearn.metrics import mutual_info_score
    
    # Discretize continuous variables
    n_bins = 10
    x_binned = np.digitize(x, bins=np.linspace(x.min(), x.max(), n_bins))
    if y.dtype in [np.float32, np.float64]:
        y_binned = np.digitize(y, bins=np.linspace(y.min(), y.max(), n_bins))
    else:
        y_binned = y
    
    return float(mutual_info_score(x_binned, y_binned))


def _digitize_vector(arr: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """
    Discretize a continuous vector for mutual information calculation.
    Keeps deterministic bin edges to preserve reproducibility.
    """
    min_v = arr.min()
    max_v = arr.max()
    if max_v == min_v:
        return np.zeros_like(arr, dtype=np.int32)
    bins = np.linspace(min_v, max_v, n_bins)
    return np.digitize(arr, bins=bins).astype(np.int32)


def mrmr_selection(
    X: np.ndarray,
    y: np.ndarray,
    n_features: int,
    logger,
    n_jobs: int = -1
) -> List[int]:
    """
    Select features using MRMR algorithm with parallel computation.
    
    Args:
        X: Feature matrix (n_features, n_samples)
        y: Labels (n_samples,)
        n_features: Number of features to select
        logger: Logger instance
        n_jobs: Number of parallel jobs (-1 for all CPUs, default: -1)
    
    Returns:
        List of selected feature indices
    """
    from concurrent.futures import ThreadPoolExecutor
    import os
    
    # Determine number of threads
    if n_jobs == -1:
        cpu_count = os.cpu_count() or 1
        n_jobs = int(cpu_count)
    if n_jobs is None or n_jobs < 1:
        n_jobs = 1
    
    n_total_features = X.shape[0]
    selected_features = []
    remaining_features = list(range(n_total_features))
    
    logger.info(f"Starting MRMR selection: {n_features} from {n_total_features}")
    logger.info(f"Using {n_jobs} threads for parallel computation")
    
    # Pre-discretize features and labels once to avoid repeated digitize cost
    logger.info("Pre-discretizing features for mutual information computation")
    X_discrete = np.empty_like(X, dtype=np.int32)
    for i in range(n_total_features):
        X_discrete[i, :] = _digitize_vector(X[i, :])
    if y.dtype in [np.float32, np.float64]:
        y_discrete = _digitize_vector(y)
    else:
        y_discrete = y.astype(np.int32)
    
    # Pre-compute relevance (MI with target) in parallel
    logger.info("Computing relevance (MI with target) in parallel")
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        relevance_pairs = list(
            executor.map(
                lambda idx: (idx, float(mutual_info_score(X_discrete[idx, :], y_discrete))),
                remaining_features
            )
        )
        
        relevance = {idx: mi for idx, mi in relevance_pairs}
        
        # Select first feature with maximum relevance
        relevance_array = np.array([relevance[i] for i in remaining_features])
        best_idx = remaining_features[np.argmax(relevance_array)]
        selected_features.append(best_idx)
        remaining_features.remove(best_idx)
        
        logger.info(f"Selected feature 1/{n_features}: index {best_idx}")
        
        # Cache for redundancy MI to avoid recomputation, protected by lock
        redundancy_cache = {}
        cache_lock = Lock()
        
        def get_pair_mi(i: int, j: int) -> float:
            key = (i, j) if i <= j else (j, i)
            with cache_lock:
                if key in redundancy_cache:
                    return redundancy_cache[key]
            mi = float(mutual_info_score(X_discrete[i, :], X_discrete[j, :]))
            with cache_lock:
                redundancy_cache[key] = mi
            return mi
        
        # Select remaining features
        for k in range(1, n_features):
            if len(remaining_features) == 0:
                logger.warning(f"No more features available. Selected {len(selected_features)} features.")
                break
            
            # Snapshot to keep deterministic redundancy computation within this round
            selected_snapshot = tuple(selected_features)
            
            def compute_mrmr_score(idx: int) -> float:
                if not selected_snapshot:
                    return relevance[idx]
                redundancy_vals = [get_pair_mi(idx, sel_idx) for sel_idx in selected_snapshot]
                redundancy = float(np.mean(redundancy_vals)) if redundancy_vals else 0.0
                return relevance[idx] - redundancy
            
            mrmr_scores = list(executor.map(compute_mrmr_score, remaining_features))
            
            # Select feature with highest MRMR score
            best_idx = remaining_features[np.argmax(mrmr_scores)]
            selected_features.append(best_idx)
            remaining_features.remove(best_idx)
            
            if (k + 1) % 10 == 0 or k + 1 == n_features:
                logger.info(f"Selected feature {k + 1}/{n_features}: index {best_idx}")
    
    return selected_features


def mrmr_feature_selection(
    input_file: Path,
    output_file: Path,
    n_features: int,
    n_jobs: int,
    logger
) -> None:
    """
    Perform MRMR feature selection on training set only.
    
    Args:
        input_file: Input .mat file from step 3
        output_file: Output .mat file with MRMR-selected features
        n_features: Number of features to select
        n_jobs: Number of parallel jobs (-1 for all CPUs)
        logger: Logger instance
    """
    logger.info(f"Loading data from: {input_file}")
    
    # Load data
    data = load_mat(input_file)
    
    Xtrain = data['Xtrain']  # Shape: (n_features, n_train)
    Ytrain = data['Ytrain']  # Shape: (1, n_train)
    Xtest = data['Xtest']    # Shape: (n_features, n_test)
    Ytest = data['Ytest']    # Shape: (1, n_test)
    Xtest1 = data['Xtest1']  # Shape: (n_features, n_test1)
    Ytest1 = data['Ytest1']  # Shape: (1, n_test1)
    
    logger.info(f"Input shapes:")
    logger.info(f"  Xtrain: {Xtrain.shape}")
    logger.info(f"  Xtest: {Xtest.shape}")
    logger.info(f"  Xtest1: {Xtest1.shape}")
    
    ytrain = Ytrain.flatten()
    
    # Limit n_features to available features
    max_features = Xtrain.shape[0]
    if n_features > max_features:
        logger.warning(f"Requested {n_features} features but only {max_features} available")
        n_features = max_features
    
    # Perform MRMR selection on TRAINING SET ONLY
    selected_indices = mrmr_selection(Xtrain, ytrain, n_features, logger, n_jobs)
    
    logger.info(f"MRMR selected {len(selected_indices)} features")
    
    # Apply selection to all datasets
    selected_indices_array = np.array(selected_indices)
    Xtrain_selected = Xtrain[selected_indices_array, :]
    Xtest_selected = Xtest[selected_indices_array, :]
    Xtest1_selected = Xtest1[selected_indices_array, :]
    
    logger.info(f"Selected feature shapes:")
    logger.info(f"  Xtrain: {Xtrain_selected.shape}")
    logger.info(f"  Xtest: {Xtest_selected.shape}")
    logger.info(f"  Xtest1: {Xtest1_selected.shape}")
    
    # Save selected features
    output_dict = {
        'Xtrain': Xtrain_selected,
        'Ytrain': Ytrain,
        'Xtest': Xtest_selected,
        'Ytest': Ytest,
        'Xtest1': Xtest1_selected,
        'Ytest1': Ytest1
    }
    save_mat(output_file, output_dict)
    logger.info(f"Saved MRMR-selected features to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Step 4: MRMR Feature Selection')
    parser.add_argument('--input', type=str, required=True,
                        help='Input .mat file from step 3')
    parser.add_argument('--output', type=str, required=True,
                        help='Output .mat file with MRMR-selected features')
    parser.add_argument('--n_features', type=int, default=50,
                        help='Number of features to select (default: 50)')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of parallel jobs, -1 for all CPUs (default: -1)')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file)
    logger.info("Starting Step 4: MRMR Feature Selection")
    logger.info(f"Target number of features: {args.n_features}")
    
    # Run MRMR selection
    mrmr_feature_selection(
        input_file=Path(args.input),
        output_file=Path(args.output),
        n_features=args.n_features,
        n_jobs=args.n_jobs,
        logger=logger
    )
    
    logger.info("Step 4 complete!")


if __name__ == '__main__':
    main()
