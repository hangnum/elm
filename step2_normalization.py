"""
Step 2: Feature Normalization

Apply z-score normalization to features.
Fixes from original code:
- Vectorized operations instead of loops
- Proper handling of zero variance features
- Inf/NaN handling
"""

import argparse
import numpy as np
from pathlib import Path

from utils import setup_logging, load_mat, save_mat, handle_inf_nan


def normalize_features(
    input_file: Path,
    output_file: Path,
    meanstd_file: Path,
    logger
) -> None:
    """
    Normalize features using z-score on training set.
    
    Args:
        input_file: Input .mat file with Xtrain, Ytrain, Xtest, Ytest, Xtest1, Ytest1
        output_file: Output .mat file with normalized features
        meanstd_file: Output .mat file with mean and std statistics
        logger: Logger instance
    """
    logger.info(f"Loading data from: {input_file}")
    
    # Load data
    data = load_mat(input_file)
    
    # Extract arrays
    Xtrain = data['Xtrain']  # Shape: (n_features, n_train)
    Ytrain = data['Ytrain']  # Shape: (1, n_train)
    Xtest = data['Xtest']    # Shape: (n_features, n_test)
    Ytest = data['Ytest']    # Shape: (1, n_test)
    Xtest1 = data['Xtest1']  # Shape: (n_features, n_test1)
    Ytest1 = data['Ytest1']  # Shape: (1, n_test1)
    
    logger.info(f"Xtrain shape: {Xtrain.shape}")
    logger.info(f"Xtest shape: {Xtest.shape}")
    logger.info(f"Xtest1 shape: {Xtest1.shape}")
    
    # Compute mean and std on training set
    # Each column is a sample, so compute along axis=1
    mean_train = np.mean(Xtrain, axis=1, keepdims=True)  # Shape: (n_features, 1)
    std_train = np.std(Xtrain, axis=1, keepdims=True, ddof=0)    # Shape: (n_features, 1)
    
    logger.info(f"Mean shape: {mean_train.shape}")
    logger.info(f"Std shape: {std_train.shape}")
    
    # Avoid division by zero
    std_train = np.where(std_train == 0, 1.0, std_train)
    
    # Normalize training set
    Xtrain_norm = (Xtrain - mean_train) / std_train
    
    # Normalize test sets using training statistics
    Xtest_norm = (Xtest - mean_train) / std_train
    Xtest1_norm = (Xtest1 - mean_train) / std_train
    
    # Handle Inf and NaN
    Xtrain_norm = handle_inf_nan(Xtrain_norm, inf_value=1.0, nan_value=0.0)
    Xtest_norm = handle_inf_nan(Xtest_norm, inf_value=1.0, nan_value=0.0)
    Xtest1_norm = handle_inf_nan(Xtest1_norm, inf_value=1.0, nan_value=0.0)
    
    logger.info(f"Normalization complete")
    logger.info(f"  Xtrain: min={Xtrain_norm.min():.3f}, max={Xtrain_norm.max():.3f}, mean={Xtrain_norm.mean():.3f}")
    logger.info(f"  Xtest: min={Xtest_norm.min():.3f}, max={Xtest_norm.max():.3f}, mean={Xtest_norm.mean():.3f}")
    logger.info(f"  Xtest1: min={Xtest1_norm.min():.3f}, max={Xtest1_norm.max():.3f}, mean={Xtest1_norm.mean():.3f}")
    
    # Save normalized data
    output_dict = {
        'Xtrain': Xtrain_norm,
        'Ytrain': Ytrain,
        'Xtest': Xtest_norm,
        'Ytest': Ytest,
        'Xtest1': Xtest1_norm,
        'Ytest1': Ytest1
    }
    save_mat(output_file, output_dict)
    logger.info(f"Saved normalized data to: {output_file}")
    
    # Save mean and std for future use
    meanstd = np.vstack([mean_train.flatten(), std_train.flatten()])  # Shape: (2, n_features)
    meanstd_dict = {'meanstd': meanstd}
    save_mat(meanstd_file, meanstd_dict)
    logger.info(f"Saved mean/std to: {meanstd_file}")


def main():
    parser = argparse.ArgumentParser(description='Step 2: Feature Normalization')
    parser.add_argument('--input', type=str, required=True,
                        help='Input .mat file from step 1')
    parser.add_argument('--output', type=str, required=True,
                        help='Output .mat file with normalized features')
    parser.add_argument('--meanstd', type=str, required=True,
                        help='Output .mat file with mean/std statistics')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file)
    logger.info("Starting Step 2: Feature Normalization")
    
    # Run normalization
    normalize_features(
        input_file=Path(args.input),
        output_file=Path(args.output),
        meanstd_file=Path(args.meanstd),
        logger=logger
    )
    
    logger.info("Step 2 complete!")


if __name__ == '__main__':
    main()
