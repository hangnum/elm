"""
Step 3: U-test Feature Selection

Use Mann-Whitney U test for feature selection.
Critical fix: Only use TRAINING SET for feature selection to avoid data leakage.
"""

import argparse
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu

from utils import setup_logging, load_mat, save_mat


def utest_feature_selection(
    input_file: Path,
    output_result: Path,
    output_features: Path,
    p_threshold: float,
    logger
) -> None:
    """
    Perform U-test feature selection on training set only.
    
    Args:
        input_file: Input .mat file with normalized features
        output_result: Output .mat file with U-test results
        output_features: Output .mat file with selected features
        p_threshold: P-value threshold for feature selection
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
    
    # Flatten labels
    ytrain = Ytrain.flatten()
    ytest = Ytest.flatten()
    ytest1 = Ytest1.flatten()
    
    # Get unique labels
    unique_labels = np.unique(ytrain)
    logger.info(f"Unique labels in training: {unique_labels}")
    
    if len(unique_labels) != 2:
        logger.warning(f"Expected 2 classes, found {len(unique_labels)}")
    
    # Perform U-test on TRAINING SET ONLY
    n_features = Xtrain.shape[0]
    p_values_train = np.zeros(n_features)
    
    logger.info(f"Performing U-test on {n_features} features (training set only)...")
    
    for i in range(n_features):
        feature_values = Xtrain[i, :]
        
        # Split by label
        class0_values = feature_values[ytrain == unique_labels[0]]
        class1_values = feature_values[ytrain == unique_labels[1]]
        
        # Perform Mann-Whitney U test
        try:
            statistic, p_value = mannwhitneyu(class0_values, class1_values, alternative='two-sided')
            p_values_train[i] = p_value
        except Exception as e:
            logger.warning(f"U-test failed for feature {i}: {e}")
            p_values_train[i] = 1.0  # Not significant
    
    # Select features based on p-value threshold (using TRAINING SET ONLY)
    selected_indices = np.where(p_values_train < p_threshold)[0]
    
    logger.info(f"Features selected: {len(selected_indices)} / {n_features} (p < {p_threshold})")
    
    if len(selected_indices) == 0:
        logger.warning("No features selected! Consider relaxing p_threshold")
        logger.info(f"Min p-value: {p_values_train.min():.6f}")
        logger.info(f"Selecting top 10 features by p-value instead")
        selected_indices = np.argsort(p_values_train)[:10]
    
    # Apply selection to all datasets
    Xtrain_selected = Xtrain[selected_indices, :]
    Xtest_selected = Xtest[selected_indices, :]
    Xtest1_selected = Xtest1[selected_indices, :]
    
    logger.info(f"Selected feature shapes:")
    logger.info(f"  Xtrain: {Xtrain_selected.shape}")
    logger.info(f"  Xtest: {Xtest_selected.shape}")
    logger.info(f"  Xtest1: {Xtest1_selected.shape}")
    
    # Save U-test results
    result_dict = {
        'trainp_data': p_values_train.reshape(1, -1),
        'index_utest': selected_indices.reshape(1, -1)
    }
    save_mat(output_result, result_dict)
    logger.info(f"Saved U-test results to: {output_result}")
    
    # Save selected features
    feature_dict = {
        'Xtrain': Xtrain_selected,
        'Ytrain': Ytrain,
        'Xtest': Xtest_selected,
        'Ytest': Ytest,
        'Xtest1': Xtest1_selected,
        'Ytest1': Ytest1
    }
    save_mat(output_features, feature_dict)
    logger.info(f"Saved selected features to: {output_features}")


def main():
    parser = argparse.ArgumentParser(description='Step 3: U-test Feature Selection')
    parser.add_argument('--input', type=str, required=True,
                        help='Input .mat file from step 2')
    parser.add_argument('--output_result', type=str, required=True,
                        help='Output .mat file with U-test results')
    parser.add_argument('--output_features', type=str, required=True,
                        help='Output .mat file with selected features')
    parser.add_argument('--p_threshold', type=float, default=0.05,
                        help='P-value threshold for feature selection (default: 0.05)')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file)
    logger.info("Starting Step 3: U-test Feature Selection")
    logger.info(f"P-value threshold: {args.p_threshold}")
    
    # Run U-test selection
    utest_feature_selection(
        input_file=Path(args.input),
        output_result=Path(args.output_result),
        output_features=Path(args.output_features),
        p_threshold=args.p_threshold,
        logger=logger
    )
    
    logger.info("Step 3 complete!")


if __name__ == '__main__':
    main()
