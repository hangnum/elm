"""
Step 6: Model Evaluation

Calculate classification metrics (confusion matrix, sensitivity, specificity, etc.)
Based on MATLAB evaluation code.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

from utils import setup_logging, load_mat, save_mat


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cutoff: float
) -> Dict:
    """
    Calculate classification metrics based on cutoff threshold.
    
    Args:
        y_true: True labels (n_samples,)
        y_pred: Predicted probabilities (n_samples,)
        cutoff: Classification threshold
    
    Returns:
        Dictionary with all metrics
    """
    # Apply cutoff
    y_pred_binary = (y_pred > cutoff).astype(int)
    
    # Calculate confusion matrix
    TP = np.sum((y_true == 1) & (y_pred_binary == 1))
    FP = np.sum((y_true == 0) & (y_pred_binary == 1))
    FN = np.sum((y_true == 1) & (y_pred_binary == 0))
    TN = np.sum((y_true == 0) & (y_pred_binary == 0))
    
    # Calculate metrics
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (FP + TN) if (FP + TN) > 0 else 0.0
    fpr = 1 - specificity
    accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0.0
    ppv = TP / (TP + FP) if (TP + FP) > 0 else 0.0  # Positive Predictive Value
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0.0  # Negative Predictive Value
    
    # Calculate AUC
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(y_true, y_pred)
    except:
        auc = 0.5
    
    metrics = {
        'TP': int(TP),
        'FP': int(FP),
        'FN': int(FN),
        'TN': int(TN),
        'Cutoff': float(cutoff),
        'AUC': float(auc),
        'Sensitivity': float(sensitivity),
        'Specificity': float(specificity),
        'FPR': float(fpr),
        'Accuracy': float(accuracy),
        'PPV': float(ppv),
        'NPV': float(npv)
    }
    
    return metrics


def evaluate_model(
    input_file: Path,
    output_dir: Path,
    use_train_cutoff: bool,
    export_csv: bool,
    export_mat: bool,
    logger
) -> None:
    """
    Evaluate model and calculate metrics for all datasets.
    
    Args:
        input_file: Input .mat file from step 5 (ELM results)
        output_dir: Output directory for results
        use_train_cutoff: If True, use training cutoff for test/test1
        export_csv: Export results to CSV
        export_mat: Export results to .mat
        logger: Logger instance
    """
    logger.info(f"Loading results from: {input_file}")
    
    # Load ELM results
    data = load_mat(input_file)
    
    # Extract predictions and labels
    # Format: Ytrain/Ytest/Ytest1 = [true_labels; predicted_scores] (2 x n_samples)
    Ytrain = data['Ytrain']  # Shape: (2, n_train)
    Ytest = data['Ytest']    # Shape: (2, n_test)
    Ytest1 = data['Ytest1']  # Shape: (2, n_test1)
    
    # Extract AUC matrix (cutoffs in first row, AUC in second row)
    AUC_matrix = data['AUC']  # Shape: (2, 3)
    cutoff_train = float(AUC_matrix[0, 0])
    
    logger.info(f"Training set cutoff: {cutoff_train:.4f}")
    
    # Prepare data for evaluation
    datasets = {
        'train': (Ytrain[0, :], Ytrain[1, :]),
        'test': (Ytest[0, :], Ytest[1, :]),
        'test1': (Ytest1[0, :], Ytest1[1, :])
    }
    
    all_metrics = {}
    
    for split_name, (y_true, y_pred) in datasets.items():
        logger.info(f"\nEvaluating {split_name} set...")
        
        # Determine cutoff to use
        if split_name == 'train' or not use_train_cutoff:
            # Calculate optimal cutoff for this set
            from utils import youden_index
            cutoff, _ = youden_index(y_true, y_pred)
        else:
            # Use training set cutoff
            cutoff = cutoff_train
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred, cutoff)
        all_metrics[split_name] = metrics
        
        # Log metrics
        logger.info(f"  Cutoff: {metrics['Cutoff']:.4f}")
        logger.info(f"  AUC: {metrics['AUC']:.4f}")
        logger.info(f"  Sensitivity: {metrics['Sensitivity']:.4f}")
        logger.info(f"  Specificity: {metrics['Specificity']:.4f}")
        logger.info(f"  Accuracy: {metrics['Accuracy']:.4f}")
        logger.info(f"  PPV: {metrics['PPV']:.4f}")
        logger.info(f"  NPV: {metrics['NPV']:.4f}")
        logger.info(f"  Confusion Matrix: TP={metrics['TP']}, FP={metrics['FP']}, FN={metrics['FN']}, TN={metrics['TN']}")
        
        # Export to .mat file
        if export_mat:
            mat_file = output_dir / f'{split_name.capitalize()}_metrics.mat'
            save_mat(mat_file, metrics)
            logger.info(f"  Saved .mat to: {mat_file}")
    
    # Export summary to CSV
    if export_csv:
        df = pd.DataFrame(all_metrics).T
        csv_file = output_dir / 'metrics_summary.csv'
        df.to_csv(csv_file, encoding='utf-8')
        logger.info(f"\nSaved metrics summary to: {csv_file}")
    
    # Export label and prediction pairs for ROC analysis
    if export_csv:
        for split_name, (y_true, y_pred) in datasets.items():
            df_roc = pd.DataFrame({
                'label': y_true,
                'data': y_pred
            })
            roc_csv = output_dir / f'{split_name}_predictions.csv'
            df_roc.to_csv(roc_csv, index=False, encoding='utf-8')
            logger.info(f"Saved {split_name} predictions to: {roc_csv}")


def main():
    parser = argparse.ArgumentParser(description='Step 6: Model Evaluation')
    parser.add_argument('--input', type=str, required=True,
                        help='Input .mat file from step 5')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--use_train_cutoff', action='store_true',
                        help='Use training cutoff for test sets')
    parser.add_argument('--export_csv', action='store_true', default=True,
                        help='Export results to CSV (default: True)')
    parser.add_argument('--export_mat', action='store_true', default=True,
                        help='Export results to .mat (default: True)')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file)
    logger.info("Starting Step 6: Model Evaluation")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run evaluation
    evaluate_model(
        input_file=Path(args.input),
        output_dir=output_dir,
        use_train_cutoff=args.use_train_cutoff,
        export_csv=args.export_csv,
        export_mat=args.export_mat,
        logger=logger
    )
    
    logger.info("Step 6 complete!")


if __name__ == '__main__':
    main()
