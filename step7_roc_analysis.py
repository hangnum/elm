"""
Step 7: ROC Analysis and Visualization

Generate ROC curves, calculate AUC with confidence intervals.
Based on R pROC code.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy import stats

from utils import setup_logging, load_mat


def calculate_auc_ci(y_true: np.ndarray, y_scores: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Calculate AUC with confidence interval using DeLong method approximation.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted scores
        confidence: Confidence level (default: 0.95)
    
    Returns:
        Tuple of (auc_value, ci_lower, ci_upper)
    """
    from sklearn.metrics import roc_auc_score
    
    # Calculate AUC
    auc_value = float(roc_auc_score(y_true, y_scores))
    
    # Bootstrap for confidence interval
    n_bootstraps = 1000
    rng = np.random.RandomState(42)
    bootstrapped_aucs = []
    
    for _ in range(n_bootstraps):
        # Bootstrap by sampling with replacement
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            # Skip if bootstrap sample has only one class
            continue
        
        auc_boot = float(roc_auc_score(y_true[indices], y_scores[indices]))
        bootstrapped_aucs.append(auc_boot)
    
    # Calculate confidence interval
    alpha = 1 - confidence
    ci_lower = float(np.percentile(bootstrapped_aucs, alpha / 2 * 100))
    ci_upper = float(np.percentile(bootstrapped_aucs, (1 - alpha / 2) * 100))
    
    return auc_value, ci_lower, ci_upper


def plot_roc_curves(
    datasets: Dict[str, tuple],
    output_dir: Path,
    confidence: float,
    dpi: int,
    logger
) -> None:
    """
    Plot ROC curves for all datasets.
    
    Args:
        datasets: Dictionary mapping dataset names to (y_true, y_scores) tuples
        output_dir: Output directory for plots
        confidence: Confidence level for CI
        dpi: DPI for saved figures
        logger: Logger instance
    """
    colors = {
        'train': '#1f77b4',  # Blue
        'test': '#ff7f0e',   # Orange
        'test1': '#2ca02c'   # Green
    }
    
    # Create figure
    plt.figure(figsize=(8, 8))
    
    results = {}
    
    for name, (y_true, y_scores) in datasets.items():
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc, ci_lower, ci_upper = calculate_auc_ci(y_true, y_scores, confidence)
        
        # Plot ROC curve
        label = f'{name.capitalize()} (AUC={roc_auc:.3f}, 95% CI: {ci_lower:.3f}-{ci_upper:.3f})'
        plt.plot(fpr, tpr, color=colors.get(name, 'black'), lw=2, label=label)
        
        # Store results
        results[name] = {
            'auc': roc_auc,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'fpr': fpr,
            'tpr': tpr
        }
        
        logger.info(f"{name.capitalize()} - AUC: {roc_auc:.4f} (95% CI: {ci_lower:.4f}-{ci_upper:.4f})")
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Chance')
    
    # Formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    fig_path = output_dir / 'roc_curves.png'
    plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
    logger.info(f"Saved ROC curve plot to: {fig_path}")
    plt.close()
    
    return results


def roc_analysis(
    input_file: Path,
    output_dir: Path,
    plot_roc: bool,
    confidence_level: float,
    export_csv: bool,
    figure_dpi: int,
    logger
) -> None:
    """
    Perform ROC analysis and visualization.
    
    Args:
        input_file: Input .mat file from step 5
        output_dir: Output directory for results
        plot_roc: Whether to generate ROC plots
        confidence_level: Confidence level for CI
        export_csv: Export data to CSV
        figure_dpi: DPI for figures
        logger: Logger instance
    """
    logger.info(f"Loading results from: {input_file}")
    
    # Load ELM results
    data = load_mat(input_file)
    
    # Extract predictions and labels
    Ytrain = data['Ytrain']  # Shape: (2, n_train)
    Ytest = data['Ytest']    # Shape: (2, n_test)
    Ytest1 = data['Ytest1']  # Shape: (2, n_test1)
    
    # Prepare datasets
    datasets = {
        'train': (Ytrain[0, :], Ytrain[1, :]),
        'test': (Ytest[0, :], Ytest[1, :]),
        'test1': (Ytest1[0, :], Ytest1[1, :])
    }
    
    # Export data for R analysis
    if export_csv:
        for name, (y_true, y_scores) in datasets.items():
            df = pd.DataFrame({
                'label': y_true,
                'data': y_scores
            })
            csv_path = output_dir / f'roc_data_{name}.csv'
            df.to_csv(csv_path, index=False, encoding='utf-8')
            logger.info(f"Exported {name} data to: {csv_path}")
    
    # Plot ROC curves
    if plot_roc:
        logger.info("\nGenerating ROC curves...")
        results = plot_roc_curves(
            datasets=datasets,
            output_dir=output_dir,
            confidence=confidence_level,
            dpi=figure_dpi,
            logger=logger
        )
        
        # Save AUC results to CSV
        auc_df = pd.DataFrame({
            name: {
                'AUC': res['auc'],
                'CI_Lower': res['ci_lower'],
                'CI_Upper': res['ci_upper']
            }
            for name, res in results.items()
        }).T
        
        auc_csv = output_dir / 'auc_results.csv'
        auc_df.to_csv(auc_csv, encoding='utf-8')
        logger.info(f"Saved AUC results to: {auc_csv}")


def main():
    parser = argparse.ArgumentParser(description='Step 7: ROC Analysis')
    parser.add_argument('--input', type=str, required=True,
                        help='Input .mat file from step 5')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--plot_roc', action='store_true', default=True,
                        help='Generate ROC curve plots (default: True)')
    parser.add_argument('--confidence_level', type=float, default=0.95,
                        help='Confidence level for CI (default: 0.95)')
    parser.add_argument('--export_csv', action='store_true', default=True,
                        help='Export data to CSV (default: True)')
    parser.add_argument('--figure_dpi', type=int, default=300,
                        help='DPI for saved figures (default: 300)')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file)
    logger.info("Starting Step 7: ROC Analysis")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run ROC analysis
    roc_analysis(
        input_file=Path(args.input),
        output_dir=output_dir,
        plot_roc=args.plot_roc,
        confidence_level=args.confidence_level,
        export_csv=args.export_csv,
        figure_dpi=args.figure_dpi,
        logger=logger
    )
    
    logger.info("Step 7 complete!")


if __name__ == '__main__':
    main()
