"""
Pipeline Orchestrator

Run the complete feature extraction and classification pipeline.
Supports configuration files and includes evaluation + ROC analysis.
"""

import argparse
from pathlib import Path
import sys

from utils import setup_logging, load_config


def run_pipeline_with_config(
    data_root: Path,
    data_types: list,
    config: dict,
    log_file: str,
    logger,
    run_evaluation: bool = True,
    run_roc: bool = True
) -> None:
    """
    Run the complete pipeline for all data types using configuration.
    
    Args:
        data_root: Root directory containing data
        data_types: List of data types to process
        config: Configuration dictionary
        log_file: Log file path
        logger: Logger instance
        run_evaluation: Whether to run step 6 (evaluation)
        run_roc: Whether to run step 7 (ROC analysis)
    """
    import subprocess
    
    # Extract config parameters
    p_threshold = config.get('utest', {}).get('p_threshold', 0.05)
    n_mrmr_features = config.get('mrmr', {}).get('n_features', 50)
    n_mrmr_jobs = config.get('mrmr', {}).get('n_jobs', -1)
    elm_config = config.get('elm', {})
    n_hidden_min = elm_config.get('n_hidden_min', 2)
    n_hidden_max = elm_config.get('n_hidden_max', 5)
    n_folds = elm_config.get('n_folds', 5)
    n_trials = elm_config.get('n_trials', 100)
    
    eval_config = config.get('evaluation', {})
    roc_config = config.get('roc_analysis', {})
    
    for data_type in data_types:
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing data type: {data_type}")
        logger.info(f"{'='*80}\n")
        
        base_path = data_root / f'jiangmen_{data_type}_CMTA'
        
        # Step 1: Feature extraction
        logger.info("Step 1: Feature Extraction")
        cmd = [
            sys.executable, 'step1_feature_extraction.py',
            '--data_root', str(data_root),
            '--data_types', data_type
        ]
        if log_file:
            cmd.extend(['--log_file', log_file])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Step 1 failed: {result.stderr}")
            continue
        logger.info("Step 1 complete\n")
        
        # Step 2: Normalization
        logger.info("Step 2: Normalization")
        input_file = base_path / f'feature_{data_type}_map.mat'
        output_file = base_path / 'feature_normalized.mat'
        meanstd_file = base_path / 'feature_meanstd.mat'
        
        cmd = [
            sys.executable, 'step2_normalization.py',
            '--input', str(input_file),
            '--output', str(output_file),
            '--meanstd', str(meanstd_file)
        ]
        if log_file:
            cmd.extend(['--log_file', log_file])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Step 2 failed: {result.stderr}")
            continue
        logger.info("Step 2 complete\n")
        
        # Step 3: U-test selection
        logger.info("Step 3: U-test Selection")
        input_file = output_file
        output_result = base_path / 'Utest_result.mat'
        output_features = base_path / 'feature_Utest.mat'
        
        cmd = [
            sys.executable, 'step3_utest_selection.py',
            '--input', str(input_file),
            '--output_result', str(output_result),
            '--output_features', str(output_features),
            '--p_threshold', str(p_threshold)
        ]
        if log_file:
            cmd.extend(['--log_file', log_file])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Step 3 failed: {result.stderr}")
            continue
        logger.info("Step 3 complete\n")
        
        # Step 4: MRMR selection
        logger.info("Step 4: MRMR Selection")
        input_file = output_features
        output_file = base_path / 'MRMRfeature.mat'
        
        cmd = [
            sys.executable, 'step4_mrmr_selection.py',
            '--input', str(input_file),
            '--output', str(output_file),
            '--n_features', str(n_mrmr_features),
            '--n_jobs', str(n_mrmr_jobs)
        ]
        if log_file:
            cmd.extend(['--log_file', log_file])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Step 4 failed: {result.stderr}")
            continue
        logger.info("Step 4 complete\n")
        
        # Step 5: ELM training
        logger.info("Step 5: ELM Training")
        input_file = output_file
        elm_output_file = base_path / f'elm_model_result_{data_type}.mat'
        
        cmd = [
            sys.executable, 'step5_elm_training.py',
            '--input', str(input_file),
            '--output', str(elm_output_file),
            '--n_hidden_min', str(n_hidden_min),
            '--n_hidden_max', str(n_hidden_max),
            '--n_folds', str(n_folds),
            '--n_trials', str(n_trials)
        ]
        if log_file:
            cmd.extend(['--log_file', log_file])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Step 5 failed: {result.stderr}")
            continue
        logger.info("Step 5 complete\n")
        
        # Step 6: Evaluation (optional)
        if run_evaluation:
            logger.info("Step 6: Model Evaluation")
            eval_output_dir = base_path / 'evaluation'
            
            cmd = [
                sys.executable, 'step6_evaluation.py',
                '--input', str(elm_output_file),
                '--output_dir', str(eval_output_dir)
            ]
            if eval_config.get('use_train_cutoff', True):
                cmd.append('--use_train_cutoff')
            if eval_config.get('export_csv', True):
                cmd.append('--export_csv')
            if eval_config.get('export_mat', True):
                cmd.append('--export_mat')
            if log_file:
                cmd.extend(['--log_file', log_file])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Step 6 failed: {result.stderr}")
            else:
                logger.info("Step 6 complete\n")
        
        # Step 7: ROC Analysis (optional)
        if run_roc:
            logger.info("Step 7: ROC Analysis")
            roc_output_dir = base_path / 'roc_analysis'
            
            cmd = [
                sys.executable, 'step7_roc_analysis.py',
                '--input', str(elm_output_file),
                '--output_dir', str(roc_output_dir)
            ]
            if roc_config.get('plot_roc', True):
                cmd.append('--plot_roc')
            cmd.extend(['--confidence_level', str(roc_config.get('confidence_level', 0.95))])
            if roc_config.get('export_csv', True):
                cmd.append('--export_csv')
            cmd.extend(['--figure_dpi', str(roc_config.get('figure_dpi', 300))])
            if log_file:
                cmd.extend(['--log_file', log_file])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Step 7 failed: {result.stderr}")
            else:
                logger.info("Step 7 complete\n")
        
        logger.info(f"Pipeline complete for {data_type}!")
        logger.info(f"Results saved in: {base_path}\n")


def main():
    parser = argparse.ArgumentParser(description='Run complete pipeline')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML configuration file (optional)')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Root directory containing data')
    parser.add_argument('--data_types', type=str, nargs='+', default=None,
                        help='Data types to process (default: CT BL)')
    parser.add_argument('--p_threshold', type=float, default=None,
                        help='P-value threshold for U-test (default: 0.05)')
    parser.add_argument('--n_mrmr_features', type=int, default=None,
                        help='Number of MRMR features (default: 50)')
    parser.add_argument('--n_hidden_min', type=int, default=None,
                        help='Minimum hidden neurons (default: 2)')
    parser.add_argument('--n_hidden_max', type=int, default=None,
                        help='Maximum hidden neurons (default: 5)')
    parser.add_argument('--n_folds', type=int, default=None,
                        help='Number of CV folds (default: 5)')
    parser.add_argument('--n_trials', type=int, default=None,
                        help='Number of ELM trials (default: 100)')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Log file path (default: pipeline.log)')
    parser.add_argument('--skip_evaluation', action='store_true',
                        help='Skip evaluation step')
    parser.add_argument('--skip_roc', action='store_true',
                        help='Skip ROC analysis step')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and Path(args.config).exists():
        config = load_config(Path(args.config))
        print(f"Loaded configuration from: {args.config}")
    else:
        # Use default configuration
        default_config_path = Path(__file__).parent / 'config_default.yaml'
        if default_config_path.exists():
            config = load_config(default_config_path)
            print(f"Using default configuration from: {default_config_path}")
        else:
            config = {}
            print("No configuration file found, using command-line arguments only")
    
    # Command-line arguments override config file
    data_root = args.data_root if args.data_root else config.get('data', {}).get('root', 'dummy_data')
    data_types = args.data_types if args.data_types else config.get('data', {}).get('types', ['CT', 'BL'])
    log_file = args.log_file if args.log_file else config.get('logging', {}).get('log_file', 'pipeline.log')
    
    # Override config with command-line arguments if provided
    if args.p_threshold is not None:
        config.setdefault('utest', {})['p_threshold'] = args.p_threshold
    if args.n_mrmr_features is not None:
        config.setdefault('mrmr', {})['n_features'] = args.n_mrmr_features
    if args.n_hidden_min is not None:
        config.setdefault('elm', {})['n_hidden_min'] = args.n_hidden_min
    if args.n_hidden_max is not None:
        config.setdefault('elm', {})['n_hidden_max'] = args.n_hidden_max
    if args.n_folds is not None:
        config.setdefault('elm', {})['n_folds'] = args.n_folds
    if args.n_trials is not None:
        config.setdefault('elm', {})['n_trials'] = args.n_trials
    
    # Setup logging
    logger = setup_logging(log_file)
    logger.info("="*80)
    logger.info("STARTING FULL PIPELINE")
    logger.info("="*80)
    logger.info(f"Data root: {data_root}")
    logger.info(f"Data types: {data_types}")
    logger.info(f"Parameters:")
    logger.info(f"  - U-test p-threshold: {config.get('utest', {}).get('p_threshold', 0.05)}")
    logger.info(f"  - MRMR features: {config.get('mrmr', {}).get('n_features', 50)}")
    elm_config = config.get('elm', {})
    logger.info(f"  - ELM hidden neurons: {elm_config.get('n_hidden_min', 2)}-{elm_config.get('n_hidden_max', 5)}")
    logger.info(f"  - CV folds: {elm_config.get('n_folds', 5)}")
    logger.info(f"  - ELM trials: {elm_config.get('n_trials', 100)}")
    logger.info(f"  - Run evaluation: {not args.skip_evaluation}")
    logger.info(f"  - Run ROC analysis: {not args.skip_roc}")
    logger.info("")
    
    # Run pipeline
    run_pipeline_with_config(
        data_root=Path(data_root),
        data_types=data_types,
        config=config,
        log_file=log_file,
        logger=logger,
        run_evaluation=not args.skip_evaluation,
        run_roc=not args.skip_roc
    )
    
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*80)


if __name__ == '__main__':
    main()
