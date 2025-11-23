"""
Step 5: ELM Training and Evaluation

Extreme Learning Machine classifier with proper cross-validation.
Fixes from original code:
- Proper normalization without overwriting
- Model selection using cross-validation on training set only
- No data leakage from test/test1 sets
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

from utils import setup_logging, load_mat, save_mat, youden_index


class ELMClassifier:
    """Extreme Learning Machine for binary classification."""
    
    def __init__(self, n_hidden: int, activation: str = 'sigmoid', random_state: int = None):
        """
        Initialize ELM classifier.
        
        Args:
            n_hidden: Number of hidden neurons
            activation: Activation function ('sigmoid' or 'tanh')
            random_state: Random seed for reproducibility
        """
        self.n_hidden = n_hidden
        self.activation = activation
        self.random_state = random_state
        
        # Model parameters (will be set during training)
        self.input_weights = None  # IW
        self.biases = None         # B
        self.output_weights = None # LW
        
    def _activation_function(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-x))
        elif self.activation == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ELMClassifier':
        """
        Train ELM classifier.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,) - binary 0/1
        
        Returns:
            self
        """
        n_samples, n_features = X.shape
        
        # Initialize random weights and biases
        rng = np.random.RandomState(self.random_state)
        self.input_weights = rng.randn(n_features, self.n_hidden)
        self.biases = rng.randn(self.n_hidden)
        
        # Calculate hidden layer output
        H = self._activation_function(X @ self.input_weights + self.biases)
        
        # Convert labels to {-1, 1} for better numerical stability
        y_transformed = 2 * y - 1
        
        # Calculate output weights using Moore-Penrose pseudoinverse
        self.output_weights = np.linalg.pinv(H) @ y_transformed
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities.
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            Predicted probabilities (n_samples,)
        """
        # Calculate hidden layer output
        H = self._activation_function(X @ self.input_weights + self.biases)
        
        # Calculate output
        output = H @ self.output_weights
        
        # Convert from {-1, 1} back to [0, 1] probability
        proba = (output + 1) / 2
        
        return proba
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary labels.
        
        Args:
            X: Features (n_samples, n_features)
            threshold: Classification threshold
        
        Returns:
            Predicted labels (n_samples,)
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)


def cross_validate_elm(
    X: np.ndarray,
    y: np.ndarray,
    n_hidden_range: Tuple[int, int],
    n_folds: int,
    n_trials: int,
    logger
) -> Dict:
    """
    Cross-validate ELM with different hidden layer sizes.
    
    Args:
        X: Training features (n_samples, n_features)
        y: Training labels (n_samples,)
        n_hidden_range: Range of hidden neurons (min, max)
        n_folds: Number of CV folds
        n_trials: Number of random trials per configuration
        logger: Logger instance
    
    Returns:
        Dictionary with best model and results
    """
    best_auc = 0
    best_n_hidden = None
    best_model = None
    
    logger.info(f"Cross-validation: hidden neurons {n_hidden_range}, {n_folds}-fold CV, {n_trials} trials")
    
    for trial in range(n_trials):
        # Random hidden layer size
        n_hidden = np.random.randint(n_hidden_range[0], n_hidden_range[1] + 1)
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=trial)
        cv_aucs = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Train ELM
            model = ELMClassifier(n_hidden=n_hidden, random_state=trial + fold_idx)
            model.fit(X_train_fold, y_train_fold)
            
            # Predict on validation fold
            y_val_proba = model.predict_proba(X_val_fold)
            auc = roc_auc_score(y_val_fold, y_val_proba)
            cv_aucs.append(auc)
        
        # Average AUC across folds
        mean_auc = np.mean(cv_aucs)
        
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_n_hidden = n_hidden
            # Retrain on full training set with best config
            best_model = ELMClassifier(n_hidden=n_hidden, random_state=trial)
            best_model.fit(X, y)
            
            logger.info(f"Trial {trial + 1}/{n_trials}: n_hidden={n_hidden}, CV AUC={mean_auc:.4f} (NEW BEST)")
        elif (trial + 1) % 10 == 0:
            logger.info(f"Trial {trial + 1}/{n_trials}: n_hidden={n_hidden}, CV AUC={mean_auc:.4f}")
    
    logger.info(f"Best configuration: n_hidden={best_n_hidden}, CV AUC={best_auc:.4f}")
    
    return {
        'model': best_model,
        'n_hidden': best_n_hidden,
        'cv_auc': best_auc
    }


def elm_training(
    input_file: Path,
    output_file: Path,
    n_hidden_range: Tuple[int, int],
    n_folds: int,
    n_trials: int,
    logger
) -> None:
    """
    Train ELM classifier with cross-validation.
    
    Args:
        input_file: Input .mat file from step 4
        output_file: Output .mat file with results
        n_hidden_range: Range of hidden neurons
        n_folds: Number of CV folds
        n_trials: Number of random trials
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
    
    # Transpose to (n_samples, n_features) for sklearn compatibility
    Xtrain = Xtrain.T
    Xtest = Xtest.T
    Xtest1 = Xtest1.T
    
    ytrain = Ytrain.flatten()
    ytest = Ytest.flatten()
    ytest1 = Ytest1.flatten()
    
    logger.info(f"Data shapes:")
    logger.info(f"  Xtrain: {Xtrain.shape}, ytrain: {ytrain.shape}")
    logger.info(f"  Xtest: {Xtest.shape}, ytest: {ytest.shape}")
    logger.info(f"  Xtest1: {Xtest1.shape}, ytest1: {ytest1.shape}")
    
    # Cross-validate and train best model
    result = cross_validate_elm(
        X=Xtrain,
        y=ytrain,
        n_hidden_range=n_hidden_range,
        n_folds=n_folds,
        n_trials=n_trials,
        logger=logger
    )
    
    best_model = result['model']
    
    # Evaluate on all sets
    logger.info("\nEvaluating final model...")
    
    # Training set
    ytrain_proba = best_model.predict_proba(Xtrain)
    cut_train, auc_train = youden_index(ytrain, ytrain_proba)
    
    # Test set
    ytest_proba = best_model.predict_proba(Xtest)
    cut_test, auc_test = youden_index(ytest, ytest_proba)
    
    # Test1 set
    ytest1_proba = best_model.predict_proba(Xtest1)
    cut_test1, auc_test1 = youden_index(ytest1, ytest1_proba)
    
    logger.info(f"Results:")
    logger.info(f"  Train - AUC: {auc_train:.4f}, Cutoff: {cut_train:.4f}")
    logger.info(f"  Test  - AUC: {auc_test:.4f}, Cutoff: {cut_test:.4f}")
    logger.info(f"  Test1 - AUC: {auc_test1:.4f}, Cutoff: {cut_test1:.4f}")
    
    # Save results
    output_dict = {
        'Model': {
            'n_hidden': result['n_hidden'],
            'IW': best_model.input_weights,
            'B': best_model.biases,
            'LW': best_model.output_weights,
            'activation': best_model.activation
        },
        'AUC': np.array([[cut_train, cut_test, cut_test1],
                         [auc_train, auc_test, auc_test1]]),
        'Ytrain': np.vstack([ytrain, ytrain_proba]),
        'Ytest': np.vstack([ytest, ytest_proba]),
        'Ytest1': np.vstack([ytest1, ytest1_proba])
    }
    
    save_mat(output_file, output_dict)
    logger.info(f"\nSaved results to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Step 5: ELM Training')
    parser.add_argument('--input', type=str, required=True,
                        help='Input .mat file from step 4')
    parser.add_argument('--output', type=str, required=True,
                        help='Output .mat file with model and results')
    parser.add_argument('--n_hidden_min', type=int, default=2,
                        help='Minimum hidden neurons (default: 2)')
    parser.add_argument('--n_hidden_max', type=int, default=5,
                        help='Maximum hidden neurons (default: 5)')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of CV folds (default: 5)')
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Number of random trials (default: 100)')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Log file path')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file)
    logger.info("Starting Step 5: ELM Training")
    
    # Set global random seed
    np.random.seed(args.random_state)
    
    # Run ELM training
    elm_training(
        input_file=Path(args.input),
        output_file=Path(args.output),
        n_hidden_range=(args.n_hidden_min, args.n_hidden_max),
        n_folds=args.n_folds,
        n_trials=args.n_trials,
        logger=logger
    )
    
    logger.info("Step 5 complete!")


if __name__ == '__main__':
    main()
