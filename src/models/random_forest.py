"""
Random Forest baseline model for molecular property prediction.

Provides a simple but effective baseline using RDKit descriptors
as input features. Includes SHAP-based feature importance analysis.
"""

import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)


class RandomForestBaseline:
    """
    Random Forest baseline for molecular property prediction.
    
    Uses RDKit molecular descriptors as features and supports
    hyperparameter optimization via randomized search.
    
    Args:
        task_name: Name of the prediction task
        n_jobs: Number of parallel jobs for training
        random_state: Random seed
    """
    
    # Columns to exclude from features
    META_COLS = [
        'Drug_ID', 'original_smiles', 'Y', 'task_name',
        'source', 'task', 'canonical_smiles'
    ]
    EXCLUDE_COLS = {'ecfp4_nbits_set', 'ecfp4_bits', 'ecfp4_packed_hex'}
    
    def __init__(
        self,
        task_name: str,
        n_jobs: int = -1,
        random_state: int = 42
    ):
        self.task_name = task_name
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        self.model: Optional[RandomForestRegressor] = None
        self.feature_names: List[str] = []
        self.y_scaler: Optional[StandardScaler] = None
        self.best_params: Dict = {}
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Extract numeric feature column names."""
        all_exclude = set(self.META_COLS) | self.EXCLUDE_COLS
        
        feature_cols = [
            c for c in df.columns
            if c not in all_exclude and np.issubdtype(df[c].dtype, np.number)
        ]
        
        return sorted(feature_cols)
    
    def _build_feature_matrix(
        self,
        df: pd.DataFrame,
        feature_cols: List[str] = None
    ) -> np.ndarray:
        """Build feature matrix from DataFrame."""
        if feature_cols is None:
            feature_cols = self.feature_names
        
        X = df[feature_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X
    
    def fit(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame = None,
        y_column: str = 'Y',
        normalize_y: bool = True,
        n_iter: int = 50,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Fit the Random Forest model with hyperparameter optimization.
        
        Args:
            train_df: Training DataFrame with features and target
            valid_df: Validation DataFrame (optional)
            y_column: Name of target column
            normalize_y: Whether to normalize target values
            n_iter: Number of random search iterations
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training RF baseline for {self.task_name}")
        logger.info(f"  Train: {len(train_df)} samples")
        
        # Extract feature columns
        self.feature_names = self._get_feature_columns(train_df)
        logger.info(f"  Features: {len(self.feature_names)}")
        
        # Build feature matrices
        X_train = self._build_feature_matrix(train_df, self.feature_names)
        
        # Extract and optionally normalize targets
        y_train_raw = train_df[y_column].values.astype(np.float32)
        
        if normalize_y:
            self.y_scaler = StandardScaler()
            y_train = self.y_scaler.fit_transform(
                y_train_raw.reshape(-1, 1)
            ).ravel()
        else:
            y_train = y_train_raw
        
        # Define hyperparameter search space
        param_distributions = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 20, 30, 50, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.3, 0.5],
            'bootstrap': [True, False]
        }
        
        # Spearman scorer
        def spearman_scorer(y_true, y_pred):
            r, _ = spearmanr(y_true, y_pred)
            return r if not np.isnan(r) else -1.0
        
        scorer = make_scorer(spearman_scorer)
        
        # Run randomized search
        logger.info("  Running hyperparameter search...")
        start_time = time.time()
        
        base_rf = RandomForestRegressor(
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        search = RandomizedSearchCV(
            base_rf,
            param_distributions,
            n_iter=n_iter,
            cv=KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
            scoring=scorer,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=0
        )
        
        search.fit(X_train, y_train)
        
        self.model = search.best_estimator_
        self.best_params = search.best_params_
        
        elapsed = time.time() - start_time
        logger.info(f"  Search completed in {elapsed:.1f}s")
        logger.info(f"  Best params: {self.best_params}")
        
        # Evaluate on validation set
        results = {
            'best_params': self.best_params,
            'cv_best_score': float(search.best_score_),
            'n_features': len(self.feature_names),
            'train_size': len(train_df)
        }
        
        if valid_df is not None:
            valid_metrics = self.evaluate(valid_df, y_column)
            results['valid_metrics'] = valid_metrics
            logger.info(f"  Valid Spearman: {valid_metrics['spearman']:.4f}")
        
        return results
    
    def predict(
        self,
        df: pd.DataFrame,
        inverse_transform: bool = False
    ) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            df: DataFrame with features
            inverse_transform: Whether to inverse transform predictions
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Prepare features (handle missing columns)
        X = pd.DataFrame(0, index=df.index, columns=self.feature_names)
        available = [f for f in self.feature_names if f in df.columns]
        X[available] = df[available].values
        X = X.fillna(0).values.astype(np.float32)
        
        preds = self.model.predict(X)
        
        if inverse_transform and self.y_scaler is not None:
            preds = self.y_scaler.inverse_transform(
                preds.reshape(-1, 1)
            ).ravel()
        
        return preds
    
    def evaluate(
        self,
        df: pd.DataFrame,
        y_column: str = 'Y',
        inverse_transform: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            df: DataFrame with features and targets
            y_column: Target column name
            inverse_transform: Whether to inverse transform for metrics
            
        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(df, inverse_transform=inverse_transform)
        
        if inverse_transform and self.y_scaler is not None:
            y_true = df[y_column].values
        else:
            y_true = df[y_column].values
            if self.y_scaler is not None:
                y_true = self.y_scaler.transform(
                    y_true.reshape(-1, 1)
                ).ravel()
        
        sp, _ = spearmanr(y_true, y_pred)
        pr, _ = pearsonr(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        return {
            'spearman': float(sp) if not np.isnan(sp) else -1.0,
            'pearson': float(pr) if not np.isnan(pr) else -1.0,
            'rmse': float(rmse),
            'n_samples': len(df)
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained model."""
        if self.model is None:
            raise ValueError("Model not fitted.")
        
        importances = self.model.feature_importances_
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        })
        
        return df.sort_values('importance', ascending=False)
    
    def compute_shap_values(
        self,
        df: pd.DataFrame,
        max_samples: int = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute SHAP values for interpretability.
        
        Args:
            df: DataFrame to explain
            max_samples: Maximum samples to compute SHAP for
            
        Returns:
            Tuple of (shap_values array, feature_names)
        """
        try:
            import shap
        except ImportError:
            raise ImportError("shap package required. Install with: pip install shap")
        
        X = pd.DataFrame(0, index=df.index, columns=self.feature_names)
        available = [f for f in self.feature_names if f in df.columns]
        X[available] = df[available].values
        X = X.fillna(0).values.astype(np.float32)
        
        if max_samples and len(X) > max_samples:
            idx = np.random.choice(len(X), max_samples, replace=False)
            X = X[idx]
        
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        
        return shap_values, self.feature_names
    
    def save(self, output_dir: Path):
        """Save model and metadata."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, output_dir / "rf_model.pkl")
        
        # Save scaler
        if self.y_scaler is not None:
            joblib.dump(self.y_scaler, output_dir / "y_scaler.pkl")
        
        # Save feature importances
        importance_df = self.get_feature_importance()
        importance_df.to_csv(output_dir / "feature_importances.csv", index=False)
        
        # Save metadata
        metadata = {
            'task_name': self.task_name,
            'n_features': len(self.feature_names),
            'best_params': self.best_params,
            'feature_names': self.feature_names
        }
        
        import json
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved RF model to {output_dir}")
    
    @classmethod
    def load(cls, model_dir: Path) -> 'RandomForestBaseline':
        """Load saved model."""
        model_dir = Path(model_dir)
        
        import json
        with open(model_dir / "metadata.json") as f:
            metadata = json.load(f)
        
        instance = cls(
            task_name=metadata['task_name']
        )
        
        instance.model = joblib.load(model_dir / "rf_model.pkl")
        instance.feature_names = metadata['feature_names']
        instance.best_params = metadata.get('best_params', {})
        
        scaler_path = model_dir / "y_scaler.pkl"
        if scaler_path.exists():
            instance.y_scaler = joblib.load(scaler_path)
        
        return instance


def train_rf_baseline(
    task_name: str,
    train_path: Path,
    valid_path: Path,
    test_path: Path,
    output_dir: Path,
    compute_shap: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to train RF baseline on a task.
    
    Args:
        task_name: Task name
        train_path: Path to training parquet
        valid_path: Path to validation parquet
        test_path: Path to test parquet
        output_dir: Output directory
        compute_shap: Whether to compute SHAP values
        
    Returns:
        Results dictionary
    """
    logger.info(f"Training RF baseline for {task_name}")
    
    # Load data
    train_df = pd.read_parquet(train_path)
    valid_df = pd.read_parquet(valid_path)
    test_df = pd.read_parquet(test_path)
    
    # Train model
    model = RandomForestBaseline(task_name=task_name)
    train_results = model.fit(train_df, valid_df)
    
    # Evaluate on test set
    test_metrics = model.evaluate(test_df)
    train_results['test_metrics'] = test_metrics
    
    logger.info(f"Test Spearman: {test_metrics['spearman']:.4f}")
    
    # Save model
    model.save(output_dir)
    
    # Save predictions
    for split_name, df in [('train', train_df), ('valid', valid_df), ('test', test_df)]:
        preds = model.predict(df)
        np.save(output_dir / f"y_pred_{split_name}.npy", preds)
        np.save(output_dir / f"y_{split_name}.npy", df['Y'].values)
    
    # Compute SHAP values on test set
    if compute_shap:
        try:
            shap_values, feat_names = model.compute_shap_values(test_df)
            
            shap_df = pd.DataFrame(shap_values, columns=feat_names)
            shap_df.to_csv(output_dir / "shap_values_test.csv", index=False)
            
            # Mean absolute SHAP
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            shap_importance = pd.DataFrame({
                'feature': feat_names,
                'mean_abs_shap': mean_abs_shap
            }).sort_values('mean_abs_shap', ascending=False)
            shap_importance.to_csv(output_dir / "shap_importance.csv", index=False)
            
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
    
    # Save results
    import json
    with open(output_dir / "results.json", 'w') as f:
        json.dump(train_results, f, indent=2, default=str)
    
    return train_results
