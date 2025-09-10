import numpy as np
import pandas as pd

from scipy.stats import kruskal
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE, mutual_info_classif
from typing import Optional, List

class AttributeSubsetSelector:
    """
    A class to perform Attribute Subset Selection on a dataset.
    """
    def __init__(self):
        """
        Initialization. 
        """
        pass

    def select_by_variance(self, dataset: pd.DataFrame, columns: List[str], threshold: float = 0.0) -> List[str]:
        """
        Select features whose variance is greater than a given threshold.

        param dataset: DataFrame containing the data.
        param columns: List of column names to evaluate.
        param threshold: Minimum variance threshold to keep a feature (default=0.0).

        return: List of column names whose variance is above the threshold.
        """
        missing = [col for col in columns if col not in dataset.columns]
        if missing:
            raise ValueError(f"Dataset does not contain columns: {missing}")

        variances = dataset[columns].var()
        selected = variances[variances > threshold].index.tolist()

        print(f"Columns selected by variance > {threshold}: {selected}")
        return selected

    def select_by_correlation(self, dataset: pd.DataFrame, columns: List[str], target_col: str, threshold: float = 0.0, method: str = 'pearson') -> List[str]:
        """
        Select features whose absolute correlation with the target column is greater than a given threshold.

        param dataset: DataFrame containing the data.
        param columns: List of column names to evaluate.
        param target_col: Name of the target column.
        param threshold: Minimum absolute correlation threshold (default=0.0).
        param method: Correlation method to use ('pearson', 'spearman', 'kendall').

        return: List of column names whose correlation is above the threshold.
        """
        if target_col not in dataset.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")

        missing = [col for col in columns if col not in dataset.columns]
        if missing:
            raise ValueError(f"Dataset does not contain columns: {missing}")

        corrs = dataset[columns + [target_col]].corr(method=method)[target_col].drop(target_col)
        selected = corrs[abs(corrs) > threshold].index.tolist()

        print(f"Columns selected by correlation (|r| > {threshold}) with '{target_col}': {selected}")
        return selected
    
    def select_by_mutual_info(self, dataset: pd.DataFrame, columns: List[str], target_col: str, k: int = 10, random_state: int=None):
        """
        Select the top-k features based on mutual information with the target.

        param dataset: DataFrame containing the data.
        param columns: List of column names to evaluate.
        param target_col: Name of the target column.
        param k: Number of top features to select (default=10).
        param random_state: Random state for reproducibility (default=None).

        return: List of top-k column names ranked by mutual information.
        """
        X = dataset[columns].fillna(0)
        y = dataset[target_col]
        mi = mutual_info_classif(X, y, random_state=random_state)
        mi_s = pd.Series(mi, index=columns).sort_values(ascending=False)
        selected = mi_s.head(k).index.tolist()
        print(f"Top {k} features by mutual information:", selected)
        return selected

    def select_by_kbest(self, dataset: pd.DataFrame, columns: List[str], target_col: str, k: int = 10, score_func: str = 'f_classif') -> List[str]:
        """
        Select the top-k features based on either ANOVA F-test (f_classif) or Kruskal-Wallis test.

        param dataset: DataFrame containing the data.
        param columns: List of column names to evaluate.
        param target_col: Name of the target column.
        param k: Number of top features to select (default=10).
        param score_func: Which scoring function to use: 'f_classif' or 'kruskal' (default='f_classif').

        return: List of top-k column names according to the chosen scoring function.
        """
        # Validate inputs
        if target_col not in dataset.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")
        missing = [col for col in columns if col not in dataset.columns]
        if missing:
            raise ValueError(f"Dataset does not contain columns: {missing}")

        # Ensure k is not larger than number of candidate features
        k = min(k, len(columns))

        X = dataset[columns]
        y = dataset[target_col]

        if score_func not in ('f_classif', 'kruskal'):
            raise ValueError("score_func must be either 'f_classif' or 'kruskal'.")

        if score_func == 'f_classif':
            # Use sklearn's built-in ANOVA F-test scorer
            selector = SelectKBest(score_func=f_classif, k=k)
            selector.fit(X, y)
            mask = selector.get_support()
            selected = [col for col, keep in zip(columns, mask) if keep]
            scores = selector.scores_
            print(f"Top {k} features selected by SelectKBest (ANOVA F-test):")
            for col, score, keep in zip(columns, scores, mask):
                if keep:
                    print(f"  {col}: score = {score:.4f}")

        else:  # score_func == 'kruskal'
            # Define a custom scoring function compatible with SelectKBest:
            # it must return (scores, pvalues) arrays.
            def _kruskal_score(X_arr, y_arr):
                """
                Compute Kruskal-Wallis H-statistic for each feature column.
                Returns (scores, pvalues) compatible with sklearn's SelectKBest.
                """
                # Convert to numpy arrays
                X_np = np.asarray(X_arr, dtype=float)
                y_np = np.asarray(y_arr)

                n_features = X_np.shape[1]
                scores = np.zeros(n_features)
                pvalues = np.ones(n_features)

                # Unique class labels in y
                classes = np.unique(y_np)

                for feat_idx in range(n_features):
                    # Collect values per class, dropping NaNs
                    groups = []
                    for cls in classes:
                        vals = X_np[y_np == cls, feat_idx]
                        vals = vals[~np.isnan(vals)]
                        if vals.size > 0:
                            groups.append(vals)
                    # Kruskal-Wallis requires at least two groups with data
                    if len(groups) < 2:
                        scores[feat_idx] = 0.0
                        pvalues[feat_idx] = 1.0
                        continue
                    try:
                        stat, p = kruskal(*groups)
                        # stat is the H-statistic; use it as "score"
                        scores[feat_idx] = stat
                        pvalues[feat_idx] = p
                    except Exception:
                        # On any failure fallback to neutral values
                        scores[feat_idx] = 0.0
                        pvalues[feat_idx] = 1.0

                return scores, pvalues

            selector = SelectKBest(score_func=_kruskal_score, k=k)
            # SelectKBest expects arrays without non-numeric columns; X may contain NaNs but sklearn will accept numeric NaNs
            selector.fit(X, y)
            mask = selector.get_support()
            selected = [col for col, keep in zip(columns, mask) if keep]

            # Obtain scores for reporting by calling the score func directly
            scores, pvals = _kruskal_score(X.values, y.values)
            print(f"Top {k} features selected by SelectKBest (Kruskal-Wallis):")
            for col, score, keep, p in zip(columns, scores, mask, pvals):
                if keep:
                    print(f"  {col}: H-statistic = {score:.4f}, p-value = {p:.4g}")
        
        return selected

    def select_by_rfe(self, dataset: pd.DataFrame, columns: List[str], target_col: str, n_features_to_select: int = 5, estimator: Optional[BaseEstimator] = None) -> List[str]:
        """
        Select features using Recursive Feature Elimination (RFE) with a given estimator.

        param dataset: DataFrame containing the data.
        param columns: List of column names to evaluate.
        param target_col: Name of the target column.
        param n_features_to_select: Number of features to select (default=5).
        param estimator: Estimator to use for RFE (default=RandomForestClassifier).
        
        return: List of selected column names by RFE.
        """
        if target_col not in dataset.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")
        missing = [col for col in columns if col not in dataset.columns]
        if missing:
            raise ValueError(f"Dataset does not contain columns: {missing}")

        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)

        X = dataset[columns]
        y = dataset[target_col]

        rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select)
        rfe.fit(X, y)

        mask = rfe.support_
        selected = [col for col, keep in zip(columns, mask) if keep]

        print(f"Features selected by RFE ({n_features_to_select} features): {selected}")
        return selected
