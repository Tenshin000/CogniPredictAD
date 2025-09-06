import pandas as pd

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

    def select_by_kbest(self, dataset: pd.DataFrame, columns: List[str], target_col: str, k: int = 10) -> List[str]:
        """
        Select the top-k features based on the f_classif score function.

        param dataset: DataFrame containing the data.
        param columns: List of column names to evaluate.
        param target_col: Name of the target column.
        param k: Number of top features to select (default=10).

        return: List of top-k column names according to SelectKBest.
        """
        if target_col not in dataset.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")
        missing = [col for col in columns if col not in dataset.columns]
        if missing:
            raise ValueError(f"Dataset does not contain columns: {missing}")

        X = dataset[columns]
        y = dataset[target_col]

        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X, y)

        mask = selector.get_support()
        selected = [col for col, keep in zip(columns, mask) if keep]

        scores = selector.scores_
        print(f"Top {k} features selected by SelectKBest (f_classif):")
        for col, score, keep in zip(columns, scores, mask):
            if keep:
                print(f"  {col}: score = {score:.4f}")
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
