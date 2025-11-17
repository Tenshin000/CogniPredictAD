import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import zscore
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from typing import Dict, List, Optional


class OutlierDetector:
    """
    A class to perform outlier detection on a dataset.
    """
    # ----------------------------#
    #        INITIALIZATION       #
    # ----------------------------# 
    def __init__(self):
        """
        Initialization.
        """
        pass

    
    def _get_columns(self, dataset: pd.DataFrame, columns: Optional[List[str]]) -> List[str]:
        """
        Get the columns to analyze. If None, select all numeric columns.

        param dataset: The input DataFrame.
        param columns: List of column names or None.

        return: List of numeric column names.
        """
        if columns is None:
            columns = dataset.select_dtypes(include=[np.number]).columns.tolist()
        missing = [col for col in columns if col not in dataset.columns]
        if missing:
            raise ValueError(f"Dataset does not contain columns: {missing}")
        return columns

    # ----------------------------#
    #     UNIVARIATE ANALYSIS     #
    # ----------------------------# 
    def detect_by_iqr(self, dataset: pd.DataFrame, columns: Optional[List[str]] = None,
                  factor: float = 1.5, verbose: bool = True) -> Dict[str, Dict[str, List]]:
        """
        Detect outliers in numeric columns using the Interquartile Range (IQR) method.
        Also displays boxplots for each analyzed column.

        param dataset: DataFrame containing the data.
        param columns: List of column names to analyze. If None, all numeric columns are used.
        param factor: Multiplicative factor for IQR to determine bounds (default=1.5).
        param verbose: If True, prints detailed information about detected outliers.

        return: Dictionary with column names as keys and dictionaries containing indices and values of outliers.
        """
        cols = self._get_columns(dataset, columns)
        results = {}

        for col in cols:
            Q1 = dataset[col].quantile(0.25)
            Q3 = dataset[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR

            mask = (dataset[col] < lower_bound) | (dataset[col] > upper_bound)
            outlier_indices = dataset[mask].index.tolist()
            outlier_values = dataset.loc[mask, col].tolist()

            results[col] = {
                "indices": outlier_indices,
                "values": outlier_values
            }

            if verbose:
                print(f"[ IQR Method ] Column: {col}")
                print(f"Lower bound: {lower_bound:.4f}, Upper bound: {upper_bound:.4f}")
                print(f"Detected {len(outlier_indices)} outliers.")
                for idx, val in zip(outlier_indices, outlier_values):
                    print(f"  Index: {idx}, Value: {val}")

            plt.figure(figsize=(6, 4))
            sns.boxplot(x=dataset[col], color='skyblue')
            plt.title(f'Boxplot for {col} (IQR)')
            plt.show()

        return results


    def detect_by_zscore(self, dataset: pd.DataFrame, columns: Optional[List[str]] = None,
                     threshold: float = 3.0, verbose: bool = True) -> Dict[str, Dict[str, List]]:
        """
        Detect outliers in numeric columns using the Z-score method.

        param dataset: DataFrame containing the data.
        param columns: List of column names to analyze. If None, all numeric columns are used.
        param threshold: Z-score threshold above which values are considered outliers (default=3.0).
        param verbose: If True, prints detailed information about detected outliers.

        return: Dictionary with column names as keys and dictionaries containing indices and values of outliers.
        """
        cols = self._get_columns(dataset, columns)
        results = {}

        # Compute z-scores column-wise using pandas to have clearer NaN behavior
        for col in cols:
            ser = dataset[col]
            mean = ser.mean(skipna=True)
            std = ser.std(skipna=True, ddof=0)
            if std == 0 or np.isnan(std):
                # No Variation -> No Outliers
                outlier_mask = pd.Series(False, index=ser.index)
            else:
                z = (ser - mean).abs() / std
                outlier_mask = z > threshold

            outlier_idx_labels = ser.index[outlier_mask].tolist() # original labels
            outlier_values = ser[outlier_mask].tolist()

            results[col] = {"indices": outlier_idx_labels, "values": outlier_values}

            if verbose:
                print(f"[ Z-score ] Column: {col}  Threshold: {threshold}")
                print(f"Detected {len(outlier_idx_labels)} outliers.")
                for idx_label, val in zip(outlier_idx_labels, outlier_values):
                    print(f"  Index: {idx_label}, Value: {val}")

        return results

    # ----------------------------#
    #    MULTIVARIATE ANALYSIS    #
    # ----------------------------# 
    def detect_by_lof(self, dataset: pd.DataFrame, columns: Optional[List[str]] = None, n_neighbors: int = 20,
                     impute_strategy: Optional[str] = None, verbose: bool = True) -> Dict[str, List]:
        """
        Detect outliers using the Local Outlier Factor (LOF) method.
        Produces scatter plots of LOF scores and highlights detected outliers.

        param dataset: DataFrame containing the data.
        param columns: List of column names to analyze. If None, all numeric columns are used.
        param n_neighbors: Number of neighbor points to use for LOF (default=20).
        impute_strategy: None (default) -> do not impute, 'zero' -> fillna(0), 'median' -> fill with median
        param verbose: If True, prints detailed information about detected outliers.

        return: Dictionary containing indices, values, and scores of outliers.
        """
        cols = self._get_columns(dataset, columns)
        X_df = dataset[cols].copy()

        # optional simple imputation (user can also pre-clean dataset)
        if impute_strategy == 'zero':
            X = X_df.fillna(0).values
            idx_labels = X_df.index
        elif impute_strategy == 'median':
            X = X_df.fillna(X_df.median()).values
            idx_labels = X_df.index
        else:
            # do not impute: drop rows with NaN for LOF (but remember original indices)
            not_nan_mask = ~X_df.isna().any(axis=1)
            X = X_df[not_nan_mask].values
            idx_labels = X_df.index[not_nan_mask]

        n_samples = X.shape[0]

        if n_samples == 0:
            return {"indices": [], "values": [], "scores": []}

        # ensure n_neighbors < n_samples
        if n_neighbors >= n_samples:
            n_neighbors_eff = max(1, n_samples - 1)
            if verbose:
                print(f"Warning: requested n_neighbors={n_neighbors} >= n_samples={n_samples}. "
                    f"Using n_neighbors={n_neighbors_eff} instead.")
            n_neighbors = n_neighbors_eff

        lof = LocalOutlierFactor(n_neighbors=n_neighbors)
        y_pred = lof.fit_predict(X)
        scores = -lof.negative_outlier_factor_

        outlier_positions = np.where(y_pred == -1)[0]
        outlier_idx_labels = idx_labels[outlier_positions].tolist()
        outlier_values = dataset.loc[outlier_idx_labels, cols].values.tolist()
        outlier_scores = scores[outlier_positions].tolist()

        if verbose:
            print(f"[ LOF ] n_neighbors={n_neighbors}, detected {len(outlier_idx_labels)} outliers.")
            for lbl, val, sc in zip(outlier_idx_labels, outlier_values, outlier_scores):
                print(f"  Index: {lbl}, Values: {val}, Score: {sc:.4f}")

        return {
            "indices": outlier_idx_labels,
            "values": outlier_values,
            "scores": outlier_scores
        }



    def detect_by_dbscan(self, dataset: pd.DataFrame, columns: Optional[List[str]] = None, eps: float = 0.5, 
                     min_samples: Optional[int] = None, impute_strategy: Optional[str] = None, verbose: bool = True) -> Dict[str, List]:
        """
        Detect outliers using the DBSCAN clustering algorithm.
        Points assigned to cluster -1 are considered outliers.
        Produces a 2D scatter plot if exactly two numeric columns are analyzed.

        param dataset: DataFrame containing the data.
        param columns: List of column names to analyze. If None, all numeric columns are used.
        param eps: Maximum distance between two samples for them to be considered neighbors (default=0.5).
        param min_samples: Minimum number of points to form a cluster. If None, defaults to 2 x number of features.
        impute_strategy: None (default) -> do not impute, 'zero' -> fillna(0), 'median' -> fill with median
        param verbose: If True, prints detailed information about detected outliers.

        return: Dictionary with keys:
            - "indices": List of indices of detected outliers.
            - "values": List of corresponding outlier values.
            - "labels": List of cluster labels for all points (-1 = outlier).
        """
        cols = self._get_columns(dataset, columns)
        X_df = dataset[cols].copy()

        # imputation choices
        if impute_strategy == 'zero':
            X = X_df.fillna(0).values
            idx_labels = X_df.index
        elif impute_strategy == 'median':
            X = X_df.fillna(X_df.median()).values
            idx_labels = X_df.index
        else:
            # drop rows with NaN
            not_nan_mask = ~X_df.isna().any(axis=1)
            X = X_df[not_nan_mask].values
            idx_labels = X_df.index[not_nan_mask]

        if min_samples is None:
            min_samples = max(1, 2 * X.shape[1])

        if X.shape[0] == 0:
            return {"indices": [], "values": [], "labels": []}

        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X)

        outlier_positions = np.where(labels == -1)[0]
        outlier_idx_labels = idx_labels[outlier_positions].tolist()
        outlier_values = dataset.loc[outlier_idx_labels, cols].values.tolist()

        if verbose:
            print(f"[ DBSCAN ] eps={eps}, min_samples={min_samples}, detected {len(outlier_idx_labels)} outliers.")
            for lbl, val in zip(outlier_idx_labels, outlier_values):
                print(f"  Index: {lbl}, Values: {val}")

        return {
            "indices": outlier_idx_labels,
            "values": outlier_values,
            "labels": labels.tolist()
        }

