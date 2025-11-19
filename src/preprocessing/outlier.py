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
        Detect outliers using the Interquartile Range (IQR) method on each column.

        For each column:
          - Compute Q1, Q3 and IQR = Q3 - Q1.
          - Define lower = Q1 - factor*IQR and upper = Q3 + factor*IQR.
          - Report indices and values falling outside [lower, upper].
          - Display a boxplot (matplotlib/seaborn) for the column.

        Parameters:
            dataset: DataFrame with data to analyze.
            columns: Optional list of column names to analyze. If None, numeric columns are used.
            factor: Multiplicative factor for the IQR (default 1.5).
            verbose: If True, print bounds and each detected outlier.

        Returns:
            dict mapping column -> {"indices": [...], "values": [...]}
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

            # Visual summary: seaborn boxplot (shows median, quartiles, and whiskers)
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=dataset[col], color='skyblue')
            plt.title(f'Boxplot for {col} (IQR)')
            plt.show()

        return results


    def detect_by_zscore(self, dataset: pd.DataFrame, columns: Optional[List[str]] = None,
                         threshold: float = 3.0, verbose: bool = True) -> Dict[str, Dict[str, List]]:
        """
        Detect outliers using a Z-score threshold per column.

        Implementation notes:
          - Uses column-wise mean and population standard deviation (ddof=0).
          - If a column has zero variance (std == 0) or std is NaN, no outliers are returned.
          - Z-score computed as absolute((value - mean) / std). Values above `threshold` are flagged.

        Parameters:
            dataset: DataFrame to analyze.
            columns: Optional list of columns, default = numeric columns.
            threshold: Z-score cut-off (default 3.0).
            verbose: If True, prints per-column counts and indices of detected outliers.

        Returns:
            dict mapping column -> {"indices": [...], "values": [...]}
        """
        cols = self._get_columns(dataset, columns)
        results = {}

        # Compute z-scores column-wise using pandas to have clearer NaN behavior
        for col in cols:
            ser = dataset[col]
            mean = ser.mean(skipna=True)
            std = ser.std(skipna=True, ddof=0)
            if std == 0 or np.isnan(std):
                # No variation -> No Outliers
                outlier_mask = pd.Series(False, index=ser.index)
            else:
                z = (ser - mean).abs() / std
                outlier_mask = z > threshold

            outlier_idx_labels = ser.index[outlier_mask].tolist()  # original labels
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
        Detect outliers using Local Outlier Factor (LOF).

        Implementation details:
          - Select analysis columns via _get_columns.
          - Optionally impute missing values:
              - None: drop rows with any NaN (LOF requires complete observations).
              - 'zero': fill missing with 0.
              - 'median': fill missing with column medians.
          - Fit sklearn.neighbors.LocalOutlierFactor and compute negative_outlier_factor_
            (larger positive values of -negative_outlier_factor_ mean more outlying).
          - Points with LOF prediction -1 are considered outliers.

        Parameters:
            dataset: DataFrame to analyze.
            columns: Optional list of feature columns (defaults to numeric).
            n_neighbors: LOF neighborhood size (default 20). Will be reduced if >= n_samples.
            impute_strategy: None | 'zero' | 'median' (controls simple imputation behavior).
            verbose: If True prints summary and each detected outlier's score.

        Returns:
            dict with keys:
              - "indices": list of original index labels flagged as outliers,
              - "values": list of the associated feature vectors (lists),
              - "scores": LOF anomaly scores for the detected outliers.
        """
        cols = self._get_columns(dataset, columns)
        X_df = dataset[cols].copy()

        # Optional simple imputation (user can also pre-clean dataset)
        if impute_strategy == 'zero':
            X = X_df.fillna(0).values
            idx_labels = X_df.index
        elif impute_strategy == 'median':
            X = X_df.fillna(X_df.median()).values
            idx_labels = X_df.index
        else:
            # Do not impute: drop rows with NaN for LOF (but remember original indices)
            not_nan_mask = ~X_df.isna().any(axis=1)
            X = X_df[not_nan_mask].values
            idx_labels = X_df.index[not_nan_mask]

        n_samples = X.shape[0]

        if n_samples == 0:
            # No usable rows after imputation/drop: return empty result
            return {"indices": [], "values": [], "scores": []}

        # Ensure n_neighbors < n_samples (LOF requires at most n_samples - 1 neighbors)
        if n_neighbors >= n_samples:
            n_neighbors_eff = max(1, n_samples - 1)
            if verbose:
                print(f"Warning: requested n_neighbors={n_neighbors} >= n_samples={n_samples}. "
                      f"Using n_neighbors={n_neighbors_eff} instead.")
            n_neighbors = n_neighbors_eff

        lof = LocalOutlierFactor(n_neighbors=n_neighbors)
        # fit_predict returns 1 for inliers and -1 for outliers
        y_pred = lof.fit_predict(X)
        # negative_outlier_factor_ is negative. Invert sign so higher = more outlying.
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
        Detect outliers using DBSCAN clustering.

        Implementation details:
          - Select columns via _get_columns.
          - Optional simple imputation identical to LOF logic.
          - If min_samples is None, default to max(1, 2 * n_features).
          - Run DBSCAN. Points assigned label -1 are considered noise/outliers.
          - Returns both the list of outlier indices/values and the labels for all points.

        Parameters:
            dataset: DataFrame to analyze.
            columns: Optional list of columns (defaults to numeric columns).
            eps: DBSCAN epsilon parameter (neighborhood radius).
            min_samples: Minimum points in a neighborhood to form a cluster. If None,
                         set to 2 * n_features as a reasonable default.
            impute_strategy: None | 'zero' | 'median' controlling pre-processing for NaNs.
            verbose: If True, prints summary and detected outlier entries.

        Returns:
            dict with keys:
              - "indices": indices of points labeled -1 (outliers),
              - "values": corresponding feature vectors,
              - "labels": list of cluster labels for all considered points (len == n_rows_used).
        """
        cols = self._get_columns(dataset, columns)
        X_df = dataset[cols].copy()

        # Imputation choices
        if impute_strategy == 'zero':
            X = X_df.fillna(0).values
            idx_labels = X_df.index
        elif impute_strategy == 'median':
            X = X_df.fillna(X_df.median()).values
            idx_labels = X_df.index
        else:
            # Drop rows with NaN
            not_nan_mask = ~X_df.isna().any(axis=1)
            X = X_df[not_nan_mask].values
            idx_labels = X_df.index[not_nan_mask]

        if min_samples is None:
            # Default heuristic: at least two points per feature dimension
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