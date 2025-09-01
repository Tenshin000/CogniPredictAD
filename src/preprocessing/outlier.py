import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import zscore
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from typing import Optional, List, Dict


class OutlierDetector:
    """
    A class to perform outlier detection on a dataset.
    """

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
    

    def plausible_min(self, dataset: pd.DataFrame, column: str, 
                    method: str = 'iqr', 
                    iqr_factor: float = 1.5, 
                    z_threshold: float = 3.0) -> float:
        """
        Calculate the plausible minimum of a numeric column using IQR, Z-score, or both.

        param dataset: DataFrame containing the data.
        param column: Name of the column to analyze.
        param method: 'iqr', 'zscore', or 'both' (default='iqr')
        param iqr_factor: Multiplicative factor for the IQR (default=1.5)
        param z_threshold: Z-score threshold (default=3.0)
        return: plausible minimum value
        """
        if column not in dataset.columns:
            raise ValueError(f"Column '{column}' not found in dataset.")
        
        col_data = dataset[column].dropna()
        min_vals = {}

        # IQR-based plausible minimum
        if method in ('iqr', 'both'):
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            min_vals['iqr'] = Q1 - iqr_factor * IQR
        
        # Z-score-based plausible minimum
        if method in ('zscore', 'both'):
            mean = col_data.mean()
            std = col_data.std()
            min_vals['zscore'] = mean - z_threshold * std
        
        # Select the most conservative value if both methods are used
        plausible_min_value = max(min_vals.values())

        # Output which minima were estimated and which one is returned
        print(f"[Plausible Minimum] Column: {column}")
        for k, v in min_vals.items():
            print(f"  Estimated minimum by {k.upper()}: {v:.6f}")
        print(f"  Returned plausible minimum: {plausible_min_value:.6f}")

        return plausible_min_value


    def detect_by_iqr(self, dataset: pd.DataFrame, columns: Optional[List[str]] = None,
                  factor: float = 1.5, verbose: bool = True) -> Dict[str, Dict[str, List]]:
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

            # Grafico sempre mostrato
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=dataset[col], color='skyblue')
            plt.title(f'Boxplot for {col} (IQR)')
            plt.show()

        return results


    def detect_by_zscore(self, dataset: pd.DataFrame, columns: Optional[List[str]] = None,
                     threshold: float = 3.0, verbose: bool = True) -> Dict[str, Dict[str, List]]:
        cols = self._get_columns(dataset, columns)
        results = {}
        z_scores = np.abs(zscore(dataset[cols], nan_policy='omit'))

        for idx, col in enumerate(cols):
            mask = z_scores[:, idx] > threshold
            outlier_indices = np.where(mask)[0].tolist()
            outlier_values = dataset.iloc[outlier_indices][col].tolist()

            results[col] = {
                "indices": outlier_indices,
                "values": outlier_values
            }

            if verbose:
                print(f"[ Z-score Method ] Column: {col}")
                print(f"Threshold: {threshold}")
                print(f"Detected {len(outlier_indices)} outliers.")
                for i, val in zip(outlier_indices, outlier_values):
                    print(f"  Index: {i}, Value: {val}")

        return results


    def detect_by_lof(self, dataset: pd.DataFrame, columns: Optional[List[str]] = None,
                  n_neighbors_list: List[int] = [20], verbose: bool = True) -> Dict[int, Dict[str, List]]:
        cols = self._get_columns(dataset, columns)
        X = dataset[cols].fillna(0)
        results = {}

        for n_neighbors in n_neighbors_list:
            lof = LocalOutlierFactor(n_neighbors=n_neighbors)
            y_pred = lof.fit_predict(X)
            scores = -lof.negative_outlier_factor_

            outlier_indices = np.where(y_pred == -1)[0].tolist()
            outlier_values = dataset.iloc[outlier_indices][cols].values.tolist()

            results[n_neighbors] = {
                "indices": outlier_indices,
                "values": outlier_values,
                "scores": scores[outlier_indices].tolist()
            }

            if verbose:
                print(f"[ LOF Method ] n_neighbors={n_neighbors}")
                print(f"Detected {len(outlier_indices)} outliers.")
                for idx, val, sc in zip(outlier_indices, outlier_values, scores[outlier_indices]):
                    print(f"  Index: {idx}, Values: {val}, Score: {sc:.4f}")

            plt.figure(figsize=(6, 4))
            plt.scatter(range(len(scores)), scores, c='blue', s=20, edgecolor='k')
            plt.scatter(outlier_indices, np.array(scores)[outlier_indices], c='red', s=30, label='Outliers')
            plt.xlabel('Data Point Index')
            plt.ylabel('LOF Score')
            plt.title(f'Local Outlier Factor Scores (n_neighbors={n_neighbors})')
            plt.legend()
            plt.show()

        return results

    def detect_by_dbscan(self, dataset: pd.DataFrame, columns: Optional[List[str]] = None,
                     eps: float = 0.5, min_samples: Optional[int] = None, verbose: bool = True) -> Dict[str, List]:
        cols = self._get_columns(dataset, columns)
        X = dataset[cols].fillna(0)

        if min_samples is None:
            min_samples = 2 * X.shape[1]

        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X)

        outlier_indices = np.where(labels == -1)[0].tolist()
        outlier_values = dataset.iloc[outlier_indices][cols].values.tolist()

        if verbose:
            print(f"[ DBSCAN Method ] eps={eps}, min_samples={min_samples}")
            print(f"Detected {len(outlier_indices)} outliers.")
            for idx, val in zip(outlier_indices, outlier_values):
                print(f"  Index: {idx}, Values: {val}")

        # 2D graph if possible
        if X.shape[1] == 2:
            plt.figure(figsize=(6, 4))
            plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis', s=20)
            plt.title(f'DBSCAN Clustering (eps={eps}, min_samples={min_samples})')
            plt.show()

        return {
            "indices": outlier_indices,
            "values": outlier_values,
            "labels": labels.tolist()
        }

