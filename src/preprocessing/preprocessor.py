import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from typing import Optional, List, Any


class ADNIPreprocessor(BaseEstimator, TransformerMixin):
    """
    ADNIPreprocessor that reproduces preprocessing logic from the Data Preprocessing Notebook:
    - Detect integer-like columns and optionally force conversion.
    - Compute train mean/std to scale data before KNN imputation and inverse-scale after.
    - Convert float columns that represent integers using configurable rounding.
    - Create safe ratio columns preserving NaNs and avoiding divide-by-zero.
    - Normalize selected MRI measures by ICV and drop originals.
    - Optionally perform hybrid sampling: RandomUnderSampler then SMOTENC.
    """
    # ----------------------------#
    #        INITIALIZATION       #
    # ----------------------------# 
    def __init__(self, random_state: int = 42):
        # Configuration
        self.k = 5
        self.random_state = int(random_state)

        # Attributes set in fit
        self.int_columns_: List[str] = []
        self.numeric_cols_: List[str] = []
        self.means_: Optional[np.ndarray] = None
        self.stds_: Optional[np.ndarray] = None
        self.knn_: Optional[KNNImputer] = None
        self.feature_names_out_: Optional[List[str]] = None

        # Default int columns
        self.force_int_columns: List[str] = [
            'AGE', 'APOE4', 'ADAS13', 'ADAS11', 'ADASQ4', 'MMSE', 'RAVLT_immediate',
            'RAVLT_learning', 'RAVLT_forgetting', 'LDELTOTAL', 'TRABSCOR', 'FAQ',
            'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp', 'ICV', 'MOCA',
            'PTGENDER', 'PTEDUCAT', 'PTDEMOGROUP', 'MARRIED'
        ]
        self.categorical_columns: List[str] = ['PTGENDER', 'APOE4']
        self.mri_measures: List[str] = ['Ventricles', 'Hippocampus', 'Entorhinal', 'Fusiform', 'MidTemp', 'WholeBrain']


    # ----------------------------#
    #          UTILITIES          #
    # ----------------------------# 
    def _ensure_df(self, X: Any) -> pd.DataFrame:
        """Ensure X is a pandas DataFrame and keep column order consistent with fit when possible."""
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if self.feature_names_out_ is not None:
            return pd.DataFrame(X, columns=self.feature_names_out_).copy()
        return pd.DataFrame(X).copy()

    @staticmethod
    def _is_float_column_integral(s: pd.Series) -> bool:
        """Return True if non-missing values are floats representing integers."""
        if s.dropna().empty:
            return False
        vals = pd.to_numeric(s.dropna(), errors='coerce')
        is_numeric = vals.notna().all()
        if not is_numeric:
            return False
        frac = (vals % 1).abs()
        return (frac < 1e-8).all()

    def _detect_integer_like(self, X: pd.DataFrame) -> List[str]:
        """Detect columns that are integer-like (float values equal to integers) or already integer dtype."""
        floats = [c for c in X.select_dtypes(include=['float']).columns if self._is_float_column_integral(X[c])]
        ints = list(X.select_dtypes(include=['int']).columns)
        forced = [c for c in self.force_int_columns if c in X.columns]
        seen = {}
        for c in floats + ints + forced:
            seen[c] = None
        return list(seen.keys())

    def _convert_float_to_int(self, dataset: pd.DataFrame, column: str, method: str = "round", fillna_value: int = 0) -> None:
        """Convert column in dataset from float-like to np.int64 using chosen rounding."""
        if column not in dataset.columns:
            return
        col = pd.to_numeric(dataset[column], errors='coerce').copy()
        if method == "floor":
            col = np.floor(col)
        elif method == "ceil":
            col = np.ceil(col)
        elif method == "round":
            frac = col - np.floor(col)
            col = np.where(frac < 0.5, np.floor(col), np.ceil(col))
        else:
            raise ValueError(f"Unknown method '{method}' for integer conversion.")
        dataset[column] = pd.Series(col, index=dataset.index).fillna(fillna_value).astype(np.int64)

    def _create_ratio(self, dataset: pd.DataFrame, num_col: str, den_col: str, new_col: Optional[str] = None) -> None:
        """Safely create ratio column = num_col / den_col. Preserve NaNs and avoid divide-by-zero."""
        if num_col not in dataset.columns or den_col not in dataset.columns:
            return
        if new_col is None:
            new_col = f"{num_col}_over_{den_col}"
        num = pd.to_numeric(dataset[num_col], errors='coerce')
        den = pd.to_numeric(dataset[den_col], errors='coerce')
        safe = num.notna() & den.notna() & (den != 0)
        out = pd.Series(np.nan, index=dataset.index, dtype=float)
        if safe.any():
            out.loc[safe] = (num.loc[safe].astype(float) / den.loc[safe].astype(float)).values
        dataset[new_col] = out

    def _knn_impute(self, dataset: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Scaled KNN imputation: scale, impute, inverse-scale, write back numeric columns."""
        if self.knn_ is None or self.means_ is None or self.stds_ is None or not numeric_cols:
            return dataset
        cols = [c for c in numeric_cols if c in dataset.columns]
        if not cols:
            return dataset
        block = dataset[cols].apply(pd.to_numeric, errors='coerce').astype(float)
        idxs = [self.numeric_cols_.index(c) for c in cols]
        means_aligned = self.means_[idxs]
        stds_aligned = self.stds_[idxs]
        scaled = (block.values - means_aligned) / stds_aligned
        imputed_scaled = self.knn_.transform(scaled)
        imputed = (imputed_scaled * stds_aligned) + means_aligned
        imputed_df = pd.DataFrame(imputed, index=dataset.index, columns=cols)
        for c in cols:
            if c in self.int_columns_ or c in self.force_int_columns:
                dataset[c] = pd.Series(imputed_df[c].round().fillna(0).astype(np.int64), index=dataset.index)
            else:
                dataset[c] = imputed_df[c].astype(float)
        return dataset

    # ----------------------------#
    #         FIT/TRANSFORM       #
    # ----------------------------# 
    def fit(self, X: pd.DataFrame, y: Any = None):
        """Compute train statistics (means/stds) for numeric columns and detect integer-like columns."""
        X_train = self._ensure_df(X)
        self.int_columns_ = self._detect_integer_like(X_train)
        self.numeric_cols_ = [c for c in X_train.columns if X_train[c].dtype.kind in 'fi']
        if not self.numeric_cols_:
            self.means_ = np.array([])
            self.stds_ = np.array([])
            self.knn_ = KNNImputer(n_neighbors=self.k)
            self.feature_names_out_ = list(X_train.columns)
            return self
        train_block = X_train[self.numeric_cols_].astype(float)
        self.means_ = np.nanmean(train_block.values, axis=0)
        self.stds_ = np.nanstd(train_block.values, axis=0)
        self.stds_[self.stds_ == 0] = 1.0
        train_scaled = (train_block.values - self.means_) / self.stds_
        self.knn_ = KNNImputer(n_neighbors=self.k)
        self.knn_.fit(train_scaled)
        self.feature_names_out_ = list(X_train.columns)
        return self

    def transform(self, X: pd.DataFrame, y: Optional[Any] = None) -> pd.DataFrame:
        """Apply preprocessing pipeline and return a DataFrame with correct feature names."""
        X_train = self._ensure_df(X)
        if self.numeric_cols_:
            X_train = self._knn_impute(X_train, self.numeric_cols_)
        for col in self.int_columns_:
            if col in X_train.columns:
                method = 'floor' if col == 'AGE' else 'round'
                self._convert_float_to_int(X_train, col, method=method, fillna_value=0)
        for col in self.force_int_columns:
            if col in X_train.columns and X_train[col].dtype.kind not in 'iu':
                self._convert_float_to_int(X_train, col, method='round', fillna_value=0)
        
        if {'TAU', 'PTAU', 'ABETA'}.issubset(X_train.columns):
            self._create_ratio(X_train, 'TAU', 'ABETA', 'TAU/ABETA')
            X_train.drop(columns=['TAU'], inplace=True, errors='ignore')
            self._create_ratio(X_train, 'PTAU', 'ABETA', 'PTAU/ABETA')
            X_train.drop(columns=['PTAU', 'ABETA'], inplace=True, errors='ignore')
        elif {'TAU', 'ABETA'}.issubset(X_train.columns):
            self._create_ratio(X_train, 'TAU', 'ABETA', 'TAU/ABETA')
            X_train.drop(columns=['TAU', 'ABETA'], inplace=True, errors='ignore')
        elif {'PTAU', 'ABETA'}.issubset(X_train.columns):
            self._create_ratio(X_train, 'PTAU', 'ABETA', 'PTAU/ABETA')
            X_train.drop(columns=['PTAU', 'ABETA'], inplace=True, errors='ignore')
        
        if 'ICV' in X_train.columns:
            for m in self.mri_measures:
                if m in X_train.columns:
                    self._create_ratio(X_train, m, 'ICV', f'{m}/ICV')
            drop_cols = [c for c in (self.mri_measures + ['ICV']) if c in X_train.columns]
            if drop_cols:
                X_train.drop(columns=drop_cols, inplace=True, errors='ignore')
        redundant = [
            'ADAS11', 'ADASQ4', 'EcogPtTotal', 'EcogSPTotal',
            'mPACCtrailsB', 'MARRIED', 'PTDEMOGROUP', 'RAVLT_forgetting'
        ]
        to_drop = [c for c in redundant if c in X_train.columns]
        if to_drop:
            X_train.drop(columns=to_drop, inplace=True, errors='ignore')
        self.feature_names_out_ = list(X_train.columns)
        
        # Define the final desired column order
        desired_cols = [
            "AGE", "PTGENDER", "PTEDUCAT", "APOE4", "MMSE", "CDRSB", "ADAS13", "LDELTOTAL",
            "FAQ", "MOCA", "TRABSCOR", "RAVLT_immediate", "RAVLT_learning", "RAVLT_perc_forgetting",
            "mPACCdigit", "EcogPtMem", "EcogPtLang", "EcogPtVisspat", "EcogPtPlan", "EcogPtOrgan",
            "EcogPtDivatt", "EcogSPMem", "EcogSPLang", "EcogSPVisspat", "EcogSPPlan", "EcogSPOrgan",
            "EcogSPDivatt", "FDG", "TAU/ABETA", "PTAU/ABETA", "Hippocampus/ICV", "Entorhinal/ICV",
            "Fusiform/ICV", "MidTemp/ICV", "Ventricles/ICV", "WholeBrain/ICV"
        ]

        # Reindex the DataFrame to match the desired column order
        X_train = X_train.reindex(columns=desired_cols)

        # Identify columns that were newly created (i.e., all NaN values)
        newly_created_cols = [col for col in desired_cols if X_train[col].isna().all()]

        # Drop these fully-NaN columns immediately to avoid propagating missing values
        if newly_created_cols:
            X_train.drop(columns=newly_created_cols, axis=1, inplace=True)
        
        return X_train

    def fit_transform(self, X: pd.DataFrame, y: Optional[Any] = None) -> pd.DataFrame:
        """Convenience: fit then transform and return a DataFrame."""
        self.fit(X, y)
        return self.transform(X, y=y)

    def get_feature_names_out(self) -> List[str]:
        """Return final feature names after transform (or those seen at fit)."""
        return list(self.feature_names_out_) if self.feature_names_out_ is not None else []
