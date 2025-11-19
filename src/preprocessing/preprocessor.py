import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from typing import Optional, List, Any


class ADNIPreprocessor(BaseEstimator, TransformerMixin):
    """
    ADNIPreprocessor
    ----------------
    Reproduces preprocessing logic used in the Data Preprocessing Notebook for ADNI-style data.
    Key Features:
      - Detect columns that are integer-like and optionally coerce them to integer dtype.
      - Compute train set mean/std for numeric features and use them to scale before KNN imputation.
      - Perform KNN imputation in the scaled space and inverse-scale the imputed values.
      - Convert float columns that represent integers according to configurable rounding rules.
      - Create safe ratio features (preserving NaNs and avoiding division-by-zero).
      - Normalize selected MRI measures by ICV and remove the original MRI columns.
      - Drop a set of predetermined redundant columns.
    Usage:
      - The transformer is stateful: 'fit' computes means/stds and fits the KNN imputer,
        while 'transform' applies imputation, conversions, ratios and column ordering.
      - 'fit_transform' is provided for convenience.
    """
    # ----------------------------#
    #        INITIALIZATION       #
    # ----------------------------#
    def __init__(self, k: int = 5, random_state: int = 42):
        # User-configurable parameters
        self.k = k
        self.random_state = random_state

        # Attributes filled during fit()
        self.int_columns_: List[str] = []           # columns detected as integer-like
        self.numeric_cols_: List[str] = []          # numeric columns considered for scaling/imputation
        self.means_: Optional[np.ndarray] = None    # per-column means (train)
        self.stds_: Optional[np.ndarray] = None     # per-column std devs (train)
        self.knn_: Optional[KNNImputer] = None      # fitted KNNImputer (on scaled training block)
        self.feature_names_out_: Optional[List[str]] = None  # names after transform

        # Defaults and domain knowledge
        # Columns that should be forced to integer dtype when present
        self.force_int_columns: List[str] = [
            'AGE', 'APOE4', 'ADAS13', 'ADAS11', 'ADASQ4', 'MMSE', 'RAVLT_immediate',
            'RAVLT_learning', 'RAVLT_forgetting', 'LDELTOTAL', 'TRABSCOR', 'FAQ',
            'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp', 'ICV', 'MOCA',
            'PTGENDER', 'PTEDUCAT', 'PTDEMOGROUP', 'MARRIED'
        ]
        # Columns treated as categorical for downstream logic (e.g., sampling)
        self.categorical_columns: List[str] = ['PTGENDER', 'APOE4']
        # MRI regional measures to normalize by ICV if present
        self.mri_measures: List[str] = ['Ventricles', 'Hippocampus', 'Entorhinal', 'Fusiform', 'MidTemp', 'WholeBrain']


    # ----------------------------#
    #          UTILITIES          #
    # ----------------------------#
    def _ensure_df(self, X: Any) -> pd.DataFrame:
        """
        Ensure input X is returned as a pandas DataFrame.

        If X is already a DataFrame: return a shallow copy to avoid mutating caller data.
        If feature_names_out_ exists (from fit), attempt to preserve column ordering by
        constructing a DataFrame with those column names.
        Otherwise construct a DataFrame without explicit column names.
        """
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if self.feature_names_out_ is not None:
            return pd.DataFrame(X, columns=self.feature_names_out_).copy()
        return pd.DataFrame(X).copy()

    @staticmethod
    def _is_float_column_integral(s: pd.Series) -> bool:
        """
        Check whether a float-typed column only contains values that are mathematically integers.
        - Returns False for non-numeric values or empty (all-missing) columns.
        - Uses a small tolerance (1e-8) to account for floating-point representation.
        """
        if s.dropna().empty:
            return False
        vals = pd.to_numeric(s.dropna(), errors='coerce')
        is_numeric = vals.notna().all()
        if not is_numeric:
            return False
        frac = (vals % 1).abs()
        return (frac < 1e-8).all()

    def _detect_integer_like(self, X: pd.DataFrame) -> List[str]:
        """
        Detect columns that are integer-like to be coerced later.
        - Considers float columns whose non-missing values are integer-valued.
        - Also includes actual int dtypes and any columns explicitly listed in force_int_columns.
        - Returns a stable list preserving first-seen order and removing duplicates.
        """
        floats = [c for c in X.select_dtypes(include=['float']).columns if self._is_float_column_integral(X[c])]
        ints = list(X.select_dtypes(include=['int']).columns)
        forced = [c for c in self.force_int_columns if c in X.columns]
        seen = {}
        for c in floats + ints + forced:
            seen[c] = None
        return list(seen.keys())

    def _convert_float_to_int(self, dataset: pd.DataFrame, column: str, method: str = "round", fillna_value: int = 0) -> None:
        """
        Convert a float-like column into integer dtype using a chosen strategy.
        - method: 'floor' | 'ceil' | 'round'
            - 'round' uses round-half-up behavior implemented with fractional check.
        - Missing values are coerced to `fillna_value`.
        - Operates in-place on the provided DataFrame.
        """
        if column not in dataset.columns:
            return
        col = pd.to_numeric(dataset[column], errors='coerce').copy()
        if method == "floor":
            col = np.floor(col)
        elif method == "ceil":
            col = np.ceil(col)
        elif method == "round":
            # explicit half-up rounding: fractional part < 0.5 -> floor else ceil
            frac = col - np.floor(col)
            col = np.where(frac < 0.5, np.floor(col), np.ceil(col))
        else:
            raise ValueError(f"Unknown method '{method}' for integer conversion.")
        dataset[column] = pd.Series(col, index=dataset.index).fillna(fillna_value).astype(np.int64)

    def _create_ratio(self, dataset: pd.DataFrame, num_col: str, den_col: str, new_col: Optional[str] = None) -> None:
        """
        Safely create a ratio column = num_col / den_col.
        - Preserves NaNs: only compute ratio when both numerator and denominator are non-missing and denominator != 0.
        - If new_col is None, name it "{num_col}_over_{den_col}".
        - Writes the new column into `dataset` in-place as float dtype.
        """
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
        """
        Perform KNN imputation on numeric columns using the KNN fitted in fit():
        - Align the requested numeric columns with the order used during fit (self.numeric_cols_).
        - Scale the block using stored means_ and stds_ (train statistics).
        - Apply the fitted KNN imputer to the scaled block.
        - Inverse-scale the imputed values, coerce integer columns back to integer dtype (rounding).
        - Return the dataset modified in-place (but also return reference).
        Notes:
          - If KNN or training statistics are not available, this is a no-op.
          - Columns not present in dataset are skipped.
        """
        if self.knn_ is None or self.means_ is None or self.stds_ is None or not numeric_cols:
            return dataset
        cols = [c for c in numeric_cols if c in dataset.columns]
        if not cols:
            return dataset
        block = dataset[cols].apply(pd.to_numeric, errors='coerce').astype(float)
        idxs = [self.numeric_cols_.index(c) for c in cols]
        means_aligned = self.means_[idxs]
        stds_aligned = self.stds_[idxs]
        # Scale, impute, inverse-scale
        scaled = (block.values - means_aligned) / stds_aligned
        imputed_scaled = self.knn_.transform(scaled)
        imputed = (imputed_scaled * stds_aligned) + means_aligned
        imputed_df = pd.DataFrame(imputed, index=dataset.index, columns=cols)
        # Write back, rounding integer-like columns
        for c in cols:
            if c in self.int_columns_ or c in self.force_int_columns:
                dataset[c] = pd.Series(imputed_df[c].round().fillna(0).astype(np.int64), index=dataset.index)
            else:
                dataset[c] = imputed_df[c].astype(float)
        return dataset

    # ----------------------------#
    #        FIT/TRANSFORM        #
    # ----------------------------#
    def fit(self, X: pd.DataFrame, y: Any = None):
        """
        Fit the preprocessor on the training data:
          - Detect integer-like columns.
          - Identify numeric columns for scaling/imputation.
          - Compute per-column means and stds (train statistics).
          - Fit a KNN imputer on the scaled numeric block.
        After fit, the transformer stores means_, stds_, knn_, numeric_cols_, int_columns_
        and feature_names_out_ for use during transform.
        """
        X_train = self._ensure_df(X)
        # Detect integer-like columns (float columns that contain integer values)
        self.int_columns_ = self._detect_integer_like(X_train)
        # Numeric columns considered for scaling/imputation (floats and ints)
        self.numeric_cols_ = [c for c in X_train.columns if X_train[c].dtype.kind in 'fi']
        if not self.numeric_cols_:
            # No numeric columns: set empty attributes and a default KNN (no-op but consistent)
            self.means_ = np.array([])
            self.stds_ = np.array([])
            self.knn_ = KNNImputer(n_neighbors=self.k)
            self.feature_names_out_ = list(X_train.columns)
            return self
        train_block = X_train[self.numeric_cols_].astype(float)
        # Compute column-wise train mean/std (ignore NaNs)
        self.means_ = np.nanmean(train_block.values, axis=0)
        self.stds_ = np.nanstd(train_block.values, axis=0)
        # Avoid division by zero during scaling
        self.stds_[self.stds_ == 0] = 1.0
        train_scaled = (train_block.values - self.means_) / self.stds_
        # fit KNN on scaled training block
        self.knn_ = KNNImputer(n_neighbors=self.k)
        self.knn_.fit(train_scaled)
        self.feature_names_out_ = list(X_train.columns)
        return self

    def transform(self, X: pd.DataFrame, y: Optional[Any] = None) -> pd.DataFrame:
        """
        Apply preprocessing to a dataset:
          - Ensure DataFrame input and preserve fitted feature ordering if available.
          - Apply KNN imputation (scaled) to numeric columns.
          - Convert detected integer-like columns to integer dtype (AGE uses floor; others use round).
          - Force conversion of columns in force_int_columns if present and not already integer dtype.
          - Create biomarker ratios (TAU/ABETA, PTAU/ABETA) when the corresponding columns exist.
          - Normalize MRI measures by ICV and drop original MRI columns and ICV.
          - Drop a predefined list of redundant columns.
          - Reindex DataFrame to a desired column order; drop any newly created columns that are all NaN.
        Returns:
            DataFrame containing the transformed features (may have fewer columns than the desired list).
        """
        X_train = self._ensure_df(X)
        # Impute numeric columns if available
        if self.numeric_cols_:
            X_train = self._knn_impute(X_train, self.numeric_cols_)
        # Coerce integer-like columns to integers; AGE uses floor, others use round
        for col in self.int_columns_:
            if col in X_train.columns:
                method = 'floor' if col == 'AGE' else 'round'
                self._convert_float_to_int(X_train, col, method=method, fillna_value=0)
        # Ensure forced integer columns are integer-typed
        for col in self.force_int_columns:
            if col in X_train.columns and X_train[col].dtype.kind not in 'iu':
                self._convert_float_to_int(X_train, col, method='round', fillna_value=0)
        
        # Create biomarker ratios and drop originals according to availability
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
        
        # Normalize selected MRI measures by ICV and drop the originals
        if 'ICV' in X_train.columns:
            for m in self.mri_measures:
                if m in X_train.columns:
                    self._create_ratio(X_train, m, 'ICV', f'{m}/ICV')
            # Remove raw MRI measures and ICV once normalized ratios are created
            drop_cols = [c for c in (self.mri_measures + ['ICV']) if c in X_train.columns]
            if drop_cols:
                X_train.drop(columns=drop_cols, inplace=True, errors='ignore')
        # Drop predetermined redundant columns to match notebook behavior
        redundant = [
            'ADAS11', 'ADASQ4', 'EcogPtTotal', 'EcogSPTotal',
            'mPACCtrailsB', 'MARRIED', 'PTDEMOGROUP', 'RAVLT_forgetting'
        ]
        to_drop = [c for c in redundant if c in X_train.columns]
        if to_drop:
            X_train.drop(columns=to_drop, inplace=True, errors='ignore')
        self.feature_names_out_ = list(X_train.columns)
        
        # Define a desired column order matching the notebook's canonical feature list.
        desired_cols = [
            "AGE", "PTGENDER", "PTEDUCAT", "APOE4", "MMSE", "CDRSB", "ADAS13", "LDELTOTAL",
            "FAQ", "MOCA", "TRABSCOR", "RAVLT_immediate", "RAVLT_learning", "RAVLT_perc_forgetting",
            "mPACCdigit", "EcogPtMem", "EcogPtLang", "EcogPtVisspat", "EcogPtPlan", "EcogPtOrgan",
            "EcogPtDivatt", "EcogSPMem", "EcogSPLang", "EcogSPVisspat", "EcogSPPlan", "EcogSPOrgan",
            "EcogSPDivatt", "FDG", "TAU/ABETA", "PTAU/ABETA", "Hippocampus/ICV", "Entorhinal/ICV",
            "Fusiform/ICV", "MidTemp/ICV", "Ventricles/ICV", "WholeBrain/ICV"
        ]

        # Reindex to the canonical order. Missing columns will appear as all-NaN.
        X_train = X_train.reindex(columns=desired_cols)

        # Identify columns created by reindexing that are entirely NaN and drop them immediately.
        newly_created_cols = [col for col in desired_cols if X_train[col].isna().all()]
        if newly_created_cols:
            X_train.drop(columns=newly_created_cols, axis=1, inplace=True)
        
        return X_train

    def fit_transform(self, X: pd.DataFrame, y: Optional[Any] = None) -> pd.DataFrame:
        """Fit followed by transform. Returns the transformed DataFrame."""
        self.fit(X, y)
        return self.transform(X, y=y)

    def get_feature_names_out(self) -> List[str]:
        """Return the output feature names discovered at fit/transform time (safe copy)."""
        return list(self.feature_names_out_) if self.feature_names_out_ is not None else []
