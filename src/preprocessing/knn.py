import numpy as np
import pandas as pd 

from sklearn.impute import KNNImputer

def knn_impute_group(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: list, n_neighbors: int = 5):
    """
    Fit a KNNImputer on train_df[cols] after scaling (train mean/std),
    transform both train and test for these cols, and replace the values in-place.

    Steps:
      1. Keep only columns present in train_df.
      2. Coerce non-numeric values to NaN (safe conversion).
      3. Compute train mean/std ignoring NaNs (nan-aware).
      4. Standardize train and test using train statistics.
      5. Fit KNNImputer on scaled train and transform both train and test.
      6. Inverse-scale imputed results back to the original scale.
      7. Put imputed columns back into the original DataFrames preserving indices.

    Returns:
      (knn_imputer_object, means_array, stds_array) or (None, None, None) if no cols.
    """
    # keep only columns that actually exist in train_df
    cols = [c for c in cols if c in train_df.columns]
    if len(cols) == 0:
        return None, None, None

    # Coerce to numeric: non-convertible values become NaN
    # Use copy() to avoid modifying original slices unexpectedly
    train_block = train_df[cols].apply(pd.to_numeric, errors="coerce").astype(float).copy()
    test_block = test_df[cols].apply(pd.to_numeric, errors="coerce").astype(float).copy()

    # Compute mean/std on train, ignoring NaNs
    means = np.nanmean(train_block.values, axis=0)
    stds = np.nanstd(train_block.values, axis=0)
    # Prevent division by zero for constant columns
    stds[stds == 0] = 1.0

    # Standardize (nan-aware)
    train_scaled = (train_block.values - means) / stds
    test_scaled = (test_block.values - means) / stds

    # Fit KNNImputer on scaled train and transform both train and test
    knn = KNNImputer(n_neighbors=n_neighbors)
    imputed_train_scaled = knn.fit_transform(train_scaled)
    imputed_test_scaled = knn.transform(test_scaled)

    # Inverse scale back to original units
    imputed_train = imputed_train_scaled * stds + means
    imputed_test = imputed_test_scaled * stds + means

    # Put the imputed arrays back into the original DataFrames (preserve indices)
    train_df.loc[:, cols] = pd.DataFrame(imputed_train, index=train_df.index, columns=cols)
    test_df.loc[:, cols] = pd.DataFrame(imputed_test, index=test_df.index, columns=cols)

    print("K-Nearest Neighbors imputation applied...")
    return knn, means, stds