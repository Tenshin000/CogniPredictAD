import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder

class ADNITransformator:
    """
    A class for the Data Transformation of the ADNIMERGE dataset.
    """
    # ----------------------------#
    #        INITIALIZATION       #
    # ----------------------------# 
    def __init__(self, dataset: pd.DataFrame):
        """
        Initialize the normalizator with the given dataset.
        
        :param dataset: A pandas DataFrame loaded from ADNIMERGE.csv, already prefiltered as specified.
        """
        self.dataset = dataset.copy()

    # ----------------------------#
    #        MISCELLANEOUS        #
    # ----------------------------# 
    def create_ratio_column(self,
                            numerator_col: str,
                            denominator_col: str,
                            new_col_name: str = None,
                            dataset: pd.DataFrame = pd.NA):
        """
        Create a new column equal to numerator_col / denominator_col.
        - If either numerator or denominator is NaN, the ratio is set to NaN.
        - If denominator is zero, the ratio is set to NaN to avoid infinities.
        - If a column is not numeric it will be coerced to numeric (non-convertible values -> NaN).

        :param numerator_col: Column name used as numerator.
        :param denominator_col: Column name used as denominator.
        :param new_col_name: Optional name for the ratio column. If None, uses '{numerator}_over_{denominator}'.
        :param dataset: Optional external dataset to use.
        
        :return: DataFrame with the new ratio column added.
        """
        if dataset is not None and dataset is not pd.NA:
            self.dataset = dataset

        # Validate columns exist
        missing = [c for c in (numerator_col, denominator_col) if c not in self.dataset.columns]
        if missing:
            print(f"Column(s) not found: {missing}. Operation skipped.")
            return self.dataset

        # Determine new column name
        if new_col_name is None:
            new_col_name = f"{numerator_col}_over_{denominator_col}"

        # Ensure numeric: coerce non-numeric values to NaN with a warning
        for col in (numerator_col, denominator_col):
            if not np.issubdtype(self.dataset[col].dropna().dtype, np.number):
                print(f"Warning: column '{col}' is not numeric. Attempting to coerce to numeric (non-convertible -> NaN).")
                self.dataset[col] = pd.to_numeric(self.dataset[col], errors='coerce')

        # Compute masks
        num = self.dataset[numerator_col]
        den = self.dataset[denominator_col]

        # mask where either is NA or denominator is zero
        either_na_mask = num.isna() | den.isna()
        denom_zero_mask = (den == 0)

        # compute ratio safely (will produce inf if denom==0, so handle it)
        ratio = pd.Series(np.nan, index=self.dataset.index, dtype=float)
        safe_mask = (~either_na_mask) & (~denom_zero_mask)

        # perform division only where safe_mask is True
        ratio.loc[safe_mask] = (num.loc[safe_mask].astype(float) / den.loc[safe_mask].astype(float)).values

        # Assign new column safely
        self.dataset.loc[:, new_col_name] = ratio

        # Compute NaN counts and differences
        na_num = num.isna().sum()
        na_den = den.isna().sum()
        na_ratio = self.dataset[new_col_name].isna().sum()

        diff_vs_num = na_ratio - na_num
        diff_vs_den = na_ratio - na_den

        print(f"Ratio column '{new_col_name}' created from '{numerator_col}' / '{denominator_col}'.")
        print(f"NaNs -> {numerator_col}: {na_num}, {denominator_col}: {na_den}, {new_col_name}: {na_ratio}")
        print(f"'{new_col_name}' has {diff_vs_num} more NaN(s) than '{numerator_col}' and {diff_vs_den} more NaN(s) than '{denominator_col}'.")

        return self.dataset

    # ----------------------------#
    #        NORMALIZATION        #
    # ----------------------------# 
    def min_max_normalization(self, columns: list[str], dataset: pd.DataFrame = pd.NA):
        """
        Applies Min-Max normalization to the specified columns.

        :param columns: List of column names to normalize.
        :param dataset: Optional DataFrame to override self.dataset.
        :return: DataFrame with normalized columns.
        """      
        if dataset is not None and dataset is not pd.NA:
            self.dataset = dataset


        # Check columns
        missing = [col for col in columns if col not in self.dataset.columns]
        if missing:
            raise ValueError(f"The dataset must contain the following columns: {missing}")

        non_numeric = [col for col in columns if not pd.api.types.is_numeric_dtype(self.dataset[col])]
        if non_numeric:
            raise ValueError(f"The following columns are not numeric and cannot be normalized: {non_numeric}")

        scaler = MinMaxScaler()
        self.dataset[columns] = scaler.fit_transform(self.dataset[columns])

        print("Min-Max normalization applied to columns:")
        for col, (min_val, max_val) in zip(columns, scaler.data_min_, scaler.data_max_):
            print(f" - {col}: original min={min_val}, original max={max_val}")

        return self.dataset

    def z_score_normalization(self, columns: list[str], dataset: pd.DataFrame = pd.NA):
        """
        Applies Z-score normalization to the specified columns.

        :param columns: List of column names to normalize.
        :param dataset: Optional DataFrame to override self.dataset.
        :return: DataFrame with normalized columns.
        """
        if dataset is not None and dataset is not pd.NA:
            self.dataset = dataset

        # Check columns
        missing = [col for col in columns if col not in self.dataset.columns]
        if missing:
            raise ValueError(f"The dataset must contain the following columns: {missing}")

        non_numeric = [col for col in columns if not pd.api.types.is_numeric_dtype(self.dataset[col])]
        if non_numeric:
            raise ValueError(f"The following columns are not numeric and cannot be normalized: {non_numeric}")

        scaler = StandardScaler()
        self.dataset[columns] = scaler.fit_transform(self.dataset[columns])

        print("Z-score normalization applied to columns:")
        for col, mean_val, std_val in zip(columns, scaler.mean_, scaler.scale_):
            print(f" - {col}: mean={mean_val}, std={std_val}")

        return self.dataset

    def robust_normalization(self, columns: list[str], dataset: pd.DataFrame = pd.NA):
        """
        Applies Robust normalization to the specified columns using sklearn's RobustScaler.
        This method scales features using statistics that are robust to outliers:
        (x - median) / IQR

        :param columns: List of column names to normalize.
        :param dataset: Optional DataFrame to override self.dataset.
        :return: DataFrame with normalized columns.
        """
        if dataset is not None and dataset is not pd.NA:
            self.dataset = dataset


        # Check columns
        missing = [col for col in columns if col not in self.dataset.columns]
        if missing:
            raise ValueError(f"The dataset must contain the following columns: {missing}")

        non_numeric = [col for col in columns if not pd.api.types.is_numeric_dtype(self.dataset[col])]
        if non_numeric:
            raise ValueError(f"The following columns are not numeric and cannot be normalized: {non_numeric}")

        scaler = RobustScaler()
        self.dataset[columns] = scaler.fit_transform(self.dataset[columns])

        print("Robust normalization applied to columns:")
        for col, center, scale in zip(columns, scaler.center_, scaler.scale_):
            print(f" - {col}: median={center}, IQR={scale}")

        return self.dataset
    
    # ----------------------------#
    #           ENCODING          #
    # ----------------------------# 
    def label_encoding(self, columns: list[str], dataset: pd.DataFrame = pd.NA):
        """
        Applies Label Encoding to the specified categorical columns.

        :param columns: List of categorical columns to encode.
        :param dataset: Optional DataFrame to override self.dataset.
        :return: DataFrame with encoded columns.
        """
        if dataset is not None and dataset is not pd.NA:
            self.dataset = dataset


        # Check if all columns exist
        missing = [col for col in columns if col not in self.dataset.columns]
        if missing:
            raise ValueError(f"The dataset must contain the following columns: {missing}")

        for col in columns:
            if pd.api.types.is_numeric_dtype(self.dataset[col]):
                raise ValueError(f"Column '{col}' is numeric and may not need label encoding.")

            le = LabelEncoder()
            self.dataset[col] = le.fit_transform(self.dataset[col].astype(str))

            # Build mapping: original value -> encoded value
            value_mapping = {cls: int(label) for label, cls in enumerate(le.classes_)}

            print(f"Label Encoding applied to '{col}':")
            for original, encoded in value_mapping.items():
                print(f"  '{original}' -> {encoded}")

        return self.dataset


    def one_hot_encoding(self, columns: list[str], drop_first=False, dataset: pd.DataFrame = pd.NA):
        """
        Applies One-Hot Encoding to the specified categorical columns.

        :param columns: List of categorical columns to encode.
        :param drop_first: Whether to drop the first category (to avoid multicollinearity).
        :param dataset: Optional DataFrame to override self.dataset.
        :return: DataFrame with encoded columns.
        """
        if dataset is not None and dataset is not pd.NA:
            self.dataset = dataset


        missing = [col for col in columns if col not in self.dataset.columns]
        if missing:
            raise ValueError(f"The dataset must contain the following columns: {missing}")

        ohe = OneHotEncoder(drop='first' if drop_first else None, sparse_output=False)
        encoded = ohe.fit_transform(self.dataset[columns].astype(str))

        encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(columns), index=self.dataset.index)

        self.dataset = self.dataset.drop(columns=columns)
        self.dataset = pd.concat([self.dataset, encoded_df], axis=1)

        print(f"One-Hot Encoding applied to columns: {columns}")
        return self.dataset
    

    def ordinal_encoding(self, columns: list[str], 
                        categories: list[list[str]] = None, 
                        mapping: dict[str, dict] = None,
                        dataset: pd.DataFrame = pd.NA):
        """
        Applies Ordinal Encoding to the specified categorical columns.

        :param columns: List of categorical columns to encode.
        :param categories: Optional list of category orderings for each column.
                        Used with OrdinalEncoder when no mapping dict is provided.
        :param mapping: Optional dictionary of mappings {column_name: {category: value}}.
                        If provided for a column, mapping is applied directly.
        :param dataset: Optional DataFrame to override self.dataset.
        :return: DataFrame with encoded columns.
        """
        if dataset is not None and dataset is not pd.NA:
            self.dataset = dataset


        # Check if all columns exist in the dataset
        missing = [col for col in columns if col not in self.dataset.columns]
        if missing:
            raise ValueError(f"The dataset must contain the following columns: {missing}")

        # Apply manual mapping if provided
        if mapping:
            for col in columns:
                if col in mapping:
                    # Warn if dataset contains categories not present in mapping keys
                    unknown_cats = set(self.dataset[col].unique()) - set(mapping[col].keys())
                    if unknown_cats:
                        print(f"Warning: column '{col}' has unknown categories not in mapping: {unknown_cats}")
                    # Map categories manually (safe assignment, avoiding inplace on views)
                    self.dataset[col] = self.dataset[col].map(mapping[col])
                    print(f"Ordinal Encoding by mapping applied to '{col}': {mapping[col]}")
            # Remove columns already mapped from further encoding step
            columns = [col for col in columns if col not in mapping]

        # For remaining columns, apply OrdinalEncoder with given categories or default alphabetical order
        if columns:
            if categories and len(categories) != len(columns):
                raise ValueError("Length of 'categories' must match the number of columns.")

            # If categories is None, OrdinalEncoder orders categories alphabetically by default
            oe = OrdinalEncoder(categories=categories)
            encoded_values = oe.fit_transform(self.dataset[columns].astype(str))

            # Safe assignment to avoid chained assignment warning
            self.dataset.loc[:, columns] = encoded_values

            for col, cats in zip(columns, oe.categories_):
                print(f"Ordinal Encoding applied to '{col}': order={list(cats)}")

        return self.dataset

    # ----------------------------#
    #            BINNING          #
    # ----------------------------#
    def equal_width_binning(self,
                            column: str,
                            bins: int,
                            labels: list[str] = None,
                            dataset: pd.DataFrame = pd.NA,
                            handle_na: str = "ignore",
                            fill_value: float | None = None):
        """
        Applies equal-width binning to a numeric column and prints a human-readable
        summary mapping bins -> which values fell into them.

        :param column: Column name to bin.
        :param bins: Number of equal-width bins.
        :param labels: Optional category labels for the bins.
        :param dataset: Optional DataFrame to override self.dataset.
        :param handle_na: How to handle NA values:
            - "ignore": leave NaN in the binned column (default)
            - "separate": add a "Missing" category and assign NaNs there
            - "fill": fill NaNs with `fill_value` before binning (fill_value required)
        :param fill_value: Value used when handle_na == "fill".
        :return: DataFrame with new column "<column>_binned".
        """
        if dataset is not None and dataset is not pd.NA:
            self.dataset = dataset

        if column not in self.dataset.columns:
            raise ValueError(f"Column '{column}' not found in dataset.")

        if not pd.api.types.is_numeric_dtype(self.dataset[column]):
            raise ValueError(f"Column '{column}' must be numeric for binning.")

        # Prepare numeric series for binning (do not overwrite original yet)
        col_for_binning = self.dataset[column].copy()

        # Handle NaNs before binning
        if handle_na == "drop":
            # drop rows where column is NA
            self.dataset = self.dataset.dropna(subset=[column])
            col_for_binning = self.dataset[column].copy()
        elif handle_na == "fill":
            if fill_value is None:
                raise ValueError("When handle_na=='fill' you must provide a fill_value.")
            col_for_binning = col_for_binning.fillna(fill_value)
        elif handle_na not in ("ignore", "separate", "fill", "drop"):
            raise ValueError("handle_na must be one of ['ignore','separate','fill','drop'].")

        # Apply binning (pd.cut returns a categorical if labels provided or not)
        binned = pd.cut(col_for_binning, bins=bins, labels=labels)

        # If 'separate', mark original NaNs explicitly as 'Missing' in the categorical
        if handle_na == "separate":
            if not pd.api.types.is_categorical_dtype(binned):
                binned = binned.astype("category")
            if "Missing" not in binned.cat.categories:
                binned = binned.cat.add_categories(["Missing"])
            binned = binned.where(~self.dataset[column].isna(), other="Missing")

        # Build summary using col_for_binning and binned (before overwriting)
        total = len(self.dataset)
        print(f"Equal-width binning applied to '{column}' into {bins} bins. Summary:")

        # Ensure categorical for consistent categories list
        if not pd.api.types.is_categorical_dtype(binned):
            # convert to categorical to obtain categories
            binned = pd.Categorical(binned)

        categories = list(binned.cat.categories)

        for cat in categories:
            mask = binned == cat
            count = int(mask.sum())
            pct = count / total if total > 0 else 0.0
            if str(cat) == "Missing":
                print(f"  Bin '{cat}': count={count} ({pct:.2%})")
            else:
                values = col_for_binning[mask].dropna()
                minv = None if values.empty else values.min()
                maxv = None if values.empty else values.max()
                examples = list(values.head(5).unique())
                print(f"  Bin '{cat}': count={count} ({pct:.2%}), min={minv}, max={maxv}, examples={examples}")

        # If handle_na == "ignore" inform about leftover NaNs
        if handle_na == "ignore":
            na_count = self.dataset[column].isna().sum()
            if na_count > 0:
                print(f"  Note: original column has {na_count} NaN(s); they remain NaN after binning.")

        # Replace original column with the binned labels (stringified to avoid category dtype issues downstream)
        self.dataset.loc[:, column] = binned.astype(str)

        return self.dataset


    def equal_frequency_binning(self,
                                column: str,
                                q: int,
                                labels: list[str] = None,
                                dataset: pd.DataFrame = pd.NA,
                                handle_na: str = "ignore",
                                fill_value: float | None = None):
        """
        Applies equal-frequency (quantile) binning to a numeric column and prints a summary
        mapping bins -> which values fell into them.

        :param column: Column name to bin.
        :param q: Number of quantile bins.
        :param labels: Optional labels for the bins.
        :param dataset: Optional DataFrame to override self.dataset.
        :param handle_na: 'ignore' | 'separate' | 'fill' (see equal_width_binning).
        :param fill_value: Value used when handle_na == "fill".
        :return: DataFrame with new column "<column>_binned".
        """
        if dataset is not None and dataset is not pd.NA:
            self.dataset = dataset

        if column not in self.dataset.columns:
            raise ValueError(f"Column '{column}' not found in dataset.")

        if not pd.api.types.is_numeric_dtype(self.dataset[column]):
            raise ValueError(f"Column '{column}' must be numeric for binning.")

        # Prepare numeric series for binning
        col_for_binning = self.dataset[column].copy()

        # Handle NaNs before binning
        if handle_na == "drop":
            self.dataset = self.dataset.dropna(subset=[column])
            col_for_binning = self.dataset[column].copy()
        elif handle_na == "fill":
            if fill_value is None:
                raise ValueError("When handle_na=='fill' you must provide a fill_value.")
            col_for_binning = col_for_binning.fillna(fill_value)
        elif handle_na not in ("ignore", "separate", "fill", "drop"):
            raise ValueError("handle_na must be one of ['ignore','separate','fill','drop'].")

        # Apply qcut; use duplicates="drop" to be robust when there are not enough unique values
        try:
            binned = pd.qcut(col_for_binning, q=q, labels=labels, duplicates="drop")
        except ValueError as e:
            raise ValueError(f"qcut failed: {e}")

        # If 'separate', mark original NaNs explicitly as 'Missing' in the categorical
        if handle_na == "separate":
            if not pd.api.types.is_categorical_dtype(binned):
                binned = binned.astype("category")
            if "Missing" not in binned.cat.categories:
                binned = binned.cat.add_categories(["Missing"])
            binned = binned.where(~self.dataset[column].isna(), other="Missing")

        # Build summary using col_for_binning and binned (before overwriting)
        total = len(self.dataset)
        print(f"Equal-frequency binning applied to '{column}' into {q} quantiles. Summary:")

        if not pd.api.types.is_categorical_dtype(binned):
            binned = pd.Categorical(binned)

        categories = list(binned.cat.categories)

        for cat in categories:
            mask = binned == cat
            count = int(mask.sum())
            pct = count / total if total > 0 else 0.0
            if str(cat) == "Missing":
                print(f"  Bin '{cat}': count={count} ({pct:.2%})")
            else:
                values = col_for_binning[mask].dropna()
                minv = None if values.empty else values.min()
                maxv = None if values.empty else values.max()
                examples = list(values.head(5).unique())
                print(f"  Bin '{cat}': count={count} ({pct:.2%}), min={minv}, max={maxv}, examples={examples}")

        if handle_na == "ignore":
            na_count = self.dataset[column].isna().sum()
            if na_count > 0:
                print(f"  Note: original column has {na_count} NaN(s); they remain NaN after binning.")

        # Replace original column with the binned labels (string)
        self.dataset.loc[:, column] = binned.astype(str)

        return self.dataset


    def custom_binning(self,
                       column: str,
                       bin_edges: list[float],
                       labels: list[str] = None,
                       dataset: pd.DataFrame = pd.NA,
                       handle_na: str = "ignore",
                       fill_value: float | None = None):
        """
        Applies custom binning using provided bin edges and prints a summary of which values
        fell into each bin.

        :param column: Column name to bin.
        :param bin_edges: List of bin edges (must include min and max).
        :param labels: Optional labels for the bins.
        :param dataset: Optional DataFrame to override self.dataset.
        :param handle_na: 'ignore' | 'separate' | 'fill' (see equal_width_binning).
        :param fill_value: Value used when handle_na == "fill".
        :return: DataFrame with new column "<column>_binned".
        """
        if dataset is not None and dataset is not pd.NA:
            self.dataset = dataset

        if column not in self.dataset.columns:
            raise ValueError(f"Column '{column}' not found in dataset.")

        if not pd.api.types.is_numeric_dtype(self.dataset[column]):
            raise ValueError(f"Column '{column}' must be numeric for binning.")

        # Prepare numeric series for binning
        col_for_binning = self.dataset[column].copy()

        # Handle NaNs before binning
        if handle_na == "drop":
            self.dataset = self.dataset.dropna(subset=[column])
            col_for_binning = self.dataset[column].copy()
        elif handle_na == "fill":
            if fill_value is None:
                raise ValueError("When handle_na=='fill' you must provide a fill_value.")
            col_for_binning = col_for_binning.fillna(fill_value)
        elif handle_na not in ("ignore", "separate", "fill", "drop"):
            raise ValueError("handle_na must be one of ['ignore','separate','fill','drop'].")

        # Apply cut with provided edges
        binned = pd.cut(col_for_binning, bins=bin_edges, labels=labels, include_lowest=True)

        # If 'separate', mark original NaNs explicitly as 'Missing' in the categorical
        if handle_na == "separate":
            if not pd.api.types.is_categorical_dtype(binned):
                binned = binned.astype("category")
            if "Missing" not in binned.cat.categories:
                binned = binned.cat.add_categories(["Missing"])
            binned = binned.where(~self.dataset[column].isna(), other="Missing")

        # Build summary using col_for_binning and binned (before overwriting)
        total = len(self.dataset)
        print(f"Custom binning applied to '{column}' with edges {bin_edges}. Summary:")

        if not pd.api.types.is_categorical_dtype(binned):
            binned = pd.Categorical(binned)

        categories = list(binned.cat.categories)

        for cat in categories:
            mask = binned == cat
            count = int(mask.sum())
            pct = count / total if total > 0 else 0.0
            if str(cat) == "Missing":
                print(f"  Bin '{cat}': count={count} ({pct:.2%})")
            else:
                values = col_for_binning[mask].dropna()
                minv = None if values.empty else values.min()
                maxv = None if values.empty else values.max()
                examples = list(values.head(5).unique())
                print(f"  Bin '{cat}': count={count} ({pct:.2%}), min={minv}, max={maxv}, examples={examples}")

        if handle_na == "ignore":
            na_count = self.dataset[column].isna().sum()
            if na_count > 0:
                print(f"  Note: original column has {na_count} NaN(s); they remain NaN after binning.")

        # Replace original column with the binned labels (string)
        self.dataset.loc[:, column] = binned.astype(str)

        return self.dataset
