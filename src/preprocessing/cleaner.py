import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from typing import List

class ADNICleaner:
    """
    A class for preparing the ADNIMERGE dataset.
    """
    def __init__(self, dataset: pd.DataFrame):
        """
        Initialize the normalizator with the given dataset.
        
        :param dataset: A pandas DataFrame loaded from ADNIMERGE.csv, already prefiltered as specified.
        """
        self.dataset = dataset.copy()
    

    def select_baseline_visits(self, dataset: pd.DataFrame = None):
        """
        Filters the dataset to keep only the baseline ('bl') visit for each patient
        with a non-missing 'DX_bl' value.
        Prints summary information about the filtering process.

        :return: A pandas DataFrame containing only the filtered baseline visit rows.
        """
        if dataset is not None and dataset is not pd.NA:
            self.dataset = dataset

        # Check that necessary columns exist
        required_columns = ['VISCODE', 'PTID', 'DX_bl']
        missing = [col for col in required_columns if col not in self.dataset.columns]
        if missing:
            raise ValueError(f"The dataset must contain the following columns: {missing}")

        # Store the number of rows before filtering
        initial_row_count = len(self.dataset)

        # Filter to only baseline visits with non-null DX_bl
        condition = (self.dataset['VISCODE'] == 'bl') & (self.dataset['DX_bl'].notna())
        self.dataset = self.dataset[condition].copy()

        # Drop duplicate PTIDs if any exist
        self.dataset = self.dataset.drop_duplicates(subset='PTID', keep='first')

        # Store the number of rows after filtering
        final_row_count = len(self.dataset)
        rows_removed = initial_row_count - final_row_count

        # Print filtering summary
        print("The dataset has been filtered. Only valid baseline visits with non-null 'DX_bl' per patient are retained.")
        print("Rows before filtering:\t", initial_row_count)
        print("Rows after filtering:\t", final_row_count)
        print("Rows removed:\t\t", rows_removed)

        return self.dataset

    
    def fill_missing_bl_from_original(self, dataset: pd.DataFrame = None):
        """
        Fills missing values in '_bl' columns using values from the corresponding base columns.
        For each '_bl' column:
          - If the base column exists
          - And the value in '_bl' is NaN but the base is not
          - Then copy the base value into the '_bl' column.

        Prints a summary of how many values were filled for each column.

        :return: The updated DataFrame.
        """
        if dataset is not None and dataset is not pd.NA:
            self.dataset = dataset

        modifications = {}

        for col in self.dataset.columns:
            if col.endswith('_bl'):
                base_col = col[:-3]  # Remove '_bl' suffix
                if base_col in self.dataset.columns:
                    bl_na_mask = self.dataset[col].isna()
                    base_notna_mask = self.dataset[base_col].notna()
                    mask_to_fill = bl_na_mask & base_notna_mask
                    count_filled = mask_to_fill.sum()

                    if count_filled > 0:
                        self.dataset.loc[mask_to_fill, col] = self.dataset.loc[mask_to_fill, base_col]
                        modifications[col] = count_filled

        if modifications:
            print("Filled missing values in '_bl' columns from base columns:")
            for col, count in modifications.items():
                print(f" - {col}: {count} values filled")
        else:
            print("No missing '_bl' values were filled from base columns.")

        return self.dataset
    

    def remap_smc_baseline(self, dataset: pd.DataFrame = None) -> pd.DataFrame:
        """ 
        Replace 'SMC' values ​​in DX_bl according to DX mapping rules: 
        - If DX == 'CN' -> DX_bl = 'CN' 
        - If DX == 'MCI' -> DX_bl = 'EMCI' 
        - If DX == 'Dementia' -> DX_bl = 'AD' 
        - Otherwise -> leave as 'SMC' 

        After direct mapping, any remaining 'SMC' in DX_bl are replaced with 'CN'. 
        Prints a summary of replacements. 

        :param dataset: Optional external dataset to use. 

        :return: The updated DataFrame. 
        """
        if dataset is not None and dataset is not pd.NA:
            self.dataset = dataset

        # Safety check
        required_cols = ["DX_bl", "DX"]
        missing = [col for col in required_cols if col not in self.dataset.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Counters
        replaced_cn_direct = 0
        replaced_emci_direct = 0
        replaced_ad_direct = 0

        def map_dx_to_dx_bl(dx_value):
            if pd.isna(dx_value):
                return None
            if dx_value == "CN":
                return "CN"
            if dx_value == "MCI":
                return "EMCI"
            if dx_value == "Dementia":
                return "AD"
            return None

        # Trova gli indici da modificare
        smc_indices = self.dataset.index[self.dataset["DX_bl"] == "SMC"].tolist()

        for idx in smc_indices:
            current_dx = self.dataset.at[idx, "DX"]
            mapped = map_dx_to_dx_bl(current_dx)
            if mapped is not None:
                self.dataset.at[idx, "DX_bl"] = mapped
                if mapped == "CN":
                    replaced_cn_direct += 1
                elif mapped == "EMCI":
                    replaced_emci_direct += 1
                elif mapped == "AD":
                    replaced_ad_direct += 1

        # Fallback finale: tutto ciò che rimane SMC diventa CN
        remaining_smc = self.dataset["DX_bl"].value_counts().get("SMC", 0)
        if remaining_smc > 0:
            self.dataset.loc[self.dataset["DX_bl"] == "SMC", "DX_bl"] = "CN"
            replaced_cn_fallback = remaining_smc
        else:
            replaced_cn_fallback = 0

        total_replaced = (
            replaced_cn_direct
            + replaced_emci_direct
            + replaced_ad_direct
            + replaced_cn_fallback
        )

        # Report
        print(f"SMC -> CN (direct from DX): {replaced_cn_direct}")
        print(f"SMC -> EMCI (direct from DX MCI): {replaced_emci_direct}")
        print(f"SMC -> AD (direct from DX Dementia): {replaced_ad_direct}")
        print(f"SMC -> CN (fallback for leftovers): {replaced_cn_fallback}")
        print(f"TOTAL SMC replacements (all targets): {total_replaced}")

        return self.dataset
    

    def fix_dx_bl_discrepancies(self, dataset: pd.DataFrame = None) -> pd.DataFrame:
        """
        Fix specific discrepancies between 'DX' and 'DX_bl' according to the rules:
          - If DX == 'CN' and DX_bl == 'EMCI'     -> set DX_bl = 'CN'
          - If DX == 'MCI' and DX_bl == 'AD'      -> set DX_bl = 'LMCI'
          - If DX == 'Dementia' and DX_bl == 'LMCI' -> set DX_bl = 'AD'
        
        Print a report of how many cases were changed for each rule.

        :return: The modified DataFrame.
        """
        if dataset is not None and dataset is not pd.NA:
            self.dataset = dataset

        # Safety check
        required_cols = ["DX", "DX_bl"]
        missing = [c for c in required_cols if c not in self.dataset.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Rule 1: DX == 'CN' & DX_bl == 'EMCI' -> DX_bl = 'CN'
        mask1 = (self.dataset["DX"] == "CN") & (self.dataset["DX_bl"] == "EMCI")
        count1_before = mask1.sum()
        if count1_before > 0:
            self.dataset.loc[mask1, "DX_bl"] = "CN"

        # Rule 2: DX == 'MCI' & DX_bl == 'AD' -> DX_bl = 'LMCI'
        mask2 = (self.dataset["DX"] == "MCI") & (self.dataset["DX_bl"] == "AD")
        count2_before = mask2.sum()
        if count2_before > 0:
            self.dataset.loc[mask2, "DX_bl"] = "LMCI"

        # Rule 3: DX == 'Dementia' & DX_bl == 'LMCI' -> DX_bl = 'AD'
        mask3 = (self.dataset["DX"] == "Dementia") & (self.dataset["DX_bl"] == "LMCI")
        count3_before = mask3.sum()
        if count3_before > 0:
            self.dataset.loc[mask3, "DX_bl"] = "AD"

        total_changed = count1_before + count2_before + count3_before

        # Report
        print(f"Fixed DX_bl discrepancies:")
        print(f" - DX='CN' & DX_bl='EMCI'  -> DX_bl='CN'   : {count1_before} rows changed")
        print(f" - DX='MCI' & DX_bl='AD'   -> DX_bl='LMCI' : {count2_before} rows changed")
        print(f" - DX='Dementia' & DX_bl='LMCI' -> DX_bl='AD' : {count3_before} rows changed")
        print(f"TOTAL rows modified: {total_changed}")

        return self.dataset
    

    def consolidate_bl_columns(self, dataset: pd.DataFrame = None): 
        """
        Removes the original version of columns that have a '_bl' counterpart,
        and renames '_bl' columns by removing the '_bl' suffix,
        except for 'DX_bl', which is preserved as-is along with 'DX'.

        :return: The modified DataFrame.
        """
        if dataset is not None and dataset is not pd.NA:
            self.dataset = dataset

        cols_to_rename = {}
        cols_to_drop = []

        for col in self.dataset.columns:
            if col.lower().endswith('_bl'): 
                base_col = col[:-3]
                if base_col in self.dataset.columns:
                    cols_to_drop.append(base_col)
                cols_to_rename[col] = base_col

        self.dataset.drop(columns=cols_to_drop, inplace=True)
        self.dataset.rename(columns=cols_to_rename, inplace=True)

        print(f"Dropped {len(cols_to_drop)} columns: {cols_to_drop}")
        print(f"Renamed {len(cols_to_rename)} columns.")

        return self.dataset
    

    def remove_single_value_attributes(self, dataset: pd.DataFrame = None):
        """
        Removes columns that contain only a single unique value.
        These attributes are not informative and can be safely dropped.

        :return: The modified DataFrame.
        """
        if dataset is not None and dataset is not pd.NA:
            self.dataset = dataset

        # Identify columns with a single unique value
        single_value_cols = [col for col in self.dataset.columns if self.dataset[col].nunique(dropna=False) == 1]

        if single_value_cols:
            print("The following single-value columns have been dropped:")
            print(single_value_cols)
            # Drop those columns
            self.dataset.drop(columns=single_value_cols, inplace=True)
        else:
            print("No single-value columns found.")

        return self.dataset
    
    def remove_duplicate_rows(self, dataset: pd.DataFrame = None):
        """
        Remove duplicate rows from the dataset.
        
        :param dataset: Optional external dataset to use.

        :return: The modified DataFrame.
        """
        if dataset is not None and dataset is not pd.NA:
            self.dataset = dataset


        initial_count = len(self.dataset)
        self.dataset.drop_duplicates(inplace=True, ignore_index=True)
        removed_count = initial_count - len(self.dataset)

        if removed_count > 0:
            print(f"{removed_count} duplicate rows have been removed.")
        else:
            print("No duplicate rows found.")

        return self.dataset
    

    def clean_limit_values(self, columns: list, dataset: pd.DataFrame = None, 
                            factor_below: float = 0.99, factor_above: float = 1.01):
        """
        Cleans columns with string values like '<80' or '>1700' by converting them to numeric values.
        For values with '<', it multiplies the number by `factor_below`.
        For values with '>', it multiplies the number by `factor_above`.

        :param columns: List of column names to clean.
        :param dataset: Optional external dataset to use.
        :param factor_below: Multiplier for values below detection limit (e.g., '<80' becomes 80 * factor_below).
        :param factor_above: Multiplier for values above detection limit (e.g., '>1700' becomes 1700 * factor_above).
        
        :return: DataFrame with cleaned columns.
        """
        if dataset is not None and dataset is not pd.NA:
            self.dataset = dataset


        for col in columns:
            if col not in self.dataset.columns:
                print(f"Column '{col}' not found in dataset. Skipping.")
                continue

            # Convert all values to strings and strip whitespace
            cleaned = self.dataset[col].astype(str).str.strip()

            # Identify rows with '<' or '>'
            below_mask = cleaned.str.startswith('<')
            above_mask = cleaned.str.startswith('>')

            # Remove symbols
            cleaned_numeric = cleaned.str.replace('<', '', regex=False).str.replace('>', '', regex=False)

            # Convert to float
            numeric = pd.to_numeric(cleaned_numeric, errors='coerce')

            # Apply multipliers for limits
            numeric[below_mask] *= factor_below
            numeric[above_mask] *= factor_above

            # Replace in dataset
            self.dataset[col] = numeric

            print(f"Column '{col}' cleaned: '<' values scaled by {factor_below}, '>' values by {factor_above}.")

        return self.dataset
    

    def replace_value_in_column(self, column: str, old_value, new_value, dataset: pd.DataFrame = None):
        """
        Replace all occurrences of `old_value` with `new_value` in the specified column.

        :param column: The column name where replacement will be done.
        :param old_value: The value to be replaced.
        :param new_value: The value to replace with.
        :param dataset: Optional external dataset to use.

        :return: DataFrame with replaced values.
        """
        if dataset is not None and dataset is not pd.NA:
            self.dataset = dataset

        if column not in self.dataset.columns:
            print(f"Column '{column}' not found. Operation skipped.")
            return self.dataset

        # Replace values
        self.dataset[column] = self.dataset[column].replace(old_value, new_value)

        print(f"Replaced all occurrences of '{old_value}' with '{new_value}' in column '{column}'.")
        return self.dataset
    

    def remove_attributes_by_percentage(self, threshold: float = 50.0, dataset: pd.DataFrame = None):
        """
        Removes columns that have more than a specified percentage of missing values.
        
        :param threshold: The percentage (0-100) above which attributes will be removed. Default is 50%.

        :return: The modified DataFrame.
        """
        if dataset is not None and dataset is not pd.NA:
            self.dataset = dataset

        null_percent = (self.dataset.isna().sum() / len(self.dataset['RID']) * 100).sort_values(ascending=False)
        columns_to_drop = null_percent[null_percent > threshold].index
        print("The following columns have been dropped:")
        print(list(columns_to_drop))
        self.dataset.drop(columns=columns_to_drop, inplace=True)
        return self.dataset
    

    def fill_null_with_value(self, column: str, value, dataset: pd.DataFrame = None):
        """
        Fill NaN values in a specified column with a given constant value using SimpleImputer,
        ensuring type compatibility before applying the imputer.

        :param column: The column name in which to replace NaN values.
        :param value: The value to use for replacing NaN values.
        :param dataset: Optional external dataset to use.

        :return: The modified DataFrame.
        """
        if dataset is not None and dataset is not pd.NA:
            self.dataset = dataset

        if column not in self.dataset.columns:
            print(f"Column '{column}' not found. Operation skipped.")
            return self.dataset

        # Determine the dtype of non-null values in the column (if any) and check compatibility
        non_null_values = self.dataset[column].dropna()
        if len(non_null_values) > 0:
            col_type = type(non_null_values.iloc[0])
            if not isinstance(value, col_type):
                print(f"Type mismatch: Column '{column}' expects values of type {col_type.__name__}, "
                      f"but got {type(value).__name__}. Operation skipped.")
                return self.dataset

        # Use SimpleImputer with strategy 'constant'
        imputer = SimpleImputer(strategy='constant', fill_value=value)
        # fit_transform expects 2D array
        filled = imputer.fit_transform(self.dataset[[column]])
        # assign back avoiding chained assignment
        self.dataset[column] = pd.Series(filled.ravel(), index=self.dataset.index).where(
            ~self.dataset[column].isna(), pd.Series(filled.ravel(), index=self.dataset.index)
        )
        # The where above keeps existing non-NaN values; but since we used fit_transform on full column,
        # we can simply assign the filled result; however to be safe, we'll keep assignment:
        self.dataset[column] = pd.Series(filled.ravel(), index=self.dataset.index)

        print(f"NaN values in column '{column}' have been replaced with '{value}'.")
        return self.dataset
            
    
    def convert_float_to_int(self, column: str, method: str = "round", fillna_value: int = 0, dataset: pd.DataFrame = None):
        """
        Convert float values in a specified column to NumPy int64 using the chosen rounding method.
        
        :param column: The column to convert.
        :param method: Rounding method: 'floor', 'ceil', or 'round'.
        :param fillna_value: Value to replace NaNs before converting to int64 (default=0).
        :param dataset: Optional external dataset to use.

        :return: The modified DataFrame.
        """
        if dataset is not None and dataset is not pd.NA:
            self.dataset = dataset

        if column not in self.dataset.columns:
            print(f"Column '{column}' not found. Operation skipped.")
            return self.dataset

        if not np.issubdtype(self.dataset[column].dropna().dtype, np.number):
            print(f"Column '{column}' is not numeric. Operation skipped.")
            return self.dataset

        col_data = self.dataset[column].copy()

        # Apply rounding
        if method == "floor":
            col_data = np.floor(col_data)
            print(f"Column '{column}' converted using floor rounding.")
        elif method == "ceil":
            col_data = np.ceil(col_data)
            print(f"Column '{column}' converted using ceil rounding.")
        elif method == "round":
            col_data = np.where(
                col_data - np.floor(col_data) < 0.5,
                np.floor(col_data),
                np.ceil(col_data)
            )
            print(f"Column '{column}' converted using round-half-up rounding.")
        else:
            print(f"Invalid rounding method '{method}'. Use 'floor', 'ceil', or 'round'. Skipped.")
            return self.dataset

        # Handle NaNs before converting to int64
        col_data = pd.Series(col_data).fillna(fillna_value).astype(np.int64)

        self.dataset[column] = col_data

        return self.dataset
    

    def impute_mean_by_class(self, columns: List[str], class_col: str, dataset: pd.DataFrame = None) -> pd.DataFrame:
        """
        Impute missing values in numeric columns using the mean value within each class of a specified categorical column.  
        If a class group has all missing values, fall back to the global mean of the column.

        :param columns: List[str] -> List of numeric columns to impute.
        :param class_col: str ->  Column name representing the class/group used for group-wise mean imputation.
        :param dataset: pd.DataFrame, optional -> Alternative dataset to apply imputation on. If None, uses self.dataset.

        :return: The DataFrame with missing values imputed per class.
        """
        if dataset is not None and dataset is not pd.NA:
            self.dataset = dataset

        if class_col not in self.dataset.columns:
            print(f"Class column '{class_col}' not found in dataset. Operation skipped.")
            return self.dataset

        for col in columns:
            if col not in self.dataset.columns:
                print(f"Column '{col}' not found in dataset. Skipping.")
                continue

            if not np.issubdtype(self.dataset[col].dropna().dtype, np.number):
                print(f"Column '{col}' is not numeric. Skipping.")
                continue

            global_imputer = SimpleImputer(strategy="mean")
            try:
                global_imputer.fit(self.dataset[[col]])
            except Exception:
                global_imputer = None

            filled_col = self.dataset[col].copy()

            for group_val, idx in self.dataset.groupby(class_col).groups.items():
                group_idx = list(idx)
                group_series = self.dataset.loc[group_idx, [col]]

                if group_series[col].dropna().empty:
                    if global_imputer is not None:
                        try:
                            transformed = global_imputer.transform(group_series)
                            filled_col.loc[group_idx] = transformed.ravel()
                        except Exception:
                            continue
                    continue

                imputer = SimpleImputer(strategy="mean")
                try:
                    filled_values = imputer.fit_transform(group_series)
                    filled_col.loc[group_idx] = filled_values.ravel()
                except Exception:
                    if global_imputer is not None:
                        try:
                            transformed = global_imputer.transform(group_series)
                            filled_col.loc[group_idx] = transformed.ravel()
                        except Exception:
                            continue

            self.dataset[col] = filled_col
            print(f"Column '{col}' imputed with mean by class '{class_col}'.")

        return self.dataset
