import os
import joblib
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns

from IPython.display import display
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class ADNIClassifier:
    """
    ADNIClassifier: helper class to train and evaluate multiple classifier models on an ADNIMERG dataset.
    """
    # ----------------------------#
    #        INITIALIZATION       #
    # ----------------------------# 
    def __init__(self, classifier: str = "Standard1"):
        """
        Initialize the class and load a predefined set of classifiers 
        based on the provided selection string.
        """
        if classifier == "Standard1" or classifier == "standard1" or classifier == "STANDARD1" or classifier == "None":
            self.classifiers = self._default_classifiers_1()
        elif classifier == "Standard2" or classifier == "standard2" or classifier == "STANDARD2":
            self.classifiers = self._default_classifiers_2()
        else:
            self.classifiers = self._default_classifiers_1()

    # ----------------------------#
    #   CLASSIFIERS DEFINITIONS   #
    # ----------------------------# 
    def _default_classifiers_1(self):
        return {
            "Decision Tree": DecisionTreeClassifier(
                random_state=42, class_weight="balanced",
                criterion="entropy", max_depth=6, max_features=1.0,
                min_samples_leaf=8, min_samples_split=2
            ),
            "Random Forest": RandomForestClassifier(
                random_state=42, class_weight="balanced", n_jobs=-1,
                criterion="entropy", max_depth=None, max_features=1.0,
                min_samples_leaf=2, n_estimators=100
            ),
            "Extra Trees": ExtraTreesClassifier(
                random_state=42, class_weight="balanced", n_jobs=-1,
                criterion="entropy", max_depth=None, max_features=1.0,
                min_samples_leaf=2, n_estimators=75
            ),
            "XGBoost": XGBClassifier(
                random_state=42, use_label_encoder=False, eval_metric="mlogloss", verbosity=0,
                colsample_bytree=0.7, gamma=1.0, learning_rate=0.1,
                max_depth=6, n_estimators=50, reg_alpha=1, reg_lambda=0,
                subsample=1.0
            ),
            "LightGBM": LGBMClassifier(
                random_state=42, verbose=-1,
                colsample_bytree=1.0, learning_rate=0.01, max_depth=8,
                min_child_samples=20, n_estimators=100, num_leaves=15,
                reg_alpha=0, reg_lambda=1, subsample=0.8
            ),
            "CatBoost": CatBoostClassifier(
                random_state=42, verbose=False, loss_function="MultiClass",
                bagging_temperature=0.0, border_count=64, depth=8,
                iterations=75, l2_leaf_reg=3, learning_rate=0.1,
                random_strength=0.5
            ),
            "Multinomial Logistic Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(
                    random_state=42, solver="saga", max_iter=2000, class_weight="balanced",
                    C=1.0, penalty="l1"
                ))
            ]),
            "Bagging": BaggingClassifier(
                random_state=42, n_jobs=-1,
                bootstrap=False, max_features=1.0, max_samples=0.6,
                n_estimators=100
            )
        }

    
    def _default_classifiers_2(self):
        """
        Return a dictionary with tuned classifier instances.
        """
        return {
            "Decision Tree": DecisionTreeClassifier(
                random_state=42, class_weight="balanced",
                criterion="gini", max_depth=4, max_features=1.0,
                min_samples_leaf=1, min_samples_split=2
            ),
            "Random Forest": RandomForestClassifier(
                random_state=42, class_weight="balanced", n_jobs=-1,
                criterion="entropy", max_depth=6, max_features=0.5,
                min_samples_leaf=2, n_estimators=50
            ),
            "Extra Trees": ExtraTreesClassifier(
                random_state=42, class_weight="balanced", n_jobs=-1,
                criterion="entropy", max_depth=None, max_features=1.0,
                min_samples_leaf=8, n_estimators=50
            ),
            "XGBoost": XGBClassifier(
                random_state=42, use_label_encoder=False, eval_metric="mlogloss", verbosity=0,
                colsample_bytree=1.0, gamma=1.0, learning_rate=0.1,
                max_depth=8, n_estimators=100, reg_alpha=0, reg_lambda=1,
                subsample=0.8
            ),
            "LightGBM": LGBMClassifier(
                random_state=42, verbose=-1,
                colsample_bytree=0.7, learning_rate=0.1, max_depth=8,
                min_child_samples=5, n_estimators=100, num_leaves=15,
                reg_alpha=1, reg_lambda=1, subsample=0.8
            ),
            "CatBoost": CatBoostClassifier(
                random_state=42, verbose=False, loss_function="MultiClass",
                bagging_temperature=0.0, border_count=32, depth=6,
                iterations=100, l2_leaf_reg=1, learning_rate=0.1,
                random_strength=0.5
            ),
            "Multinomial Logistic Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(
                    random_state=42, solver="saga", max_iter=2000, class_weight="balanced",
                    C=1.0, penalty="l1"
                ))
            ]),
            "Bagging": BaggingClassifier(
                random_state=42, n_jobs=-1,
                bootstrap=False, max_features=0.8, max_samples=0.6, n_estimators=100
            )
        }

    # ----------------------------#
    #          UTILITY            #
    # ----------------------------# 
    def _softmax(self, arr):
        e = np.exp(arr - np.max(arr, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    def _get_probabilities(self, fitted_clf, X, classes):
        if hasattr(fitted_clf, "predict_proba"):
            try:
                probs = fitted_clf.predict_proba(X)
                prob_cols = getattr(fitted_clf, "classes_", None)
                if prob_cols is None and isinstance(fitted_clf, Pipeline):
                    final = fitted_clf.named_steps[list(fitted_clf.named_steps.keys())[-1]]
                    prob_cols = getattr(final, "classes_", None)
                prob_df = pd.DataFrame(probs, columns=prob_cols)
                prob_df = prob_df.reindex(columns=classes, fill_value=0)
                return prob_df.values
            except Exception:
                pass

        if hasattr(fitted_clf, "decision_function"):
            try:
                df = fitted_clf.decision_function(X)
                if df.ndim == 1:
                    df = np.vstack([-df, df]).T
                probs = self._softmax(df)
                prob_df = pd.DataFrame(probs, columns=classes[:probs.shape[1]])
                prob_df = prob_df.reindex(columns=classes, fill_value=0)
                return prob_df.values
            except Exception:
                pass

        preds = fitted_clf.predict(X)
        one_hot = np.zeros((len(preds), len(classes)))
        class_to_idx = {c: i for i, c in enumerate(classes)}
        for i, p in enumerate(preds):
            if p in class_to_idx:
                one_hot[i, class_to_idx[p]] = 1.0
        return one_hot

    def _safe_clone_and_fit(self, clf, X_train, y_train):
        cloned = clone(clf)
        cloned.fit(X_train, y_train)
        return cloned

    def _ensure_dir(self, path):
        os.makedirs(path, exist_ok=True)

    def _clean_name(self, name):
        return name.replace(" ", "_").replace("/", "_")
    
    def _unique_model_path(self, base_dir, clf_name):
        clean_name = self._clean_name(clf_name)
        i = 0
        while True:
            candidate = os.path.join(base_dir, f"{clean_name}{i}.pkl")
            if not os.path.exists(candidate):
                return candidate
            i += 1

    # ----------------------------#
    #     CORE EVALUATION (CV)    #
    # ----------------------------# 
    def _run_repeated_cv(self, clf, X, y, cv_splitter):
        """
        Run repeated stratified CV for a single classifier. 
        Return: true_all (np.array), pred_all (np.array), prob_all (np.ndarray), fold_accuracies (list)
        """
        true_all = []
        pred_all = []
        prob_all_list = []
        fold_accuracies = []
        classes = np.unique(y)

        for train_idx, val_idx in cv_splitter.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Clone and Fit on fold
            fitted = self._safe_clone_and_fit(clf, X_tr, y_tr)

            # Predict and Probabilities
            y_pred = fitted.predict(X_val)
            prob_arr = self._get_probabilities(fitted, X_val, classes)

            true_all.extend(y_val)
            pred_all.extend(y_pred)
            prob_all_list.append(prob_arr)

            fold_acc = accuracy_score(y_val, y_pred)
            fold_accuracies.append(fold_acc)

        # Stack probabilities and arrays
        prob_all = np.vstack(prob_all_list) if len(prob_all_list) > 0 else np.empty((0, len(classes)))
        return np.array(true_all), np.array(pred_all), prob_all, fold_accuracies

    # ----------------------------#
    #          PLOTTING           #
    # ----------------------------# 
    def _plot_roc_per_class(self, roc_dict, classes):
        """
        Plot ROC curves for each class comparing all models in a single figure.
        Uses a grid with 2 columns (2 graphs above, 2 below for 4 classes).
        If number of classes != 4, uses ncols=2 and computes nrows = ceil(n_classes/2).
        """
        n_classes = len(classes)
        if n_classes == 0:
            return

        ncols = 2
        nrows = math.ceil(n_classes / ncols)
        figsize = (12, 5 * nrows)  # width x height, adjust if you want bigger/smaller

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        # Ensure axes is a flattened array for simple indexing
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]

        for idx, cls in enumerate(classes):
            ax = axes[idx]
            # Plot each classifier"s ROC for this class
            for clf_name, (fpr_dict, tpr_dict, auc_dict) in roc_dict.items():
                fpr = fpr_dict.get(cls, None)
                tpr = tpr_dict.get(cls, None)
                auc_val = auc_dict.get(cls, np.nan)

                # If fpr/tpr missing or length mismatch, fallback to diagonal
                if fpr is None or tpr is None or len(fpr) == 0 or len(tpr) == 0:
                    fpr = np.array([0.0, 1.0])
                    tpr = np.array([0.0, 1.0])
                    label = f"{clf_name} (AUC=nan)"
                else:
                    if np.isnan(auc_val):
                        label = f"{clf_name} (AUC=nan)"
                    else:
                        label = f"{clf_name} (AUC={auc_val:.2f})"

                ax.plot(fpr, tpr, lw=2, label=label)

            # plot diagonal
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlim([-0.01, 1.01])
            ax.set_ylim([-0.01, 1.01])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve - Class {cls} (One-vs-Rest)")
            ax.legend(loc="lower right", fontsize="small")
            ax.grid(alpha=0.2)

        # remove unused axes if any
        total_plots = nrows * ncols
        for j in range(len(classes), total_plots):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()


    def _plot_confusion_matrices(self, confusion_dict, title_prefix="Confusion Matrix"):
        """
        Plot confusion matrices in a grid.
        Automatically chooses integer formatting for count matrices and
        decimal formatting for normalized (float) matrices.
        """
        n_classifiers = len(confusion_dict)
        if n_classifiers == 0:
            return

        n_cols = 4
        n_rows = n_classifiers // n_cols + int(n_classifiers % n_cols > 0)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        axes = axes.flatten()

        for idx, (clf_name, cm) in enumerate(confusion_dict.items()):
            ax = axes[idx]

            # Ensure numpy array
            cm_arr = np.asarray(cm, dtype=float)

            # Decide formatting: integer if values are (effectively) integer, else show floats
            if np.allclose(cm_arr, np.round(cm_arr)):
                fmt = "d"
                annot = cm_arr.astype(int)  # annotate integers for neatness
            else:
                fmt = ".2f"
                annot = cm_arr

            sns.heatmap(cm_arr, annot=annot, fmt=fmt, cmap="Blues", cbar=False, ax=ax)
            ax.set_title(f"{clf_name} {title_prefix}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

        # Remove unused axes
        for idx in range(len(confusion_dict), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.show()


    def _plot_violin(self, accuracies_per_model):
        """
        Create a violin plot of per-fold accuracies for each model.
        """
        violin_rows = []
        for model_name, acc_list in accuracies_per_model.items():
            for acc in acc_list:
                violin_rows.append({"Model": model_name, "Accuracy": acc})
        if not violin_rows:
            return
        violin_df = pd.DataFrame(violin_rows)
        plt.figure(figsize=(12, 8))
        sns.violinplot(x="Model", y="Accuracy", data=violin_df, inner="quartile")
        plt.xlabel("Model")
        plt.ylabel("Accuracy")
        plt.title("Model Accuracy Comparison (per-fold distributions)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # ----------------------------#
    #        PUBLIC METHODS       #
    # ----------------------------# 
    def fit_evaluate_store_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                                  output_dir: str = "../results/all_models",
                                  cv_splits: int = 5, cv_repeats: int = 3):
        """
        Train, evaluate, and store multiple classifiers using Repeated Stratified CV.  
        Generates metrics, per-class reports, confusion matrices, ROC curves, and saves trained models.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix.
        y_train : pd.Series
            Training labels.
        output_dir : str, default="../results/all_models"
            Directory to save trained models.
        cv_splits : int, default=5
            Number of folds in cross-validation.
        cv_repeats : int, default=3
            Number of times cross-validation is repeated.

        Returns
        -------
        dict
            {
                "results_df": pd.DataFrame  # Global metrics per model
                "per_class_df": pd.DataFrame  # Metrics per class
            }
        """
        # Ensure output directory exists
        self._ensure_dir(output_dir)

        # CV Splitter
        cv_splitter = RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=42)
        classes = np.unique(y_train)

        metrics_list = []
        per_class_metrics_list = []
        confusion_dict = {}
        confusion_norm_dict = {}
        roc_dict = {}
        saved_model_paths = {}
        accuracies_per_model = {}

        for clf_name, clf in self.classifiers.items():
            print(f"Training & Evaluating: {clf_name}")

            # Repeated CV
            true_all, pred_all, prob_all, fold_accuracies = self._run_repeated_cv(clf, X_train, y_train, cv_splitter)

            # Store fold accuracies
            accuracies_per_model[clf_name] = fold_accuracies

            # Compute global metrics
            roc_auc_macro = np.nan
            try:
                roc_auc_macro = roc_auc_score(label_binarize(true_all, classes=classes), prob_all,
                                              average="macro", multi_class="ovr")
            except Exception:
                roc_auc_macro = np.nan

            clf_metrics = {
                "Model": clf_name,
                "Accuracy": accuracy_score(true_all, pred_all),
                "Balanced Accuracy": balanced_accuracy_score(true_all, pred_all),
                "Precision (weighted)": precision_score(true_all, pred_all, average="weighted", zero_division=0),
                "Recall (weighted)": recall_score(true_all, pred_all, average="weighted"),
                "F1 Score (weighted)": f1_score(true_all, pred_all, average="weighted"),
                "F1 Score (macro)": f1_score(true_all, pred_all, average="macro"),
                "ROC AUC (macro)": roc_auc_macro
            }
            metrics_list.append(clf_metrics)

            # Per-class metrics using classification_report
            class_report = classification_report(true_all, pred_all, labels=classes, output_dict=True, zero_division=0)
            for cls in classes:
                rep = class_report.get(str(cls), {})
                per_class_metrics_list.append({
                    "Model": clf_name,
                    "Class": cls,
                    "Precision": rep.get("precision", 0.0),
                    "Recall": rep.get("recall", 0.0),
                    "F1 Score": rep.get("f1-score", 0.0),
                    "Support": rep.get("support", 0)
                })

            # Confusion matrix counts and normalized
            cm = confusion_matrix(true_all, pred_all, labels=classes)
            confusion_dict[clf_name] = cm
            row_sums = cm.sum(axis=1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                cm_norm = np.divide(cm, row_sums, where=row_sums != 0)
                cm_norm = np.nan_to_num(cm_norm)
            confusion_norm_dict[clf_name] = cm_norm

            # ROC One-vs-Rest per class
            y_true_bin = label_binarize(true_all, classes=classes)
            fpr_dict, tpr_dict, auc_dict = {}, {}, {}
            for i, cls in enumerate(classes):
                if y_true_bin[:, i].sum() == 0:
                    fpr_dict[cls] = np.array([0.0, 1.0])
                    tpr_dict[cls] = np.array([0.0, 1.0])
                    auc_dict[cls] = np.nan
                    continue
                try:
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], prob_all[:, i])
                    auc_val = roc_auc_score(y_true_bin[:, i], prob_all[:, i])
                except Exception:
                    fpr, tpr, auc_val = np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.nan
                fpr_dict[cls] = fpr
                tpr_dict[cls] = tpr
                auc_dict[cls] = auc_val
            roc_dict[clf_name] = (fpr_dict, tpr_dict, auc_dict)

            # Refit classifier on entire training set and save
            fitted_full = self._safe_clone_and_fit(clf, X_train, y_train)
            model_path = self._unique_model_path(output_dir, clf_name)
            # Save as .pkl using pickle
            try:
                with open(model_path, "wb") as _f:
                    pickle.dump(fitted_full, _f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:
                # Fallback to joblib
                joblib.dump(fitted_full, model_path)
            saved_model_paths[clf_name] = model_path

        # Assemble result DataFrames
        results_df = pd.DataFrame(metrics_list).sort_values("ROC AUC (macro)", ascending=False)
        per_class_df = pd.DataFrame(per_class_metrics_list)

        # CHANGE WITH "PRINT" IF YOU WANT TO USE OUTSIDE OF IPYTHON
        display(results_df)
        display(per_class_df)

        self._plot_roc_per_class(roc_dict, classes)
        self._plot_confusion_matrices(confusion_dict, title_prefix="Confusion Matrix")
        self._plot_confusion_matrices(confusion_norm_dict, title_prefix="Normalized Confusion Matrix")
        self._plot_violin(accuracies_per_model)

        return {
            "results_df": results_df,
            "per_class_df": per_class_df
        }
    

    def evaluate_models_from_dir(self, models_dir: str,
                                X_train: pd.DataFrame, y_train: pd.Series,
                                X_test: pd.DataFrame, y_test: pd.Series,
                                cv_splits: int = 5, cv_repeats: int = 3,
                                display_individual_tables: bool = True):
        """
        Load pre-trained .joblib models from `models_dir` and evaluate each model on:
        - Training set
        - Repeated stratified Cross-Validation (using the model's hyperparameters)
        - Testing set

        For each model this function prints (via `display`) a small table with the same

        Parameters
        ----------
        models_dir : str
            Directory containing .joblib model files (assumed to be pre-trained).
        X_train, y_train : training data (pandas DataFrame/Series)
        X_test, y_test : testing data (pandas DataFrame/Series)
        cv_splits, cv_repeats : ints for RepeatedStratifiedKFold
        display_individual_tables : if True, calls `display` for each per-model table

        Returns
        -------
        dict with keys:
        - per_model_tables: dict mapping model_name -> DataFrame (Train/CV/Test metrics)
        - per_class_test: dict mapping model_name -> DataFrame (per-class metrics on test)
        - bar_plot_data: DataFrame summarizing balanced accuracies for plotting
        - confusion_matrices: dict mapping model_name -> raw confusion array (test)
        - confusion_matrices_norm: dict mapping model_name -> normalized confusion array (test)
        - test_comparison: DataFrame summarizing scores on test set
        """
        # Collect pkl files
        all_files = [f for f in os.listdir(models_dir) if f.lower().endswith('.pkl')]
        all_files = sorted(all_files)
        if len(all_files) == 0:
            raise ValueError(f"No .pkl files found in {models_dir}")

        cv_splitter = RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=42)
        classes = np.unique(np.concatenate([y_train, y_test]))

        per_model_tables = {}
        per_class_test = {}

        # For bar plot
        bar_rows = []

        # Store confusion matrices per model (raw and normalized)
        confusion_dict = {}
        confusion_norm_dict = {}

        for fname in all_files:
            model_path = os.path.join(models_dir, fname)
            try:
                with open(model_path, "rb") as _f:
                    loaded = pickle.load(_f)
            except Exception as e:
                # Try joblib.load as a fallback for backward compatibility
                try:
                    loaded = joblib.load(model_path)
                except Exception:
                    print(f"Skipping {fname}: failed to load ({e})")
                    continue

            model_name = os.path.splitext(fname)[0]
            print(f"Evaluating model: {model_name}")

            # TRAIN EVALUATION
            try:
                y_train_pred = loaded.predict(X_train)
            except Exception as e:
                print(f"Warning: model {model_name} couldn't predict on X_train directly: {e} -- will clone+fit on full train for train/test predictions")
                fitted_full = self._safe_clone_and_fit(loaded, X_train, y_train)
                y_train_pred = fitted_full.predict(X_train)
                loaded = fitted_full  # use this fitted version for test preds as well

            prob_train = self._get_probabilities(loaded, X_train, classes)
            roc_auc_train = np.nan
            try:
                roc_auc_train = roc_auc_score(label_binarize(y_train, classes=classes), prob_train,
                                            average="macro", multi_class="ovr")
            except Exception:
                roc_auc_train = np.nan

            train_metrics = {
                "Split": "Train",
                "Accuracy": accuracy_score(y_train, y_train_pred),
                "Balanced Accuracy": balanced_accuracy_score(y_train, y_train_pred),
                "Precision (weighted)": precision_score(y_train, y_train_pred, average="weighted", zero_division=0),
                "Recall (weighted)": recall_score(y_train, y_train_pred, average="weighted"),
                "F1 Score (weighted)": f1_score(y_train, y_train_pred, average="weighted"),
                "F1 Score (macro)": f1_score(y_train, y_train_pred, average="macro"),
                "ROC AUC (macro)": roc_auc_train
            }

            # CROSS-VALIDATION EVALUATION (clone model and run repeated CV)
            true_cv, pred_cv, prob_cv, fold_accuracies = self._run_repeated_cv(loaded, X_train, y_train, cv_splitter)
            roc_auc_cv = np.nan
            try:
                roc_auc_cv = roc_auc_score(label_binarize(true_cv, classes=classes), prob_cv,
                                        average="macro", multi_class="ovr")
            except Exception:
                roc_auc_cv = np.nan

            cv_metrics = {
                "Split": "CrossVal",
                "Accuracy": accuracy_score(true_cv, pred_cv),
                "Balanced Accuracy": balanced_accuracy_score(true_cv, pred_cv),
                "Precision (weighted)": precision_score(true_cv, pred_cv, average="weighted", zero_division=0),
                "Recall (weighted)": recall_score(true_cv, pred_cv, average="weighted"),
                "F1 Score (weighted)": f1_score(true_cv, pred_cv, average="weighted"),
                "F1 Score (macro)": f1_score(true_cv, pred_cv, average="macro"),
                "ROC AUC (macro)": roc_auc_cv
            }

            # TEST EVALUATION
            try:
                y_test_pred = loaded.predict(X_test)
            except Exception as e:
                print(f"Warning: model {model_name} couldn't predict on X_test directly: {e} -- will clone+fit on full train and retry")
                fitted_full = self._safe_clone_and_fit(loaded, X_train, y_train)
                y_test_pred = fitted_full.predict(X_test)
                prob_test = self._get_probabilities(fitted_full, X_test, classes)
                # Ensure loaded points to a fitted estimator for consistency below
                loaded = fitted_full
            else:
                prob_test = self._get_probabilities(loaded, X_test, classes)

            roc_auc_test = np.nan
            try:
                roc_auc_test = roc_auc_score(label_binarize(y_test, classes=classes), prob_test,
                                            average="macro", multi_class="ovr")
            except Exception:
                roc_auc_test = np.nan

            test_metrics = {
                "Split": "Test",
                "Accuracy": accuracy_score(y_test, y_test_pred),
                "Balanced Accuracy": balanced_accuracy_score(y_test, y_test_pred),
                "Precision (weighted)": precision_score(y_test, y_test_pred, average="weighted", zero_division=0),
                "Recall (weighted)": recall_score(y_test, y_test_pred, average="weighted"),
                "F1 Score (weighted)": f1_score(y_test, y_test_pred, average="weighted"),
                "F1 Score (macro)": f1_score(y_test, y_test_pred, average="macro"),
                "ROC AUC (macro)": roc_auc_test
            }

            # Assemble per-model table
            model_table = pd.DataFrame([train_metrics, cv_metrics, test_metrics])
            model_table = model_table.set_index('Split')
            per_model_tables[model_name] = model_table

            if display_individual_tables:
                display(model_table)

            # Per-class metrics ONLY for test set (same style as fit_evaluate_store_models)
            class_report = classification_report(y_test, y_test_pred, labels=classes, output_dict=True, zero_division=0)
            per_class_rows = []
            for cls in classes:
                rep = class_report.get(str(cls), class_report.get(cls, {}))
                per_class_rows.append({
                    "Model": model_name,
                    "Class": cls,
                    "Precision": rep.get("precision", 0.0),
                    "Recall": rep.get("recall", 0.0),
                    "F1 Score": rep.get("f1-score", 0.0),
                    "Support": rep.get("support", 0)
                })
            per_class_df = pd.DataFrame(per_class_rows)
            per_class_test[model_name] = per_class_df
            if display_individual_tables:
                display(per_class_df)

            # CONFUSION MATRICES (TEST SET)
            cm = confusion_matrix(y_test, y_test_pred, labels=classes)
            confusion_dict[model_name] = cm

            # Normalized per-row (true) -> percentages
            row_sums = cm.sum(axis=1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                cm_norm = np.divide(cm.astype(float), row_sums, where=row_sums != 0)
                cm_norm = np.nan_to_num(cm_norm)
            confusion_norm_dict[model_name] = cm_norm

            # Collect bar plot rows
            bar_rows.append({"Model": model_name, "Split": "Train", "Balanced Accuracy": model_table.loc['Train', 'Balanced Accuracy']})
            bar_rows.append({"Model": model_name, "Split": "CrossVal", "Balanced Accuracy": model_table.loc['CrossVal', 'Balanced Accuracy']})
            bar_rows.append({"Model": model_name, "Split": "Test", "Balanced Accuracy": model_table.loc['Test', 'Balanced Accuracy']})

        # Create bar plot dataframe
        bar_df = pd.DataFrame(bar_rows)
        if not bar_df.empty:
            models = bar_df['Model'].unique()
            splits = ['Train', 'CrossVal', 'Test']

            y = np.arange(len(models))  # label locations (asse y)
            height = 0.2  # spessore barre

            fig, ax = plt.subplots(figsize=(8, max(6, len(models) * 0.5)))  # altezza aumentata per leggibilità

            for i, split in enumerate(splits):
                vals = [bar_df[(bar_df['Model'] == m) & (bar_df['Split'] == split)]['Balanced Accuracy'].values
                        for m in models]
                vals = [v[0] if len(v) > 0 else 0.0 for v in vals]
                ax.barh(y + (i - 1) * height, vals, height, label=split)

            ax.set_xlabel('Balanced Accuracy')
            ax.set_yticks(y)
            ax.set_yticklabels(models)
            ax.set_title('Balanced Accuracy by Model and Split')
            ax.legend()
            ax.set_xlim(0, 1)  # range Balanced Accuracy
            plt.tight_layout()
            plt.show()
        
        # Print confusion matrices (raw) as tables for each model (TEST SET)
        if confusion_dict:
            print("\nConfusion matrices (raw counts) - TEST SET:")
            for mname, cm in confusion_dict.items():
                df_cm = pd.DataFrame(cm, index=classes, columns=classes)
                df_cm.index.name = "True"
                df_cm.columns.name = "Pred"

            # Plot all raw confusion matrices as heatmaps using the class helper
            self._plot_confusion_matrices(confusion_dict, title_prefix="Confusion Matrix (Test)")

        #  Print confusion matrices (normalized) as tables for each model (TEST SET)
        if confusion_norm_dict:
            print("\nConfusion matrices (normalized by true-row) - TEST SET:")
            for mname, cmn in confusion_norm_dict.items():
                df_cmn = pd.DataFrame(cmn, index=classes, columns=classes)
                df_cmn.index.name = "True"
                df_cmn.columns.name = "Pred"
            
            # Plot all normalized confusion matrices as heatmaps using the class helper
            self._plot_confusion_matrices(confusion_norm_dict, title_prefix="Normalized Confusion Matrix (Test)")

        # Comparison table for TEST SET only. Gather test-rows from each model_table into a single DataFrame.
        test_rows = []
        for mname, table in per_model_tables.items():
            if 'Test' in table.index:
                tr = table.loc['Test'].to_dict()
                tr['Model'] = mname
                test_rows.append(tr)
            else:
                # If unexpectedly missing, create NaN row
                test_rows.append({
                    "Model": mname,
                    "Accuracy": np.nan,
                    "Balanced Accuracy": np.nan,
                    "Precision (weighted)": np.nan,
                    "Recall (weighted)": np.nan,
                    "F1 Score (weighted)": np.nan,
                    "F1 Score (macro)": np.nan,
                    "ROC AUC (macro)": np.nan
                })

        if test_rows:
            test_comparison_df = pd.DataFrame(test_rows)
            # Set Model as index and reorder columns for readability
            test_comparison_df = test_comparison_df.set_index('Model')[
                ["Accuracy", "Balanced Accuracy", "Precision (weighted)", "Recall (weighted)",
                "F1 Score (weighted)", "F1 Score (macro)", "ROC AUC (macro)"]
            ]
            # Sort by Balanced Accuracy descending
            test_comparison_df = test_comparison_df.sort_values("Balanced Accuracy", ascending=False)
            print("\nOverall comparison on TEST SET (sorted by Balanced Accuracy):")
            display(test_comparison_df)
        else:
            test_comparison_df = pd.DataFrame()

        return {
            'per_model_tables': per_model_tables,
            'per_class_test': per_class_test,
            'bar_plot_data': bar_df,            
            'confusion_matrices': confusion_dict,
            'confusion_matrices_norm': confusion_norm_dict,
            'test_comparison': test_comparison_df
        }
