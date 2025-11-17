import os
import math
import matplotlib.pyplot as plt
import numpy as np
import joblib
import pandas as pd
import pickle
import seaborn as sns

from CogniPredictAD.preprocessing import ADNIPreprocessor
from IPython.display import display
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, auc
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from typing import List, Optional, Dict, Tuple, Union

class ADNIEvaluator:
    """
    ADNIEvaluator
    Helper to evaluate multiple classifier models on learn/train/test sets.

    The class can preload .pkl models from directories. It runs repeated stratified
    cross-validation with optional preprocessing, computes train/validation/test
    metrics and plots confusion matrices, ROC curves and violin plots.
    """
    # ----------------------------#
    #        INITIALIZATION       #
    # ----------------------------# 
    def __init__(self, model_dirs: Optional[Union[str, List[str]]] = None, sampling_title: bool = False):
        # Store loaded models and their source paths.
        self.models: Dict[str, object] = {}
        self.model_paths: Dict[str, str] = {}
        # If True use "_Sampled" suffixing scheme, else use "_Alternative"
        self.sampling_title: bool = sampling_title

        if model_dirs:
            if isinstance(model_dirs, str):
                model_dirs = [model_dirs]
            self._load_models_from_dirs(model_dirs)

    # ----------------------------#
    #          UTILITIES          #
    # ----------------------------# 
    def _softmax(self, arr: np.ndarray) -> np.ndarray:
        """Compute row-wise softmax for a 2D array in a numerically-stable way."""
        e = np.exp(arr - np.max(arr, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    def _get_probabilities(self, fitted_clf, X: pd.DataFrame, classes: np.ndarray) -> np.ndarray:
        """
        Produce a (n_samples, n_classes) probability matrix aligned to `classes`.

        Attempts, in order:
        1. predict_proba (align and renormalize columns to `classes`).
        2. decision_function (softmax over scores).
        3. Fallback to one-hot from predict.
        Returns tiny non-zero values for missing classes where needed.
        """
        # Try predict_proba on estimator or final step of an imblearn Pipeline
        if hasattr(fitted_clf, "predict_proba"):
            try:
                probs = fitted_clf.predict_proba(X)
                prob_cols = getattr(fitted_clf, "classes_", None)

                # If model is a Pipeline, try the final estimator for classes_
                if prob_cols is None and isinstance(fitted_clf, Pipeline):
                    final_estimator = list(fitted_clf.named_steps.values())[-1]
                    prob_cols = getattr(final_estimator, "classes_", None)

                if prob_cols is None:
                    prob_cols = np.arange(probs.shape[1])

                # Align columns to requested classes and renormalize rows
                prob_df = pd.DataFrame(probs, columns=prob_cols)
                prob_df = prob_df.reindex(columns=classes, fill_value=1e-6)
                prob_df = prob_df.div(prob_df.sum(axis=1), axis=0)
                return prob_df.values
            except Exception:
                # If predict_proba fails, fall through to next method.
                pass

        # Try decision_function and convert scores to probabilities via softmax
        if hasattr(fitted_clf, "decision_function"):
            try:
                df = fitted_clf.decision_function(X)
                # For binary one-dimensional output, build two-column representation
                if df.ndim == 1:
                    df = np.vstack([-df, df]).T
                probs = self._softmax(df)
                prob_df = pd.DataFrame(probs, columns=classes[:probs.shape[1]])
                prob_df = prob_df.reindex(columns=classes, fill_value=1e-6)
                prob_df = prob_df.div(prob_df.sum(axis=1), axis=0)
                return prob_df.values
            except Exception:
                pass

        # Fallback: one-hot encoding of hard predictions
        preds = fitted_clf.predict(X)
        one_hot = np.zeros((len(preds), len(classes)))
        class_to_idx = {c: i for i, c in enumerate(classes)}
        for i, p in enumerate(preds):
            if p in class_to_idx:
                one_hot[i, class_to_idx[p]] = 1.0
        return one_hot

    def _safe_clone_and_fit(self, clf, X_train: pd.DataFrame, y_train: pd.Series):
        """Clone an estimator and fit it on X_train/y_train. Returns the fitted clone."""
        cloned = clone(clf)
        cloned.fit(X_train, y_train)
        return cloned

    def _ensure_dir(self, path: str):
        """Create directory if it does not exist."""
        os.makedirs(path, exist_ok=True)

    def _clean_name(self, name: str) -> str:
        """Return a filesystem-safe model name by replacing spaces and slashes."""
        return name.replace(" ", "_").replace("/", "_")

    def _unique_model_path(self, base_dir: str, clf_name: str) -> str:
        """Generate a unique .pkl path under base_dir for a given classifier name."""
        clean_name = self._clean_name(clf_name)
        i = 0
        while True:
            candidate = os.path.join(base_dir, f"{clean_name}{i}.pkl")
            if not os.path.exists(candidate):
                return candidate
            i += 1

    def _unique_name(self, base: str) -> str:
        """Return a unique key for storing a model in self.models."""
        name = base
        if name not in self.models:
            return name
        # Choose suffix based on sampling_title flag
        suffix = "_Sampled" if self.sampling_title else "_Alternative"

        base_candidate = name + suffix
        if base_candidate not in self.models:
            return base_candidate

        counter = 0
        while True:
            candidate = f"{name}{suffix}{counter}"
            if candidate not in self.models:
                return candidate
            counter += 1

    def _try_load(self, path: str):
        """Attempt to load a model via pickle, then joblib. Return None on failure."""
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            try:
                return joblib.load(path)
            except Exception:
                return None

    # ----------------------------#
    #        MODEL LOADING        #
    # ----------------------------# 
    def _load_models_from_dirs(self, dirs: List[str]):
        """Load all .pkl models from a list of directories into self.models."""
        for d in dirs:
            if not os.path.isdir(d):
                print(f"Skipping non-directory: {d}")
                continue
            for fname in sorted(f for f in os.listdir(d) if f.lower().endswith(".pkl")):
                path = os.path.join(d, fname)
                loaded = self._try_load(path)
                if loaded is None:
                    print(f"Skipping {path}: failed to load")
                    continue
                name = self._unique_name(os.path.splitext(fname)[0])
                self.models[name] = loaded
                self.model_paths[name] = path
                print(f"Loaded model '{name}' from {path}")

    # ----------------------------#
    #     CORE EVALUATION (CV)    #
    # ----------------------------#  
    def _run_repeated_cv(self, clf, X: pd.DataFrame, y: pd.Series, cv_splitter: RepeatedStratifiedKFold,
                        classes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]:
        """
        Execute repeated stratified K-fold CV for a single classifier and return
        (true_labels, predicted_labels, prob_matrix, per_fold_accuracies).
        """
        true_all = []
        pred_all = []
        prob_list = []
        fold_accs = []

        for train_idx, val_idx in cv_splitter.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            fitted = self._safe_clone_and_fit(clf, X_tr, y_tr)
            y_pred = fitted.predict(X_val)
            prob_arr = self._get_probabilities(fitted, X_val, classes)

            true_all.extend(list(y_val))
            pred_all.extend(list(y_pred))
            prob_list.append(prob_arr)
            fold_accs.append(accuracy_score(y_val, y_pred))

        prob_all = np.vstack(prob_list) if prob_list else np.empty((0, len(classes)))
        return np.array(true_all), np.array(pred_all), prob_all, fold_accs

    def _safe_roc_auc(self, y_true, prob, classes: np.ndarray) -> float:
        """Compute macro ROC AUC safely, returning nan on failure."""
        try:
            return roc_auc_score(label_binarize(y_true, classes=classes), prob,
                                 average="macro", multi_class="ovr")
        except Exception:
            return np.nan

    def _metrics_dict(self, y_true, y_pred, prob, classes, split_name: str) -> Dict:
        """Assemble commonly used metrics into a dictionary for a given split."""
        return {
            "Split": split_name,
            "F1 Score (macro)": f1_score(y_true, y_pred, average="macro"),
            "Accuracy": accuracy_score(y_true, y_pred),
            "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
            "Precision (weighted)": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "Recall (weighted)": recall_score(y_true, y_pred, average="weighted"),
            "F1 Score (weighted)": f1_score(y_true, y_pred, average="weighted"),
            "ROC AUC (macro)": self._safe_roc_auc(y_true, prob, classes)
        }

    # ----------------------------#
    #      PLOTTING HELPERS       #
    # ----------------------------# 
    def _plot_roc_per_class(self, roc_dict: Dict, classes: List[str]):
        """
        Plot ROC curves per class for all models in roc_dict.

        roc_dict format: {model_name: (fpr_dict, tpr_dict, auc_dict)} where each dict maps class->array.
        """
        if not classes:
            return

        n_cols = 2
        n_rows = math.ceil(len(classes) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
        axes = np.array(axes).reshape(-1)

        for idx, cls in enumerate(classes):
            ax = axes[idx]
            for name, (fpr_d, tpr_d, auc_d) in roc_dict.items():
                fpr = fpr_d.get(cls, np.array([]))
                tpr = tpr_d.get(cls, np.array([]))
                auc_v = auc_d.get(cls, np.nan)
                if fpr.size == 0 or tpr.size == 0:
                    fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])
                    label = f"{name} (AUC=nan)"
                else:
                    label = f"{name} (AUC={'nan' if np.isnan(auc_v) else f'{auc_v:.2f}'})"
                ax.plot(fpr, tpr, lw=2, label=label)

            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.01])
            ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve - Class {cls} (One-vs-Rest)")
            ax.legend(loc="lower right", fontsize="small"); ax.grid(alpha=0.2)

        # Remove any unused axes
        total = axes.size
        for j in range(len(classes), total):
            fig.delaxes(axes[j])

        plt.tight_layout(); plt.show()

    def _plot_confusion_matrices(self, cm_dict: Dict[str, np.ndarray], title_prefix: str = "Confusion Matrix"):
        """
        Plot confusion matrices (either raw counts or normalized) from cm_dict.
        Each value in cm_dict is an (n_classes, n_classes) array.
        """
        if not cm_dict:
            return

        n_models = len(cm_dict)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        first_shape = next(iter(cm_dict.values())).shape
        n_classes = first_shape[0]


        fig, axes = plt.subplots(n_rows, n_cols, figsize=(17, 17))
        axes = np.array(axes).reshape(-1)

        xlabel_pad = 6
        ylabel_pad = 6

        for idx, (name, cm) in enumerate(cm_dict.items()):
            ax = axes[idx]
            cm_arr = np.asarray(cm, dtype=float)
            fmt = "d" if np.allclose(cm_arr, np.round(cm_arr)) else ".2f"
            annot = cm_arr.astype(int) if fmt == "d" else cm_arr
            sns.heatmap(cm_arr, annot=annot, fmt=fmt, cmap="Blues", cbar=False, ax=ax, square=True)
            ax.set_title(f"{name} {title_prefix}", pad=10) # Add extra vertical padding
            ax.set_xlabel("Predicted")
            ax.xaxis.labelpad = xlabel_pad
            ax.set_ylabel("True")
            ax.yaxis.labelpad = ylabel_pad

        # Remove unused axes
        for idx in range(len(cm_dict), axes.size):
            fig.delaxes(axes[idx])

        # Increase spacing between subplots so titles/labels do not overlap.
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.6, wspace=0.4)
        plt.show()

    def _plot_violin(self, accuracies_per_model: Dict[str, List[float]]):
        """Create a violin plot showing accuracy distributions across CV folds for each model."""
        rows = []
        for name, accs in accuracies_per_model.items():
            for a in accs:
                rows.append({"Model": name, "Accuracy": a})
        if not rows:
            return
        df = pd.DataFrame(rows)
        plt.figure(figsize=(12, 8))
        sns.violinplot(x="Model", y="Accuracy", data=df, inner="quartile")
        plt.xlabel("Model"); plt.ylabel("Accuracy"); plt.title("Model Accuracy Comparison (per-fold distributions)")
        plt.xticks(rotation=45); plt.tight_layout(); plt.show()

    # ----------------------------#
    #     EVALUATION HELPERS      #
    # ----------------------------# 
    def _evaluate_train_direct(self, model_name: str, loaded, X_train, y_train, classes):
        """
        Try predicting with loaded model on X_train; if that fails,
        clone+fit on full train and return metrics for the Train split.
        """
        if X_train is None or y_train is None:
            return loaded, None

        working = loaded
        X_used = X_train
        try:
            y_pred = working.predict(X_used)
            prob = self._get_probabilities(working, X_used, classes)
        except Exception as e:
            print(f"Warning: model {model_name} couldn't predict on X_train directly: {e} -- will clone+fit on full train for train/test predictions")
            working = self._safe_clone_and_fit(loaded, X_used, y_train)
            y_pred = working.predict(X_used)
            prob = self._get_probabilities(working, X_used, classes)

        return working, self._metrics_dict(y_train, y_pred, prob, classes, "Train")

    def _evaluate_learn_cv(self, model_name: str, loaded, X_learn, y_learn, cv_splitter, classes):
        """
        Build a CV pipeline (optional ADNIPreprocessor + optional scaler + final estimator)
        and run repeated CV. Returns (fold_accuracies, crossval_metrics).
        """
        if X_learn is None or y_learn is None:
            return [], None

        final_est = loaded
        if isinstance(loaded, Pipeline):
            final_est = list(loaded.named_steps.values())[-1]

        needs_scaling = isinstance(final_est, LogisticRegression)
        has_preprocessor = isinstance(loaded, Pipeline) and isinstance(list(loaded.named_steps.values())[0], ADNIPreprocessor)

        steps = []
        if not has_preprocessor:
            steps.append(("pre", ADNIPreprocessor()))
        if needs_scaling:
            steps.append(("scl", StandardScaler()))
        steps.append(("clf", final_est))
        cv_pipeline = Pipeline(steps)

        true_cv, pred_cv, prob_cv, fold_accs = self._run_repeated_cv(cv_pipeline, X_learn, y_learn, cv_splitter, classes)
        return fold_accs, self._metrics_dict(true_cv, pred_cv, prob_cv, classes, "CrossVal")

    def _evaluate_test_direct(self, model_name: str, working, loaded, X_test, y_test, X_train, y_train, X_learn, y_learn, classes):
        """
        Try predicting on test with 'working'. If predict fails, try to fit on train/learn and retry.
        Returns (working_estimator, test_metrics, y_test_pred, prob_test).
        """
        if X_test is None or y_test is None:
            raise ValueError("X_test and y_test are required for test evaluation.")

        X_test_used = X_test
        try:
            y_pred = working.predict(X_test_used)
            prob = self._get_probabilities(working, X_test_used, classes)
        except Exception as e:
            print(f"Warning: model {model_name} couldn't predict on X_test directly: {e} -- will clone+fit on available data and retry")
            fitted = None
            if X_train is not None and y_train is not None:
                fitted = self._safe_clone_and_fit(loaded, X_train, y_train)
            elif X_learn is not None and y_learn is not None:
                fitted = self._safe_clone_and_fit(loaded, X_learn, y_learn)

            if fitted is None:
                raise RuntimeError(f"Model {model_name} cannot produce predictions on X_test and no train/learn data available to fit.")
            working = fitted
            y_pred = working.predict(X_test_used)
            prob = self._get_probabilities(working, X_test_used, classes)

        return working, self._metrics_dict(y_test, y_pred, prob, classes, "Test"), y_pred, prob

    # ----------------------------#
    #   PLOT AND REPORT HELPERS   #
    # ----------------------------# 
    def _prepare_per_class_test(self, model_name: str, y_test, y_test_pred, classes):
        """Prepare a per-class table (precision/recall/f1/support) for the test split."""
        report = classification_report(y_test, y_test_pred, labels=classes, output_dict=True, zero_division=0)
        rows = []
        for cls in classes:
            rep = report.get(str(cls), report.get(cls, {}))
            rows.append({
                "Model": model_name,
                "Class": cls,
                "Precision": rep.get("precision", 0.0),
                "Recall": rep.get("recall", 0.0),
                "F1 Score": rep.get("f1-score", 0.0),
                "Support": rep.get("support", 0)
            })
        return pd.DataFrame(rows)

    # ----------------------------#
    #         PUBLIC API          #
    # ----------------------------# 
    def evaluate_models(self,
                        models_dir: Optional[str] = None,
                        X_learn: Optional[pd.DataFrame] = None, y_learn: Optional[pd.Series] = None,
                        X_train: Optional[pd.DataFrame] = None, y_train: Optional[pd.Series] = None,
                        X_test: Optional[pd.DataFrame] = None, y_test: Optional[pd.Series] = None,
                        cv_splits: int = 5, cv_repeats: int = 3,
                        display_individual_tables: bool = True,
                        plot_roc: bool = True,
                        save_results_dir: Optional[str] = None):
        """
        Evaluate available models over Learn (CV), Train, and Test.

        Returns a dictionary containing per-model tables, per-class test DataFrames,
        confusion matrices, and a test comparison table.
        """
        if X_test is None or y_test is None:
            raise ValueError("X_test and y_test are required.")

        # Determine models to evaluate. If models_dir provided, load from there temporarily.
        if models_dir:
            tmp = ADNIEvaluator(sampling_title=self.sampling_title)
            tmp._load_models_from_dirs([models_dir])
            models_to_eval, model_paths = tmp.models, tmp.model_paths
        elif self.models:
            models_to_eval, model_paths = self.models, self.model_paths
        else:
            raise ValueError("No models available. Provide models_dir or preload via __init__.")

        # Build class ordering from available label sources
        label_sources = [np.asarray(arr) for arr in (y_learn, y_train, y_test) if arr is not None]
        classes = np.unique(np.concatenate(label_sources)) if label_sources else np.array([])

        cv_splitter = RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=42)

        # Containers for results and plotting
        per_model_tables, per_class_test = {}, {}
        confusion_dict, confusion_norm = {}, {}
        bar_rows = {}
        accuracies_per_model = {}
        roc_data_for_plotting = {}

        for model_name, loaded in models_to_eval.items():
            print(f"Evaluating: {model_name}")

            # Train evaluation (try direct predict, otherwise fit)
            working, train_metrics = self._evaluate_train_direct(model_name, loaded, X_train, y_train, classes)

            # Cross-validation on learn set
            fold_accs, cv_metrics = self._evaluate_learn_cv(model_name, loaded, X_learn, y_learn, cv_splitter, classes)
            if fold_accs:
                accuracies_per_model[model_name] = fold_accs

            # Test evaluation (try with 'working', else fit available data)
            working, test_metrics, y_test_pred, prob_test = self._evaluate_test_direct(
                model_name, working, loaded, X_test, y_test, X_train, y_train, X_learn, y_learn, classes
            )

            # Assemble per-model table and optionally display it
            rows = [m for m in (train_metrics, cv_metrics, test_metrics) if m is not None]
            model_table = pd.DataFrame(rows).set_index("Split") if rows else pd.DataFrame()
            per_model_tables[model_name] = model_table
            if display_individual_tables and not model_table.empty:
                display(model_table)

            # Per-class test table
            per_class_test[model_name] = self._prepare_per_class_test(model_name, y_test, y_test_pred, classes)
            if display_individual_tables:
                display(per_class_test[model_name])

            # Confusion matrices (raw and normalized by true-row)
            cm = confusion_matrix(y_test, y_test_pred, labels=classes)
            confusion_dict[model_name] = cm
            with np.errstate(divide="ignore", invalid="ignore"):
                rowsum = cm.sum(axis=1, keepdims=True)
                cm_norm = np.divide(cm.astype(float), rowsum, where=rowsum != 0)
                cm_norm = np.nan_to_num(cm_norm)
            confusion_norm[model_name] = cm_norm

            # Collect F1 macro for bar plotting
            if "Train" in model_table.index:
                bar_rows.setdefault(model_name, []).append({"Model": model_name, "Split": "Train", "F1 Score (macro)": model_table.loc["Train", "F1 Score (macro)"]})
            if "CrossVal" in model_table.index:
                bar_rows.setdefault(model_name, []).append({"Model": model_name, "Split": "CrossVal", "F1 Score (macro)": model_table.loc["CrossVal", "F1 Score (macro)"]})
            if "Test" in model_table.index:
                bar_rows.setdefault(model_name, []).append({"Model": model_name, "Split": "Test", "F1 Score (macro)": model_table.loc["Test", "F1 Score (macro)"]})

            # Prepare ROC data for plotting per class
            if prob_test is not None:
                try:
                    y_test_bin = label_binarize(y_test, classes=classes)
                    fpr_d, tpr_d, auc_d = {}, {}, {}
                    for i, cls in enumerate(classes):
                        if i >= prob_test.shape[1]:
                            fpr, tpr, auc_v = np.array([]), np.array([]), np.nan
                        else:
                            fpr, tpr, _ = roc_curve(y_test_bin[:, i], prob_test[:, i])
                            try:
                                auc_v = auc(fpr, tpr)
                            except Exception:
                                auc_v = np.nan
                        fpr_d[cls], tpr_d[cls], auc_d[cls] = fpr, tpr, auc_v
                    roc_data_for_plotting[model_name] = (fpr_d, tpr_d, auc_d)
                except Exception as e:
                    print(f"Failed preparing ROC data for {model_name}: {e}")

        # Build bar DataFrame for F1 comparison and plot
        flat_bar = [r for v in bar_rows.values() for r in v]
        bar_df = pd.DataFrame(flat_bar)

        if not bar_df.empty:
            table_df = bar_df.pivot(index='Model', columns='Split', values='F1 Score (macro)')
            # Ensure all splits exist as columns
            for split in ['Train', 'CrossVal', 'Test']:
                if split not in table_df.columns:
                    table_df[split] = np.nan
            table_df = table_df[['Train', 'CrossVal', 'Test']]  # reorder columns
            print("\nF1 Score (macro) per model and split:")
            display(table_df)

            # Horizontal bar graph
            def model_f1_score(m):
                for s in ("Test", "CrossVal", "Train"):
                    r = bar_df[(bar_df['Model'] == m) & (bar_df['Split'] == s)]
                    if not r.empty:
                        return float(r['F1 Score (macro)'].iloc[0])
                return -np.inf

            models_ordered = sorted(bar_df['Model'].unique(), key=model_f1_score, reverse=True)
            y_pos = np.arange(len(models_ordered))
            height = 0.2
            splits = ['Train', 'CrossVal', 'Test']
            fig, ax = plt.subplots(figsize=(8, max(6, len(models_ordered) * 0.5)))
            for i, split in enumerate(splits):
                vals = [bar_df[(bar_df['Model'] == m) & (bar_df['Split'] == split)]['F1 Score (macro)'].values for m in models_ordered]
                vals = [v[0] if len(v) > 0 else 0.0 for v in vals]
                ax.barh(y_pos + (i - 1) * height, vals, height, label=split)
            ax.set_xlabel('F1 Score (macro)'); ax.set_yticks(y_pos); ax.set_yticklabels(models_ordered)
            ax.set_title('F1 Score (macro) by Model and Split'); ax.legend(); ax.set_xlim(0, 1)
            plt.tight_layout(); plt.show()

        # Confusion matrices: raw and normalized
        if confusion_dict:
            print("\nConfusion matrices (raw counts) - TEST SET:")
            self._plot_confusion_matrices(confusion_dict, title_prefix="Confusion Matrix (Test)")
        if confusion_norm:
            print("\nConfusion matrices (normalized by true-row) - TEST SET:")
            self._plot_confusion_matrices(confusion_norm, title_prefix="Normalized Confusion Matrix (Test)")

        # ROC plotting per class across models
        if plot_roc and roc_data_for_plotting:
            self._plot_roc_per_class(roc_data_for_plotting, list(classes))

        # Overall test comparison table
        test_rows = []
        for mname, table in per_model_tables.items():
            if 'Test' in table.index:
                tr = table.loc['Test'].to_dict(); tr['Model'] = mname; test_rows.append(tr)
            else:
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
            test_comparison_df = pd.DataFrame(test_rows).set_index('Model')[[ 
                "F1 Score (macro)", "Accuracy", "Balanced Accuracy", 
                "Precision (weighted)", "Recall (weighted)",
                "F1 Score (weighted)", "ROC AUC (macro)"
            ]]
            if "F1 Score (macro)" in test_comparison_df.columns:
                test_comparison_df = test_comparison_df.sort_values("F1 Score (macro)", ascending=False)
            print("\nOverall comparison on TEST SET (sorted by F1 Score (macro)):") 
            display(test_comparison_df)
        else:
            test_comparison_df = pd.DataFrame()

        # Optional save of the comparison table
        if save_results_dir:
            os.makedirs(save_results_dir, exist_ok=True)
            test_comparison_df.to_csv(os.path.join(save_results_dir, 'test_comparison.csv'))
            print(f"Saved test_comparison.csv to {save_results_dir}")

        # Return structured results unchanged in semantics from original implementation
        return {
            'per_model_tables': per_model_tables,
            'per_class_test': per_class_test,
            'bar_plot_data': bar_df,
            'confusion_matrices': confusion_dict,
            'confusion_matrices_norm': confusion_norm,
            'test_comparison': test_comparison_df
        }
