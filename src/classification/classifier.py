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
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier


class ADNIClassifier:
    """
    ADNIClassifier
    --------------
    Helper to train and evaluate a set of classification pipelines on an ADNIMERGE-style dataset.

    Key Features:
      - Provide standard and alternative pre-configured classifier pipelines. 
      - Run repeated stratified outer CV to obtain unbiased per-fold performance samples.
      - Compute aggregated metrics and per-class reports across outer folds.
      - Produce visualizations (ROC per class, confusion matrices, violin plots).
      - Perform pairwise Wilcoxon signed-rank tests on outer-fold F1-macro scores.
      - Refit chosen pipelines on full training data (with sampling adjustments) and persist a reduced
        pipeline (scaler + classifier) to disk.
      - All saved models use a deterministic .pkl path returned by _unique_model_path.
    
    Public API:
        fit_evaluate_store_models
    """
    # ----------------------------#
    #        INITIALIZATION       #
    # ----------------------------# 
    def __init__(self, classifier: str = "standard"):
        """
        Initialize the class and load a predefined set of classifiers based on the provided selection string.
        """
        classifier = classifier.lower()
        if classifier == "standard" or classifier == "None":
            self.classifiers = self._default_classifiers()
        elif classifier == "alternative":
            self.classifiers = self._alernative_classifiers()
        else:
            raise ValueError(f"Unknown classifier preset: {classifier}")

    # ----------------------------#
    #   CLASSIFIERS DEFINITIONS   #
    # ----------------------------# 
    def _default_classifiers(self):
        preprocessing = ADNIPreprocessor()
        categorical_features = [1, 3]  # PTGENDER, APOE4
        undersample_dict = {"CN": 385, "LMCI": 385}
        oversample_dict = {"EMCI": 385, "AD": 385}
        
        return {
                'Decision Tree': Pipeline([
                    ('pre', preprocessing), 
                    ('clf', DecisionTreeClassifier(
                        random_state=42, class_weight='balanced',
                        criterion='entropy', max_depth=5,
                        min_samples_split=2, min_samples_leaf=8,
                        ccp_alpha=0.005
                    ))
                ]),
                'Random Forest': Pipeline([
                    ('pre', preprocessing),
                    ('clf', RandomForestClassifier(
                        random_state=42, class_weight='balanced', n_jobs=-1,
                        criterion='entropy', max_depth=6, max_features=1.0,
                        min_samples_leaf=4, n_estimators=100
                    ))
                ]),
                'Extra Trees': Pipeline([
                    ('pre', preprocessing),
                    ('clf', ExtraTreesClassifier(
                        random_state=42, class_weight='balanced', n_jobs=-1,
                        criterion='entropy', max_depth=None, max_features=1.0,
                        min_samples_leaf=4, n_estimators=75
                    ))
                ]),
                'Adaptive Boosting': Pipeline([
                    ('pre', preprocessing),
                    ('clf', AdaBoostClassifier(
                        random_state=42,
                        estimator=DecisionTreeClassifier(
                            class_weight='balanced', criterion='gini',
                            max_depth=4, min_samples_leaf=8
                        ),
                        learning_rate=0.1,
                        n_estimators=50
                    ))
                ]),
                'Multinomial Logistic Regression': Pipeline([
                    ('pre', preprocessing),
                    ('scl', StandardScaler()), 
                    ('clf', LogisticRegression(
                        random_state=42, solver='saga', max_iter=2000,
                        class_weight='balanced', penalty='l1', C=0.1
                    ))
                ]),
                'Decision Tree Sampled': Pipeline([
                    ('pre', preprocessing), 
                    ('rus', RandomUnderSampler(sampling_strategy=undersample_dict, random_state=42)),
                    ('smotenc', SMOTENC(categorical_features=categorical_features, sampling_strategy=oversample_dict, random_state=42)),
                    ('clf', DecisionTreeClassifier(
                        random_state=42, class_weight='balanced',
                        criterion='gini', max_depth=5,
                        min_samples_split=2, min_samples_leaf=8,
                        ccp_alpha=0.0
                    ))
                ]),
                'Random Forest Sampled': Pipeline([
                    ('pre', preprocessing),
                    ('rus', RandomUnderSampler(sampling_strategy=undersample_dict, random_state=42)),
                    ('smotenc', SMOTENC(categorical_features=categorical_features, sampling_strategy=oversample_dict, random_state=42)),
                    ('clf', RandomForestClassifier(
                        random_state=42, class_weight='balanced', n_jobs=-1,
                        criterion='entropy', max_depth=6, max_features=1.0,
                        min_samples_leaf=4, n_estimators=50
                    ))
                ]),
                'Extra Trees Sampled': Pipeline([
                    ('pre', preprocessing),
                    ('rus', RandomUnderSampler(sampling_strategy=undersample_dict, random_state=42)),
                    ('smotenc', SMOTENC(categorical_features=categorical_features, sampling_strategy=oversample_dict, random_state=42)),
                    ('clf', ExtraTreesClassifier(
                        random_state=42, class_weight='balanced', n_jobs=-1,
                        criterion='gini', max_depth=None, max_features=1.0,
                        min_samples_leaf=8, n_estimators=75
                    ))
                ]),
                'Adaptive Boosting Sampled': Pipeline([
                    ('pre', preprocessing),
                    ('rus', RandomUnderSampler(sampling_strategy=undersample_dict, random_state=42)),
                    ('smotenc', SMOTENC(categorical_features=categorical_features, sampling_strategy=oversample_dict, random_state=42)),
                    ('clf', AdaBoostClassifier(
                        random_state=42,
                        estimator=DecisionTreeClassifier(
                            class_weight='balanced', criterion='gini',
                            max_depth=4, min_samples_leaf=8
                        ),
                        learning_rate=0.1,
                        n_estimators=50
                    ))
                ]),
                'Multinomial Logistic Regression Sampled': Pipeline([
                    ('pre', preprocessing),
                    ('scl', StandardScaler()),
                    ('rus', RandomUnderSampler(sampling_strategy=undersample_dict, random_state=42)),
                    ('smotenc', SMOTENC(categorical_features=categorical_features, sampling_strategy=oversample_dict, random_state=42)), 
                    ('clf', LogisticRegression(
                        random_state=42, solver='saga', max_iter=2000,
                        class_weight='balanced', penalty='l1', C=1.0
                    ))
                ])
            }
    
    def _alernative_classifiers(self):
        preprocessing = ADNIPreprocessor()
        categorical_features = [1, 3]  # PTGENDER, APOE4
        undersample_dict = {"CN": 385, "LMCI": 385}
        oversample_dict = {"EMCI": 385, "AD": 385}

        return {
            'Decision Tree': Pipeline([
                ('pre', preprocessing),
                ('clf', DecisionTreeClassifier(
                    random_state=42, class_weight='balanced',
                    criterion='gini', max_depth=5,
                    min_samples_split=2, min_samples_leaf=2,
                    ccp_alpha=0.01
                ))
            ]),
            'Random Forest': Pipeline([
                ('pre', preprocessing),
                ('clf', RandomForestClassifier(
                    random_state=42, class_weight='balanced', n_jobs=-1,
                    criterion='gini', max_depth=6, max_features=0.8,
                    min_samples_leaf=2, n_estimators=100
                ))
            ]),
            'Extra Trees': Pipeline([
                ('pre', preprocessing),
                ('clf', ExtraTreesClassifier(
                    random_state=42, class_weight='balanced', n_jobs=-1,
                    criterion='gini', max_depth=None, max_features=0.8,
                    min_samples_leaf=4, n_estimators=100
                ))
            ]),
            'Adaptive Boosting': Pipeline([
                ('pre', preprocessing),
                ('clf', AdaBoostClassifier(
                    random_state=42,
                    estimator=DecisionTreeClassifier(
                        class_weight='balanced', criterion='gini',
                        max_depth=6, min_samples_leaf=8
                    ),
                    learning_rate=0.05,
                    n_estimators=100
                ))
            ]),
            'Multinomial Logistic Regression': Pipeline([
                ('pre', preprocessing),
                ('scl', StandardScaler()),
                ('clf', LogisticRegression(
                    random_state=42, solver='saga', max_iter=2000,
                    class_weight='balanced', penalty='l2', C=1.0
                ))
            ]),
            'Decision Tree Sampled': Pipeline([
                ('pre', preprocessing),
                ('rus', RandomUnderSampler(sampling_strategy=undersample_dict, random_state=42)),
                ('smotenc', SMOTENC(categorical_features=categorical_features, sampling_strategy=oversample_dict, random_state=42)),
                ('clf', DecisionTreeClassifier(
                    random_state=42, class_weight='balanced',
                    criterion='entropy', max_depth=5,
                    min_samples_split=2, min_samples_leaf=2,
                    ccp_alpha=0.01
                ))
            ]),
            'Random Forest Sampled': Pipeline([
                ('pre', preprocessing),
                ('rus', RandomUnderSampler(sampling_strategy=undersample_dict, random_state=42)),
                ('smotenc', SMOTENC(categorical_features=categorical_features, sampling_strategy=oversample_dict, random_state=42)),
                ('clf', RandomForestClassifier(
                    random_state=42, class_weight='balanced', n_jobs=-1,
                    criterion='gini', max_depth=6, max_features=0.5,
                    min_samples_leaf=4, n_estimators=100
                ))
            ]),
            'Extra Trees Sampled': Pipeline([
                ('pre', preprocessing),
                ('rus', RandomUnderSampler(sampling_strategy=undersample_dict, random_state=42)),
                ('smotenc', SMOTENC(categorical_features=categorical_features, sampling_strategy=oversample_dict, random_state=42)),
                ('clf', ExtraTreesClassifier(
                    random_state=42, class_weight='balanced', n_jobs=-1,
                    criterion='gini', max_depth=6, max_features=1.0,
                    min_samples_leaf=8, n_estimators=100
                ))
            ]),
            'Adaptive Boosting Sampled': Pipeline([
                ('pre', preprocessing),
                ('rus', RandomUnderSampler(sampling_strategy=undersample_dict, random_state=42)),
                ('smotenc', SMOTENC(categorical_features=categorical_features, sampling_strategy=oversample_dict, random_state=42)),
                ('clf', AdaBoostClassifier(
                    random_state=42,
                    estimator=DecisionTreeClassifier(
                        class_weight='balanced', criterion='gini',
                        max_depth=6, min_samples_leaf=2
                    ),
                    learning_rate=0.05,
                    n_estimators=50
                ))
            ]),
            'Multinomial Logistic Regression Sampled': Pipeline([
                ('pre', preprocessing),
                ('scl', StandardScaler()),
                ('rus', RandomUnderSampler(sampling_strategy=undersample_dict, random_state=42)),
                ('smotenc', SMOTENC(categorical_features=categorical_features, sampling_strategy=oversample_dict, random_state=42)),
                ('clf', LogisticRegression(
                    random_state=42, solver='saga', max_iter=2000,
                    class_weight='balanced', penalty='l2', C=1.0
                ))
            ])
        }

    # ----------------------------#
    #          UTILITIES          #
    # ----------------------------# 
    def _softmax(self, arr):
        """
        Compute the softmax function on rows of an array to convert values (score/decision) into normalized probabilities.
        """
        e = np.exp(arr - np.max(arr, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    def _get_probabilities(self, fitted_clf, X, classes):
        """
        Produce a (n_samples, n_classes) probability matrix aligned to "classes".

        Attempts, in order:
        1. predict_proba (align and renormalize columns to "classes").
        2. decision_function (softmax over scores).
        3. Fallback to one-hot from predict.

        Returns calibrated probabilities whenever possible to ensure reliable ROC AUC.
        """
        # Try predict_proba first
        if hasattr(fitted_clf, "predict_proba"):
            try:
                probs = fitted_clf.predict_proba(X)

                # Check if fitted_clf is a Pipeline and last step has classes_ attribute
                prob_cols = getattr(fitted_clf, "classes_", None)
                if prob_cols is None and isinstance(fitted_clf, Pipeline):
                    final_estimator = fitted_clf.named_steps[list(fitted_clf.named_steps.keys())[-1]]
                    prob_cols = getattr(final_estimator, "classes_", None)

                # Create DataFrame for alignment with expected classes
                prob_df = pd.DataFrame(probs, columns=prob_cols)
                # Reindex to include all classes, missing classes get very small probability
                prob_df = prob_df.reindex(columns=classes, fill_value=1e-6)
                # Normalize so rows sum to 1
                prob_df = prob_df.div(prob_df.sum(axis=1), axis=0)

                return prob_df.values
            except Exception:
                pass

        # Fallback to decision_function if predict_proba not available
        if hasattr(fitted_clf, "decision_function"):
            try:
                df = fitted_clf.decision_function(X)
                if df.ndim == 1:
                    # Binary case
                    df = np.vstack([-df, df]).T
                probs = self._softmax(df)
                prob_df = pd.DataFrame(probs, columns=classes[:probs.shape[1]])
                prob_df = prob_df.reindex(columns=classes, fill_value=1e-6)
                # Normalize to ensure sum to 1
                prob_df = prob_df.div(prob_df.sum(axis=1), axis=0)
                return prob_df.values
            except Exception:
                pass

        # Last resort: hard predictions -> convert to one-hot
        preds = fitted_clf.predict(X)
        one_hot = np.zeros((len(preds), len(classes)))
        class_to_idx = {c: i for i, c in enumerate(classes)}
        for i, p in enumerate(preds):
            if p in class_to_idx:
                one_hot[i, class_to_idx[p]] = 1.0
        return one_hot

    def _safe_clone_and_fit(self, clf, X_train, y_train):
        """Clone (sklearn.base.clone) an estimator and fit it on X_train/y_train. Returns the fitted clone."""
        cloned = clone(clf)
        cloned.fit(X_train, y_train)
        return cloned

    def _ensure_dir(self, path):
        """Create directory if it does not exist."""
        os.makedirs(path, exist_ok=True)

    def _clean_name(self, name):
        """Return a filesystem-safe model name by replacing spaces and slashes."""
        return name.replace(" ", "_").replace("/", "_")
    
    def _unique_model_path(self, base_dir, clf_name):
        """
        Return a deterministic filepath under base_dir for saving a model.
        Uses _clean_name and the fixed extension '.pkl'. If the file exists it will
        be overwritten when opened with "wb" (pickle) or by joblib.dump.
        """
        self._ensure_dir(base_dir)
        clean_name = self._clean_name(clf_name)
        return os.path.join(base_dir, f"{clean_name}.pkl")
    
    # ----------------------------#
    #      PLOTTING HELPERS       #
    # ----------------------------# 
    def _plot_roc_per_class(self, roc_dict, classes):
        """
        Plot ROC curves for each class comparing all models in a single figure.

        Layout:
          - Two columns by default, rows computed as ceil(n_classes / 2).
          - Each subplot contains ROC curves from every classifier for a single class.
          - Missing or invalid ROC data are shown as the diagonal baseline with AUC=nan.
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

        - Uses integer annotation format where values are effectively integers.
        - Uses decimal formatting when values are normalized floats.
        - Arranges up to 3 columns of subplots.
        """
        n_classifiers = len(confusion_dict)
        if n_classifiers == 0:
            return

        n_cols = 3
        n_rows = n_classifiers // n_cols + int(n_classifiers % n_cols > 0)
        # make n_classes dynamic by inspecting first matrix if available
        first_cm = next(iter(confusion_dict.values()))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 16))
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

            sns.heatmap(cm_arr, annot=annot, fmt=fmt, cmap="Blues", cbar=False, ax=ax, square=True)
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

        Input:
          accuracies_per_model: dict mapping model_name -> list of per-fold accuracy values
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
    #   WILCOXON PAIRWISE TEST    #
    # ----------------------------#
    def _wilcoxon_pairwise(self, scores_per_model):
        """
        Perform pairwise Wilcoxon signed-rank tests between models and print results
        in a readable table.

        Purpose:
        - Compare model performance across outer folds using the non-parametric
          Wilcoxon signed-rank test.
        - Each model must have a list of fold-level F1-macro scores.
        - Produces a structured DataFrame containing the test statistic and p-value
          for every unique model pair.

        Input:
            scores_per_model : dict
                Mapping: model_name -> list of F1 macro scores (one score per outer fold)

        Output:
            DataFrame with one row per model pair:
                - Model A
                - Model B
                - Statistic  (Wilcoxon test statistic)
                - P-value    (two-sided)
        """
        # Extract list of model names in the order they appear in the dictionary
        model_names = list(scores_per_model.keys())
        results = []

        # Iterate over all unique model pairs (A,B) without repetition
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                model_a = model_names[i]
                model_b = model_names[j]

                try:
                    # Perform Wilcoxon signed-rank test on paired fold-level scores.
                    # zero_method="wilcox" excludes zero-differences from ranking.
                    # alternative="two-sided" tests for any difference in medians.
                    # mode='auto' lets SciPy choose the most suitable computation strategy.
                    stat, p = wilcoxon(
                        scores_per_model[model_a],
                        scores_per_model[model_b],
                        zero_method="wilcox",
                        alternative="two-sided",
                        mode='auto'
                    )
                except Exception:
                    # In cases where the test cannot be computed (e.g. all differences zero),
                    # store NaN values to indicate failure.
                    stat, p = np.nan, np.nan

                # Append pairwise comparison result
                results.append({
                    "Model A": model_a,
                    "Model B": model_b,
                    "Statistic": stat,
                    "P-value": p
                })

        # Convert aggregated comparison results into a DataFrame
        df_results = pd.DataFrame(results)

        # Sort by p-value for clearer interpretation (smaller p-values first)
        df_results = df_results.sort_values("P-value").reset_index(drop=True)

        return df_results


    # ----------------------------#
    #         PUBLIC API          #
    # ----------------------------#
    def fit_evaluate_store_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                                  output_dir: str = "../results/all_models",
                                  cv_splits: int = 5, cv_repeats: int = 5):
        """
        Train, evaluate, and store multiple classifiers using outer RepeatedStratifiedKFold.

        Workflow:
          1. Run an outer repeated stratified CV to produce unbiased predictions per fold.
          2. Aggregate per-fold predictions to compute overall metrics and per-class reports.
          3. Compute confusion matrices and per-class ROC curves (one-vs-rest).
          4. Refit each pipeline on the full training set (after adjusting sampling strategies)
             and save a reduced pipeline (scaler + classifier) to disk.
          5. Produce plots (ROC, confusion matrices, violin) and perform Wilcoxon pairwise tests
             on outer-fold F1-macro scores.

        Returns:
            dict with keys "results_df", "per_class_df", "wilcoxon_results_df"
        """
        # Ensure output directory exists
        self._ensure_dir(output_dir)

        # Outer CV: used to obtain unbiased performance samples for EACH model (this plays the role of the outer loop of nested CV)
        outer_cv = RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=42)

        # Determine the set of classes present in y_train
        classes = [c for c in ["CN", "EMCI", "LMCI", "AD"] if c in y_train.values]

        # Initialize storage containers
        metrics_list = []
        per_class_metrics_list = []
        confusion_dict = {}
        confusion_norm_dict = {}
        roc_dict = {}
        accuracies_per_model = {}
        f1_macro_per_model = {}  # store per-outer-fold F1 macro for Wilcoxon

        # For each classifier, run outer-fold evaluation (this yields multiple independent scores per model)
        for clf_name, clf in self.classifiers.items():
            print(f"Training & Evaluating: {clf_name}")

            # Containers for this classifier across all outer folds
            true_all = []
            pred_all = []
            prob_all_list = []
            fold_accuracies = []
            outer_fold_f1 = []

            # Outer CV loop: fit on X_tr (outer train), evaluate on X_val (outer test)
            for train_idx, val_idx in outer_cv.split(X_train, y_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                # Clone and Fit on outer-fold training portion
                fitted = self._safe_clone_and_fit(clf, X_tr, y_tr)

                # Predict and Probabilities (align probabilities with provided 'classes')
                y_pred = fitted.predict(X_val)
                prob_arr = self._get_probabilities(fitted, X_val, classes)

                # Collect
                true_all.extend(y_val)
                pred_all.extend(y_pred)
                prob_all_list.append(prob_arr)

                # Per-outer-fold accuracy and F1 (one score per outer fold)
                fold_acc = accuracy_score(y_val, y_pred)
                fold_f1 = f1_score(y_val, y_pred, average="macro")
                fold_accuracies.append(fold_acc)
                outer_fold_f1.append(fold_f1)

            # Aggregate probabilities
            prob_all = np.vstack(prob_all_list) if len(prob_all_list) > 0 else np.empty((0, len(classes)))

            # Save for violin plots and Wilcoxon
            accuracies_per_model[clf_name] = fold_accuracies
            f1_macro_per_model[clf_name] = outer_fold_f1

            # Compute global metrics using concatenated outer-fold predictions (this is a valid estimate aggregated across outer folds)
            roc_auc_macro = np.nan
            try:
                roc_auc_macro = roc_auc_score(
                    label_binarize(np.array(true_all), classes=classes), prob_all,
                    average="macro", multi_class="ovr"
                )
            except Exception:
                pass

            clf_metrics = {
                "Model": clf_name,
                "F1 Score (macro)": f1_score(np.array(true_all), np.array(pred_all), average="macro"),
                "Accuracy": accuracy_score(np.array(true_all), np.array(pred_all)),
                "Balanced Accuracy": balanced_accuracy_score(np.array(true_all), np.array(pred_all)),
                "Precision (weighted)": precision_score(np.array(true_all), np.array(pred_all), average="weighted", zero_division=0),
                "Recall (weighted)": recall_score(np.array(true_all), np.array(pred_all), average="weighted"),
                "F1 Score (weighted)": f1_score(np.array(true_all), np.array(pred_all), average="weighted"),
                "ROC AUC (macro)": roc_auc_macro
            }
            metrics_list.append(clf_metrics)

            # Per-class metrics (aggregated across outer folds)
            class_report = classification_report(np.array(true_all), np.array(pred_all), labels=classes, output_dict=True, zero_division=0)
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

            # Confusion matrices (aggregated across outer folds)
            cm = confusion_matrix(np.array(true_all), np.array(pred_all), labels=classes)
            confusion_dict[clf_name] = cm
            row_sums = cm.sum(axis=1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                cm_norm = np.divide(cm, row_sums, where=row_sums != 0)
                cm_norm = np.nan_to_num(cm_norm)
            confusion_norm_dict[clf_name] = cm_norm

            # ROC One-vs-Rest per class (aggregated across outer folds)
            y_true_bin = label_binarize(np.array(true_all), classes=classes)
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

            # Refit on full training set and save final model (same behavior as before)
            final_clf = clone(clf)

            sample_dimension = max(len(X_train) // len(y_train.unique()), 500)

            # If the pipeline contains undersampling or oversampling, update sampling_strategy to sample_dimension
            if 'rus' in final_clf.named_steps:
                rus = final_clf.named_steps['rus']
                # Replace all original counts (e.g., 385) with sample_dimension
                new_strategy = {k: sample_dimension for k in rus.sampling_strategy}
                final_clf.named_steps['rus'].sampling_strategy = new_strategy

            if 'smotenc' in final_clf.named_steps:
                sm = final_clf.named_steps['smotenc']
                # Replace all original counts (e.g., 385) with sample_dimension
                new_strategy = {k: sample_dimension for k in sm.sampling_strategy}
                final_clf.named_steps['smotenc'].sampling_strategy = new_strategy

            # Fit the modified pipeline on the full training set
            fitted_full = self._safe_clone_and_fit(final_clf, X_train, y_train)

            # Extract only the scaler (if any) and the classifier
            steps_to_save = []
            if isinstance(fitted_full, Pipeline):
                for name, step in fitted_full.named_steps.items():
                    if isinstance(step, StandardScaler) or name == 'clf':
                        steps_to_save.append((name, step))
                pipeline_to_save = Pipeline(steps_to_save)
            else:
                pipeline_to_save = fitted_full

            # Save the fitted model
            model_path = self._unique_model_path(output_dir, clf_name)
            try:
                with open(model_path, "wb") as _f:
                    pickle.dump(pipeline_to_save, _f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:
                joblib.dump(pipeline_to_save, model_path)
                
        # Assemble results DataFrames 
        results_df = pd.DataFrame(metrics_list).sort_values("F1 Score (macro)", ascending=False)
        per_class_df = pd.DataFrame(per_class_metrics_list)

        display(results_df)
        display(per_class_df)

        # Plots 
        self._plot_roc_per_class(roc_dict, classes)
        self._plot_confusion_matrices(confusion_dict, title_prefix="Confusion Matrix")
        self._plot_confusion_matrices(confusion_norm_dict, title_prefix="Normalized Confusion Matrix")
        self._plot_violin(accuracies_per_model)

        # Wilcoxon pairwise test on OUTER-FOLD F1 macro scores (unbiased samples) 
        wilcoxon_results_df = self._wilcoxon_pairwise(f1_macro_per_model)
        display(wilcoxon_results_df)

        return {
            "results_df": results_df,
            "per_class_df": per_class_df,
            "wilcoxon_results_df": wilcoxon_results_df
        }
