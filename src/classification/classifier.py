import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    balanced_accuracy_score
)
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from imodels import OptimalTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


class ADNIClassifier:
    """
    ADNIClassifier: helper class to train/evaluate multiple classifiers on an ADNIMERG dataset.
    """
    # ----------------------------#
    #        INITIALIZATION       #
    # ----------------------------# 
    def __init__(self, classifier: str = "Standard1"):
        """
        Initialize the class and load a predefined set of classifiers 
        based on the provided selection string.

        Parameters
        ----------
        classifier : str, optional (default="Standard")
            Defines which group of classifiers to initialize:
            
            - "Standard1" : Load tuned classifiers set 1
            - "XAI1"      : Load explainable (XAI) classifiers set 1
            - "Standard2" : Load tuned classifiers set 2
            - "XAI2"      : Load explainable (XAI) classifiers set 2
            - Any other value defaults to "Standard1"

        Notes
        -----
        This constructor sets `self.classifiers` to a dictionary containing 
        sklearn-compatible models (or pipelines), preconfigured with the 
        best hyperparameters found via Grid Search.
        """
        if classifier == "Standard1" or classifier == "standard1" or classifier == "STANDARD1" or classifier == "None":
            self.classifiers = self._default_classifiers_1()
        elif classifier == "XAI1" or classifier == "xai1" or classifier == "Xai1":
            self.classifiers = self._xai_classifiers_1()
        elif classifier == "Standard2" or classifier == "standard2" or classifier == "STANDARD2":
            self.classifiers = self._default_classifiers_1()
        elif classifier == "XAI2" or classifier == "xai2" or classifier == "Xai2":
            self.classifiers = self._xai_classifiers_1()
        else:
            self.classifiers = self._default_classifiers_1()

    # ----------------------------#
    #   CLASSIFIERS DEFINITION    #
    # ----------------------------# 
    def _default_classifiers_1(self):
        return {
            'Random Forest': RandomForestClassifier(
                random_state=42, class_weight='balanced', n_jobs=-1,
                criterion='entropy', max_depth=None, max_features=1.0,
                min_samples_leaf=2, n_estimators=100
            ),
            'Extra Trees': ExtraTreesClassifier(
                random_state=42, class_weight='balanced', n_jobs=-1,
                criterion='entropy', max_depth=None, max_features=1.0,
                min_samples_leaf=2, n_estimators=75
            ),
            'XGBoost': XGBClassifier(
                random_state=42, use_label_encoder=False, eval_metric='mlogloss', verbosity=0,
                colsample_bytree=0.7, gamma=1.0, learning_rate=0.1,
                max_depth=6, n_estimators=50, reg_alpha=1, reg_lambda=0,
                subsample=1.0
            ),
            'LightGBM': LGBMClassifier(
                random_state=42, verbose=-1,
                colsample_bytree=1.0, learning_rate=0.01, max_depth=8,
                min_child_samples=20, n_estimators=100, num_leaves=15,
                reg_alpha=0, reg_lambda=1, subsample=0.8
            ),
            'CatBoost': CatBoostClassifier(
                random_state=42, verbose=False, loss_function='MultiClass',
                bagging_temperature=0.0, border_count=64, depth=8,
                iterations=75, l2_leaf_reg=3, learning_rate=0.1,
                random_strength=0.5
            ),
            'Multinomial Logistic Regression': Pipeline([
                ('scaler', StandardScaler()),
                ('logreg', LogisticRegression(
                    random_state=42, solver='saga', max_iter=2000, class_weight='balanced',
                    C=1.0, penalty='l1'
                ))
            ]),
            'KNN': Pipeline([
                ('scaler', StandardScaler()),
                ('knn', KNeighborsClassifier(n_neighbors=10, p=1, weights='distance', n_jobs=-1))
            ]),
            'Bagging': BaggingClassifier(random_state=42, n_jobs=-1, bootstrap=False, max_features=1.0, max_samples=0.6, n_estimators=100)
        }


    def _xai_classifiers_1(self):
        return {
            'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=None, max_features=0.8, min_samples_leaf=10, min_samples_split=2),
            'OptimalTree': OptimalTreeClassifier(random_state=42, balance=True, feature_exchange=True, look_ahead=True, regularization=0.0)
        }

    
    def _default_classifiers_2(self): 
        """
        Return a dictionary with tuned classifier instances.
        """
        return {
            'Random Forest': RandomForestClassifier(
                random_state=42, class_weight='balanced', n_jobs=-1,
                criterion='entropy', max_depth=6, max_features=0.5,
                min_samples_leaf=2, n_estimators=50
            ),
            'Extra Trees': ExtraTreesClassifier(
                random_state=42, class_weight='balanced', n_jobs=-1,
                criterion='entropy', max_depth=None, max_features=1.0,
                min_samples_leaf=8, n_estimators=50
            ),
            'XGBoost': XGBClassifier(
                random_state=42, use_label_encoder=False, eval_metric='mlogloss', verbosity=0,
                colsample_bytree=1.0, gamma=1.0, learning_rate=0.1,
                max_depth=8, n_estimators=100, reg_alpha=0, reg_lambda=1,
                subsample=0.8
            ),
            'LightGBM': LGBMClassifier(
                random_state=42, verbose=-1,
                colsample_bytree=0.7, learning_rate=0.1, max_depth=8,
                min_child_samples=5, n_estimators=100, num_leaves=15,
                reg_alpha=1, reg_lambda=1, subsample=0.8
            ),
            'CatBoost': CatBoostClassifier(
                random_state=42, verbose=False, loss_function='MultiClass',
                bagging_temperature=0.0, border_count=32, depth=6,
                iterations=100, l2_leaf_reg=1, learning_rate=0.1,
                random_strength=0.5
            ),
            'Multinomial Logistic Regression': Pipeline([
                ('scaler', StandardScaler()),
                ('logreg', LogisticRegression(
                    random_state=42, solver='saga', max_iter=2000, class_weight='balanced',
                    C=1.0, penalty='l1'
                ))
            ]),
            'KNN': Pipeline([
                ('scaler', StandardScaler()),
                ('knn', KNeighborsClassifier(n_neighbors=10, p=1, weights='distance', n_jobs=-1))
            ]),
            'Bagging': BaggingClassifier(random_state=42, n_jobs=-1, bootstrap=False, max_features=0.8, max_samples=0.6, n_estimators=100)
        }


    def _xai_classifiers_2(self): 
        return {
            'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=4, max_features=1.0, min_samples_leaf=1, min_samples_split=2),
            'OptimalTree': OptimalTreeClassifier(random_state=42, balance=True, feature_exchange=True, look_ahead=True, regularization=0.0)
        }

    # ----------------------------#
    #          UTILITY            #
    # ----------------------------# 
    def _softmax(self, arr):
        """
        Numerically stable softmax across last axis.
        """
        e = np.exp(arr - np.max(arr, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)


    def _get_probabilities(self, fitted_clf, X, classes):
        """
        Return probability matrix aligned to 'classes' for a fitted classifier.
        If predict_proba is unavailable, try decision_function and apply softmax.
        If neither exists, fall back to one-hot encoded predictions (less informative for ROC).
        """
        # Primary: predict_proba
        if hasattr(fitted_clf, "predict_proba"):
            try:
                probs = fitted_clf.predict_proba(X)
                prob_cols = getattr(fitted_clf, "classes_", None)
                # If classes_ not present at pipeline level, try final estimator
                if prob_cols is None and isinstance(fitted_clf, Pipeline):
                    final = fitted_clf.named_steps[list(fitted_clf.named_steps.keys())[-1]]
                    prob_cols = getattr(final, "classes_", None)
                prob_df = pd.DataFrame(probs, columns=prob_cols)
                prob_df = prob_df.reindex(columns=classes, fill_value=0)
                return prob_df.values
            except Exception:
                pass

        # Secondary: decision_function -> softmax
        if hasattr(fitted_clf, "decision_function"):
            try:
                df = fitted_clf.decision_function(X)
                # if binary, decision_function may return shape (n_samples,), convert to 2-col
                if df.ndim == 1:
                    df = np.vstack([-df, df]).T
                probs = self._softmax(df)
                prob_df = pd.DataFrame(probs, columns=classes[:probs.shape[1]])
                prob_df = prob_df.reindex(columns=classes, fill_value=0)
                return prob_df.values
            except Exception:
                pass

        # Fallback: one-hot from hard predictions
        preds = fitted_clf.predict(X)
        one_hot = np.zeros((len(preds), len(classes)))
        class_to_idx = {c: i for i, c in enumerate(classes)}
        for i, p in enumerate(preds):
            if p in class_to_idx:
                one_hot[i, class_to_idx[p]] = 1.0
        return one_hot


    def _safe_clone_and_fit(self, clf, X_train, y_train):
        """
        Clone a classifier and fit it on the provided data. Return the fitted clone.
        This avoids side-effects on the original estimator object.
        """
        cloned = clone(clf)
        cloned.fit(X_train, y_train)
        return cloned

    def _ensure_dir(self, path):
        """
        Ensure output directory exists.
        """
        os.makedirs(path, exist_ok=True)

    def _clean_name(self, name):
        """
        Sanitize a classifier name for filenames.
        """
        return name.replace(' ', '_').replace('/', '_')

    # ----------------------------#
    #     CORE EVALUATION (CV)    #
    # ----------------------------# 
    def _run_repeated_cv(self, clf, X, y, cv_splitter):
        """
        Run repeated stratified CV for a single classifier. Return:
        true_all (np.array), pred_all (np.array), prob_all (np.ndarray), fold_accuracies (list)
        """
        true_all = []
        pred_all = []
        prob_all_list = []
        fold_accuracies = []
        classes = np.unique(y)

        for train_idx, val_idx in cv_splitter.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # clone + fit on fold
            fitted = self._safe_clone_and_fit(clf, X_tr, y_tr)

            # Predict and probabilities
            y_pred = fitted.predict(X_val)
            prob_arr = self._get_probabilities(fitted, X_val, classes)

            true_all.extend(y_val)
            pred_all.extend(y_pred)
            prob_all_list.append(prob_arr)

            fold_acc = accuracy_score(y_val, y_pred)
            fold_accuracies.append(fold_acc)

        # stack probabilities and arrays
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
        import math

        n_classes = len(classes)
        if n_classes == 0:
            return

        ncols = 2
        nrows = math.ceil(n_classes / ncols)
        figsize = (12, 5 * nrows)  # width x height, adjust if you want bigger/smaller

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        # ensure axes is a flattened array for simple indexing
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]

        for idx, cls in enumerate(classes):
            ax = axes[idx]
            # plot each classifier's ROC for this class
            for clf_name, (fpr_dict, tpr_dict, auc_dict) in roc_dict.items():
                fpr = fpr_dict.get(cls, None)
                tpr = tpr_dict.get(cls, None)
                auc_val = auc_dict.get(cls, np.nan)

                # if fpr/tpr missing or length mismatch, fallback to diagonal
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
            ax.plot([0, 1], [0, 1], 'k--', lw=1)
            ax.set_xlim([-0.01, 1.01])
            ax.set_ylim([-0.01, 1.01])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve - Class {cls} (One-vs-Rest)')
            ax.legend(loc='lower right', fontsize='small')
            ax.grid(alpha=0.2)

        # remove unused axes if any
        total_plots = nrows * ncols
        for j in range(len(classes), total_plots):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()


    def _plot_confusion_matrices(self, confusion_dict, title_prefix='Confusion Matrix'):
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
                fmt = 'd'
                annot = cm_arr.astype(int)  # annotate integers for neatness
            else:
                fmt = '.2f'
                annot = cm_arr

            sns.heatmap(cm_arr, annot=annot, fmt=fmt, cmap='Blues', cbar=False, ax=ax)
            ax.set_title(f'{clf_name} {title_prefix}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')

        # remove unused axes
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
                violin_rows.append({'Model': model_name, 'Accuracy': acc})
        if not violin_rows:
            return
        violin_df = pd.DataFrame(violin_rows)
        plt.figure(figsize=(12, 8))
        sns.violinplot(x='Model', y='Accuracy', data=violin_df, inner='quartile')
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy Comparison (per-fold distributions)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # ----------------------------#
    #        PUBLIC METHODS       #
    # ----------------------------# 
    def fit_evaluate_store_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                                  output_dir: str = '../results/all_models',
                                  cv_splits: int = 5, cv_repeats: int = 3):
        """
        Train and evaluate classifiers using repeated stratified CV, save fitted models on full training set,
        and produce summary metrics and plots.

        Returns a dictionary with:
            - results_df: DataFrame of global metrics
            - per_class_df: DataFrame of per-class metrics
            - saved_model_paths: dict of saved model file paths
        """
        # ensure output directory exists
        self._ensure_dir(output_dir)

        # CV splitter
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

            # repeated CV
            true_all, pred_all, prob_all, fold_accuracies = self._run_repeated_cv(clf, X_train, y_train, cv_splitter)

            # store fold accuracies
            accuracies_per_model[clf_name] = fold_accuracies

            # compute global metrics
            roc_auc_macro = np.nan
            try:
                roc_auc_macro = roc_auc_score(label_binarize(true_all, classes=classes), prob_all,
                                              average='macro', multi_class='ovr')
            except Exception:
                roc_auc_macro = np.nan

            clf_metrics = {
                'Model': clf_name,
                'Accuracy': accuracy_score(true_all, pred_all),
                'Balanced Accuracy': balanced_accuracy_score(true_all, pred_all),
                'Precision (weighted)': precision_score(true_all, pred_all, average='weighted', zero_division=0),
                'Recall (weighted)': recall_score(true_all, pred_all, average='weighted'),
                'F1 Score (weighted)': f1_score(true_all, pred_all, average='weighted'),
                'F1 Score (macro)': f1_score(true_all, pred_all, average='macro'),
                'ROC AUC (macro)': roc_auc_macro
            }
            metrics_list.append(clf_metrics)

            # per-class metrics using classification_report
            class_report = classification_report(true_all, pred_all, labels=classes, output_dict=True, zero_division=0)
            for cls in classes:
                rep = class_report.get(str(cls), {})
                per_class_metrics_list.append({
                    'Model': clf_name,
                    'Class': cls,
                    'Precision': rep.get('precision', 0.0),
                    'Recall': rep.get('recall', 0.0),
                    'F1 Score': rep.get('f1-score', 0.0),
                    'Support': rep.get('support', 0)
                })

            # confusion matrix counts and normalized
            cm = confusion_matrix(true_all, pred_all, labels=classes)
            confusion_dict[clf_name] = cm
            row_sums = cm.sum(axis=1, keepdims=True)
            with np.errstate(divide='ignore', invalid='ignore'):
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
            model_path = os.path.join(output_dir, f"{self._clean_name(clf_name)}.joblib")
            joblib.dump(fitted_full, model_path, compress=3)
            saved_model_paths[clf_name] = model_path

        # assemble result DataFrames
        results_df = pd.DataFrame(metrics_list).sort_values('ROC AUC (macro)', ascending=False)
        per_class_df = pd.DataFrame(per_class_metrics_list)

        # CHANGE WITH "PRINT" IF YOU WANT TO USE OUTSIDE OF IPYTHON
        display(results_df)
        display(per_class_df)

        self._plot_roc_per_class(roc_dict, classes)
        self._plot_confusion_matrices(confusion_dict, title_prefix='Confusion Matrix')
        self._plot_confusion_matrices(confusion_norm_dict, title_prefix='Normalized Confusion Matrix')
        self._plot_violin(accuracies_per_model)

        return {
            'results_df': results_df,
            'per_class_df': per_class_df
        }
    

    def fit_evaluate_models_training(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fit each classifier on the full training set, evaluate on the same training set (useful
        for diagnostic / overfitting inspection), and produce metrics + plots.

        Returns:
            - results_df: DataFrame of global metrics
            - per_class_df: DataFrame of per-class metrics
        """
        classes = np.unique(y_train)
        metrics_list = []
        per_class_metrics_list = []
        confusion_dict = {}
        confusion_norm_dict = {}
        roc_dict = {}

        for clf_name, clf in self.classifiers.items():
            print(f"Training & Evaluating on training set: {clf_name}")

            # fit on full training
            fitted = self._safe_clone_and_fit(clf, X_train, y_train)

            # predictions and probabilities
            y_pred = fitted.predict(X_train)
            prob_all = self._get_probabilities(fitted, X_train, classes)

            # global metrics
            roc_auc_macro = np.nan
            try:
                roc_auc_macro = roc_auc_score(label_binarize(y_train, classes=classes), prob_all,
                                              average='macro', multi_class='ovr')
            except Exception:
                roc_auc_macro = np.nan

            clf_metrics = {
                'Model': clf_name,
                'Accuracy': accuracy_score(y_train, y_pred),
                'Precision (weighted)': precision_score(y_train, y_pred, average='weighted', zero_division=0),
                'Recall (weighted)': recall_score(y_train, y_pred, average='weighted'),
                'F1 Score (weighted)': f1_score(y_train, y_pred, average='weighted'),
                'ROC AUC (macro)': roc_auc_macro
            }
            metrics_list.append(clf_metrics)

            # per-class metrics
            class_report = classification_report(y_train, y_pred, labels=classes, output_dict=True, zero_division=0)
            for cls in classes:
                rep = class_report.get(str(cls), {})
                per_class_metrics_list.append({
                    'Model': clf_name,
                    'Class': cls,
                    'Precision': rep.get('precision', 0.0),
                    'Recall': rep.get('recall', 0.0),
                    'F1 Score': rep.get('f1-score', 0.0),
                    'Support': rep.get('support', 0)
                })

            # confusion matrices
            cm = confusion_matrix(y_train, y_pred, labels=classes)
            confusion_dict[clf_name] = cm
            row_sums = cm.sum(axis=1, keepdims=True)
            with np.errstate(divide='ignore', invalid='ignore'):
                cm_norm = np.divide(cm, row_sums, where=row_sums != 0)
                cm_norm = np.nan_to_num(cm_norm)
            confusion_norm_dict[clf_name] = cm_norm

            # ROC per class
            y_true_bin = label_binarize(y_train, classes=classes)
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

        results_df = pd.DataFrame(metrics_list).sort_values('ROC AUC (macro)', ascending=False)
        per_class_df = pd.DataFrame(per_class_metrics_list)

        # CHANGE WITH "PRINT" IF YOU WANT TO USE OUTSIDE OF IPYTHON
        display(results_df)
        display(per_class_df)

        self._plot_roc_per_class(roc_dict, classes)
        self._plot_confusion_matrices(confusion_dict, title_prefix='Confusion Matrix')
        self._plot_confusion_matrices(confusion_norm_dict, title_prefix='Normalized Confusion Matrix')

        return {
            'results_df': results_df,
            'per_class_df': per_class_df
        }
