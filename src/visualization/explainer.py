import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from lime.lime_tabular import LimeTabularExplainer
from typing import Union, Optional, List, Sequence, Tuple, Any

class ModelExplainer:
    """Explain one or more classification models with SHAP and LIME.

    Parameters
    ----------
    models : Sequence[Tuple[str, Any]]
        Iterable of (name, fitted_model) pairs. Models must implement predict and for LIME predict_proba.
    X_train : pd.DataFrame
        Feature data used for explanations (not necessarily the training set, but representative data).
    y_train : Optional[pd.Series]
        True labels (optional, useful to pick instances).
    feature_names : Optional[List[str]]
        If None, uses X_train.columns if X_train is a DataFrame.
    class_names : Optional[List[str]]
        Names of classes for multiclass (optional).
    background_size : int
        Number of background samples to use for model-agnostic explainers when needed.
    """
    # ----------------------------#
    #        INITIALIZATION       #
    # ----------------------------# 
    def __init__(self, models: Sequence[Tuple[str, Any]], X_train: pd.DataFrame, y_train: Optional[pd.Series] = None,
                 feature_names: Optional[List[str]] = None, class_names: Optional[List[str]] = None,
                 background_size: int = 100):
        self.models = list(models)
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.feature_names = feature_names or list(X_train.columns)
        self.class_names = class_names
        self.background = self.X_train.sample(min(background_size, len(self.X_train)), random_state=0)

    # ----------------------------#
    #         SHAP HELPERS        #
    # ----------------------------# 
    def _get_shap_explainer(self, name: str, model: Any):
        """Try multiple ways to create a shap.Explainer. Returns (explainer, used_callable_name) or (None, reason)."""
        # 1) If model is directly callable, try it
        if callable(model):
            try:
                expl = shap.Explainer(model, self.background, feature_names=self.feature_names)
                return expl, "callable(model)"
            except Exception as e:
                warnings.warn(f"[{name}] shap.Explainer(callable model) failed: {e}")

        # 2) Try using predict_proba (preferibile per classificatori)
        if hasattr(model, "predict_proba") and callable(getattr(model, "predict_proba")):
            try:
                predict_proba = lambda X_train: model.predict_proba(X_train)
                expl = shap.Explainer(predict_proba, self.background, feature_names=self.feature_names)
                return expl, "model.predict_proba"
            except Exception as e:
                warnings.warn(f"[{name}] shap.Explainer(model.predict_proba) failed: {e}")

        # 3) Try using predict (per regressori o classificatori che non espongono predict_proba)
        if hasattr(model, "predict") and callable(getattr(model, "predict")):
            try:
                predict = lambda X_train: model.predict(X_train)
                expl = shap.Explainer(predict, self.background, feature_names=self.feature_names)
                return expl, "model.predict"
            except Exception as e:
                warnings.warn(f"[{name}] shap.Explainer(model.predict) failed: {e}")

        # 4) Try TreeExplainer for tree-based estimators (XGBoost/LightGBM/sklearn tree ensembles)
        try:
            expl = shap.TreeExplainer(model)
            return expl, "TreeExplainer"
        except Exception as e:
            warnings.warn(f"[{name}] TreeExplainer attempt failed: {e}")

        # 5) Give up
        return None, "no usable callable or tree explainer"

    # ----------------------------#
    #            SHAP             #
    # ----------------------------# 
    def shap_all_plots(self, sample_idx: Union[int, Sequence[int]] = 0, max_models: Optional[int] = None, figsize_per_row=(15, 4), axes=None):
        """Create SHAP summary, waterfall and force plots for each model arranged in a grid.

        Parameters
        ----------
        sample_idx : Optional[int]
            Index of the instances to use for waterfall/force plots. If None, uses 0.
        max_models : Optional[int]
            Limit number of models to plot (useful for many models).
        figsize_per_row : tuple
            Width and height per row (one row per model).
        axes : Optional[List[matplotlib.axes.Axes]]
            List of Axes objects to plot.
            If None, the function creates a new figure and its axes.
        """
        if isinstance(sample_idx, int):
            sample_idx = [sample_idx]

        models = self.models[:max_models] if max_models else self.models
        n_models = len(models)
        if n_models == 0:
            warnings.warn("No models supplied.")
            return None

        fig, axes = plt.subplots(nrows=n_models, ncols=3,
                                 figsize=(figsize_per_row[0], figsize_per_row[1] * n_models))
        # normalize axes to shape (n_models, 3)
        axes = np.array(axes)
        if axes.ndim == 1 and axes.shape[0] == 3 and n_models == 1:
            axes = axes.reshape((1, 3))
        elif axes.ndim == 2 and axes.shape[0] != n_models:
            axes = axes.reshape((n_models, 3))

        for i, (name, model) in enumerate(models):
            ax_summary, ax_water, ax_force = axes[i]

            explainer, used = self._get_shap_explainer(name, model)
            if explainer is None:
                warnings.warn(f"[{name}] Could not create SHAP explainer: {used}. Skipping SHAP plots.")
                for a in (ax_summary, ax_water, ax_force):
                    a.text(0.5, 0.5, f"SHAP not available\n({name})", ha="center", va="center")
                continue

            try:
                shap_vals = explainer(self.X_train)
            except Exception as e:
                warnings.warn(f"[{name}] Explainer(...) on X_train failed: {e}. Skipping this model.")
                for a in (ax_summary, ax_water, ax_force):
                    a.text(0.5, 0.5, f"SHAP failed\n({name})", ha="center", va="center")
                continue

            # summary
            plt.sca(ax_summary)
            try:
                shap.summary_plot(shap_vals, features=self.X_train, feature_names=self.feature_names, show=False)
                ax_summary.set_title(f"{name} — SHAP summary")
            except Exception as e:
                warnings.warn(f"[{name}] summary_plot failed: {e}")
                ax_summary.text(0.5, 0.5, "summary_plot failed", ha="center", va="center")

            # validate sample_idx
            if sample_idx < 0 or sample_idx >= len(self.X_train):
                warnings.warn(f"sample_idx {sample_idx} out of range for X_train (len={len(self.X_train)}). Skipping instance plots.")
                ax_water.text(0.5, 0.5, "invalid sample_idx", ha="center", va="center")
                ax_force.text(0.5, 0.5, "invalid sample_idx", ha="center", va="center")
                continue

            x_instance = self.X_train.iloc[[sample_idx]]
            try:
                shap_single = explainer(x_instance)
            except Exception as e:
                warnings.warn(f"[{name}] explainer(x_instance) failed: {e}")
                ax_water.text(0.5, 0.5, "waterfall failed", ha="center", va="center")
                ax_force.text(0.5, 0.5, "force failed", ha="center", va="center")
                continue

            # waterfall
            plt.sca(ax_water)
            try:
                # shap_single may be multi-output; use index 0 if necessary
                entry = shap_single[0] if len(shap_single) > 0 else shap_single
                shap.plots.waterfall(entry, show=False)
                ax_water.set_title(f"{name} — Waterfall (idx={sample_idx})")
            except Exception as e:
                warnings.warn(f"[{name}] waterfall plot failed: {e}")
                ax_water.text(0.5, 0.5, "waterfall failed", ha="center", va="center")

            # force
            plt.sca(ax_force)
            try:
                entry = shap_single[0] if len(shap_single) > 0 else shap_single
                shap.plots.force(entry, matplotlib=True, show=False)
                ax_force.set_title(f"{name} — Force (idx={sample_idx})")
            except Exception as e:
                warnings.warn(f"[{name}] force plot failed: {e}")
                ax_force.text(0.5, 0.5, "force failed", ha="center", va="center")

        plt.tight_layout()
        plt.show()
    

    def shap_summary_plots(self, max_models: Optional[int] = None, figsize_per_row=(8, 5), axes=None):
        """Create SHAP summary plots for each model arranged in a grid.
        
        Parameters
        ----------
        sample_idx : Optional[int]
            Index of the instance to use for waterfall/force plots. If None, uses 0.
        max_models : Optional[int]
            Limit number of models to plot (useful for many models).
        figsize_per_row : tuple
            Width and height per row (one row per model).
        axes : Optional[List[matplotlib.axes.Axes]]
            List of Axes objects to plot.
            If None, the function creates a new figure and its axes.
        """
        models = self.models[:max_models] if max_models else self.models
        n_models = len(models)
        if n_models == 0:
            warnings.warn("No models supplied.")
            return None

        if axes is None:  # creo la figura solo se non ricevo assi
            fig, axes = plt.subplots(nrows=1, ncols=n_models,
                                    figsize=(figsize_per_row[0] * n_models, figsize_per_row[1]))
            if n_models == 1:
                axes = [axes]
        else:
            fig = None
            if n_models == 1:
                axes = [axes]

        for i, (name, model) in enumerate(models):
            ax = axes[i]
            explainer, used = self._get_shap_explainer(name, model)
            if explainer is None:
                warnings.warn(f"[{name}] Could not create SHAP explainer: {used}. Skipping.")
                ax.text(0.5, 0.5, f"SHAP not available\n({name})", ha="center", va="center")
                continue

            try:
                shap_vals = explainer(self.X_train)
                plt.sca(ax)
                shap.summary_plot(shap_vals, features=self.X_train, feature_names=self.feature_names, show=False)
                ax.set_title(f"{name} — SHAP summary")
            except Exception as e:
                warnings.warn(f"[{name}] summary plot generation failed: {e}")
                ax.text(0.5, 0.5, "summary_plot failed", ha="center", va="center")

        if fig is not None:
            plt.tight_layout()
            plt.show()


    def shap_waterfall_plots(self, sample_idx: Union[int, Sequence[int]] = 0, max_models: Optional[int] = None, figsize_per_row=(8, 5), axes=None):
        """Create SHAP waterfall plots for a specific sample for each model.
        
        Parameters
        ----------
        sample_idx : Union[int, Sequence[int]]
            Index of the instances to use for waterfall/force plots. If None, uses 0.
        max_models : Optional[int]
            Limit number of models to plot (useful for many models).
        figsize_per_row : tuple
            Width and height per row (one row per model).
        axes : Optional[List[matplotlib.axes.Axes]]
            List of Axes objects to plot.
            If None, the function creates a new figure and its axes.
        """
        if isinstance(sample_idx, int):
            sample_idx = [sample_idx]

        models = self.models[:max_models] if max_models else self.models
        n_models = len(models)
        n_samples = len(sample_idx)

        if n_models == 0:
            warnings.warn("No models supplied.")
            return None

        for idx in sample_idx:
            if idx < 0 or idx >= len(self.X_train):
                warnings.warn(f"sample_idx {idx} out of range for X_train (len={len(self.X_train)}). Skipping.")
                continue

        fig, axes = plt.subplots(nrows=n_samples, ncols=n_models,
                                figsize=(figsize_per_row[0] * n_models, figsize_per_row[1] * n_samples))
        axes = np.array(axes)
        if axes.ndim == 1:
            axes = axes.reshape((n_samples, n_models))

        for i, idx in enumerate(sample_idx):
            x_instance = self.X_train.iloc[[idx]]
            for j, (name, model) in enumerate(models):
                ax = axes[i, j]
                explainer, used = self._get_shap_explainer(name, model)
                if explainer is None:
                    warnings.warn(f"[{name}] Could not create SHAP explainer: {used}. Skipping.")
                    ax.text(0.5, 0.5, f"SHAP not available\n({name})", ha="center", va="center")
                    continue

                try:
                    shap_single = explainer(x_instance)
                    entry = shap_single[0] if len(shap_single) > 0 else shap_single
                    plt.sca(ax)
                    shap.plots.waterfall(entry, show=False)
                    ax.set_title(f"{name} — Waterfall (idx={idx})")
                except Exception as e:
                    warnings.warn(f"[{name}] waterfall generation failed: {e}")
                    ax.text(0.5, 0.5, "waterfall failed", ha="center", va="center")

        plt.tight_layout()
        plt.show()
    

    def shap_force_plots(self, sample_idx: Union[int, Sequence[int]] = 0, max_models: Optional[int] = None, figsize_per_row=(8, 5), axes=None):
        """Create SHAP force plots for a specific sample for each model.
        
        Parameters
        ----------
        sample_idx : Union[int, Sequence[int]]
            Index of the instances to use for waterfall/force plots. If None, uses 0.
        max_models : Optional[int]
            Limit number of models to plot (useful for many models).
        figsize_per_row : tuple
            Width and height per row (one row per model).
        axes : Optional[List[matplotlib.axes.Axes]]
            List of Axes objects to plot.
            If None, the function creates a new figure and its axes.
        """
        if isinstance(sample_idx, int):
            sample_idx = [sample_idx]

        models = self.models[:max_models] if max_models else self.models
        n_models = len(models)
        n_samples = len(sample_idx)

        if n_models == 0:
            warnings.warn("No models supplied.")
            return None

        fig, axes = plt.subplots(nrows=n_samples, ncols=n_models,
                                figsize=(figsize_per_row[0] * n_models, figsize_per_row[1] * n_samples))
        axes = np.array(axes)
        if axes.ndim == 1:
            axes = axes.reshape((n_samples, n_models))

        for i, idx in enumerate(sample_idx):
            if idx < 0 or idx >= len(self.X_train):
                warnings.warn(f"sample_idx {idx} out of range for X_train (len={len(self.X_train)}). Skipping.")
                continue
            x_instance = self.X_train.iloc[[idx]]
            for j, (name, model) in enumerate(models):
                ax = axes[i, j]
                explainer, used = self._get_shap_explainer(name, model)
                if explainer is None:
                    warnings.warn(f"[{name}] Could not create SHAP explainer: {used}. Skipping.")
                    ax.text(0.5, 0.5, f"SHAP not available\n({name})", ha="center", va="center")
                    continue

                try:
                    shap_single = explainer(x_instance)
                    entry = shap_single[0] if len(shap_single) > 0 else shap_single
                    plt.sca(ax)
                    shap.plots.force(entry, matplotlib=True, show=False)
                    ax.set_title(f"{name} — Force (idx={idx})")
                except Exception as e:
                    warnings.warn(f"[{name}] force generation failed: {e}")
                    ax.text(0.5, 0.5, "force failed", ha="center", va="center")

        plt.tight_layout()
        plt.show()


    # ----------------------------#
    #             LIME            #
    # ----------------------------# 
    def lime_explain(self, sample_idx: Union[int, Sequence[int]] = 0, num_features: int = 35, num_samples: int = 5000,
                 discretize_continuous: bool = True, max_models: Optional[int] = None, figsize_per_row=(6, 4)):
        """Create LIME explanations (explain_instance) for each model and lay them out side-by-side.

        Parameters
        ----------
        sample_idx : Union[int, Sequence[int]]
            index of the istances to explain (0-based).
        num_features : int
            number of features to show in explanation.
        num_samples : int
            number of synthetic samples LIME will generate.
        discretize_continuous : bool
            whether to discretize continuous features.
        max_models : Optional[int]
            limit number of models to process.
        """
        if isinstance(sample_idx, int):
            sample_idx = [sample_idx]

        models = self.models[:max_models] if max_models else self.models
        n_models = len(models)
        n_samples = len(sample_idx)

        fig, axes = plt.subplots(nrows=n_samples, ncols=n_models,
                                figsize=(figsize_per_row[0] * n_models, figsize_per_row[1] * n_samples))
        axes = np.array(axes)
        if axes.ndim == 1:
            axes = axes.reshape((n_samples, n_models))

        explainer = LimeTabularExplainer(self.X_train.values,
                                        feature_names=self.feature_names,
                                        class_names=self.class_names,
                                        discretize_continuous=discretize_continuous)

        for i, idx in enumerate(sample_idx):
            x_row = self.X_train.iloc[idx]
            for j, (name, model) in enumerate(models):
                ax = axes[i, j]
                predict_fn = model.predict_proba
                exp = explainer.explain_instance(x_row.values, predict_fn,
                                                num_features=num_features, num_samples=num_samples)
                items = exp.as_list()
                feats, weights = zip(*items)

                y_pos = np.arange(len(feats))
                ax.barh(y_pos, weights)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(feats)
                ax.invert_yaxis()
                ax.set_title(f"{name} — LIME (idx={idx})")

        plt.tight_layout()
        plt.show()
