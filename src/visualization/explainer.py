import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
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
             background_size: int = 100, random_state: Optional[Union[int, np.random.Generator]] = None):
        # Store models as list of (name, model) pairs
        self.models = list(models)
        # Copy of training data (used for explanations)
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        # Feature names for visualization
        self.feature_names = feature_names or list(X_train.columns)
        # Optional class names (useful for multiclass problems)
        self.class_names = class_names
        # Create background with controlled seed
        if isinstance(random_state, np.random.Generator):
            rng = random_state
        else:
            rng = np.random.default_rng(random_state)
        # Background sample for SHAP explanations
        self.background = self.X_train.sample(min(background_size, len(self.X_train)), random_state=rng)    

    # ----------------------------#
    #         SHAP HELPER         #
    # ----------------------------#
    def _get_shap_explainer(self, name: str, model: Any):
        """Try multiple ways to create a shap.Explainer. 
        Returns (explainer, used_method) or (None, reason)."""
        # 1) If model is callable directly
        if callable(model):
            try:
                expl = shap.Explainer(model, self.background, feature_names=self.feature_names)
                return expl, "callable(model)"
            except Exception as e:
                warnings.warn(f"[{name}] shap.Explainer(callable model) failed: {e}")

        # 2) Try using predict_proba (preferred for classifiers)
        if hasattr(model, "predict_proba") and callable(getattr(model, "predict_proba")):
            try:
                predict_proba = lambda X_train: model.predict_proba(X_train)
                expl = shap.Explainer(predict_proba, self.background, feature_names=self.feature_names)
                return expl, "model.predict_proba"
            except Exception as e:
                warnings.warn(f"[{name}] shap.Explainer(model.predict_proba) failed: {e}")

        # 3) Try using predict (useful for regressors or classifiers without predict_proba)
        if hasattr(model, "predict") and callable(getattr(model, "predict")):
            try:
                predict = lambda X_train: model.predict(X_train)
                expl = shap.Explainer(predict, self.background, feature_names=self.feature_names)
                return expl, "model.predict"
            except Exception as e:
                warnings.warn(f"[{name}] shap.Explainer(model.predict) failed: {e}")

        # 4) Try TreeExplainer for tree-based models (XGBoost, LightGBM, sklearn ensembles)
        try:
            expl = shap.TreeExplainer(model)
            return expl, "TreeExplainer"
        except Exception as e:
            warnings.warn(f"[{name}] TreeExplainer attempt failed: {e}")

        # 5) If all attempts fail
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
        if isinstance(sample_idx, (list, tuple, np.ndarray)):
            if len(sample_idx) == 0:
                raise ValueError("sample_idx list empty")
            sample_idx0 = int(sample_idx[0])
        else:
            sample_idx0 = int(sample_idx)

        models = self.models[:max_models] if max_models else self.models
        n_models = len(models)
        if n_models == 0:
            warnings.warn("No models supplied.")
            return None

        # create grid reliably: n_models rows, 3 cols
        fig, axes = plt.subplots(nrows=n_models, ncols=3,
                                figsize=(figsize_per_row[0] * 3 if n_models == 1 else figsize_per_row[0] * 3,
                                        figsize_per_row[1] * n_models))
        # normalize to 2D array shape (n_models, 3)
        axes = np.array(axes)
        if axes.ndim == 1:
            axes = axes.reshape((1, -1))  # (1,3)
        if axes.shape != (n_models, 3):
            axes = axes.reshape((n_models, 3))

        for i, (name, model) in enumerate(models):
            ax_summary, ax_water, ax_force = axes[i, 0], axes[i, 1], axes[i, 2]

            explainer, used = self._get_shap_explainer(name, model)
            if explainer is None:
                warnings.warn(f"[{name}] Could not create SHAP explainer: {used}. Skipping SHAP plots.")
                for a in (ax_summary, ax_water, ax_force):
                    a.text(0.5, 0.5, f"SHAP not available\n({name})", ha="center", va="center")
                continue

            # compute shap values
            try:
                shap_vals = explainer(self.X_train)
            except Exception as e:
                warnings.warn(f"[{name}] Explainer(...) on X_train failed: {e}. Skipping this model.")
                for a in (ax_summary, ax_water, ax_force):
                    a.text(0.5, 0.5, f"SHAP failed\n({name})", ha="center", va="center")
                continue

            # --- SUMMARY (beeswarm / summary) ---
            try:
                plt.sca(ax_summary)   # set current axis
                # try to use the SHAP matplotlib backend (many versions respect current axis with matplotlib=True)
                # prefer shap.plots.beeswarm or summary_plot with show=False; try both defensively
                try:
                    # shap.plots.beeswarm accepts an Explanation or array; matplotlib backend usually honors current ax
                    shap.plots.beeswarm(shap_vals, show=False)
                except Exception:
                    # fallback to summary_plot (older API)
                    shap.summary_plot(shap_vals, features=self.X_train, feature_names=self.feature_names, show=False)
                ax_summary.set_title(f"{name} — SHAP summary")
            except Exception as e:
                warnings.warn(f"[{name}] summary_plot failed: {e}")
                ax_summary.text(0.5, 0.5, "summary_plot failed", ha="center", va="center")

            # validate sample index
            if sample_idx0 < 0 or sample_idx0 >= len(self.X_train):
                warnings.warn(f"sample_idx {sample_idx0} out of range (len={len(self.X_train)}). Skipping instance plots.")
                ax_water.text(0.5, 0.5, "invalid sample_idx", ha="center", va="center")
                ax_force.text(0.5, 0.5, "invalid sample_idx", ha="center", va="center")
                continue

            x_instance = self.X_train.iloc[[sample_idx0]]
            try:
                shap_single = explainer(x_instance)
            except Exception as e:
                warnings.warn(f"[{name}] explainer(x_instance) failed: {e}")
                ax_water.text(0.5, 0.5, "waterfall failed", ha="center", va="center")
                ax_force.text(0.5, 0.5, "force failed", ha="center", va="center")
                continue

            # WATERFALL
            try:
                plt.sca(ax_water)
                entry = shap_single[0] if len(shap_single) > 0 else shap_single
                # prefer matplotlib backend when available
                shap.plots.waterfall(entry, show=False)
                ax_water.set_title(f"{name} — Waterfall (idx={sample_idx0})")
            except Exception as e:
                warnings.warn(f"[{name}] waterfall plot failed: {e}")
                ax_water.text(0.5, 0.5, "waterfall failed", ha="center", va="center")

            # FORCE
            try:
                plt.sca(ax_force)
                entry = shap_single[0] if len(shap_single) > 0 else shap_single
                shap.plots.force(entry, matplotlib=True, show=False)
                ax_force.set_title(f"{name} — Force (idx={sample_idx0})")
            except Exception as e:
                warnings.warn(f"[{name}] force plot failed: {e}")
                ax_force.text(0.5, 0.5, "force failed", ha="center", va="center")

        plt.tight_layout()
        plt.show()


    def shap_summary_plots(self, max_models: Optional[int] = None, figsize=(8, 5), axes=None):
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
        if not models:
            warnings.warn("No models supplied.")
            return

        for name, model in models:
            explainer, used = self._get_shap_explainer(name, model)
            if explainer is None:
                warnings.warn(f"[{name}] Could not create SHAP explainer: {used}. Skipping.")
                continue

            try:
                shap_vals = explainer(self.X_train)
            except Exception as e:
                warnings.warn(f"[{name}] Explainer(...) failed: {e}. Skipping.")
                continue

            # crea una nuova figura per ogni modello
            plt.figure(figsize=figsize)
            try:
                # prova a usare beeswarm (matplotlib backend)
                try:
                    shap.plots.beeswarm(shap_vals, show=False)
                except Exception:
                    # fallback a summary_plot
                    shap.summary_plot(shap_vals, features=self.X_train, feature_names=self.feature_names, show=False)
                plt.title(f"{name} — SHAP summary")
            except Exception as e:
                warnings.warn(f"[{name}] summary plot generation failed: {e}")
                plt.text(0.5, 0.5, "summary_plot failed", ha="center", va="center")
            
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

        # Validate sample indices
        for idx in sample_idx:
            if idx < 0 or idx >= len(self.X_train):
                warnings.warn(f"sample_idx {idx} out of range for X_train (len={len(self.X_train)}). Skipping.")
                continue

        # Create subplot grid
        fig, axes = plt.subplots(nrows=n_samples, ncols=n_models,
                                figsize=(figsize_per_row[0] * n_models, figsize_per_row[1] * n_samples))
        axes = np.array(axes)
        if axes.ndim == 1:
            axes = axes.reshape((n_samples, n_models))

        # Loop over instances and models
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

        # Create subplot grid
        fig, axes = plt.subplots(nrows=n_samples, ncols=n_models,
                                figsize=(figsize_per_row[0] * n_models, figsize_per_row[1] * n_samples))
        axes = np.array(axes)
        if axes.ndim == 1:
            axes = axes.reshape((n_samples, n_models))

        # Loop over instances and models
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

        if n_models == 0:
            warnings.warn("No models supplied.")
            return None

        fig, axes = plt.subplots(nrows=n_samples, ncols=n_models,
                                figsize=(figsize_per_row[0] * max(1, n_models), figsize_per_row[1] * max(1, n_samples)))
        axes = np.array(axes)
        if axes.ndim == 1:
            axes = axes.reshape((n_samples, n_models))

        explainer = LimeTabularExplainer(self.X_train.values,
                                        feature_names=self.feature_names,
                                        class_names=self.class_names,
                                        discretize_continuous=discretize_continuous)

        for i, idx in enumerate(sample_idx):
            if idx < 0 or idx >= len(self.X_train):
                warnings.warn(f"sample_idx {idx} out of range. Skipping.")
                for j in range(n_models):
                    ax = axes[i, j]
                    ax.text(0.5, 0.5, "invalid sample_idx", ha="center", va="center")
                continue

            x_row = self.X_train.iloc[idx]
            for j, (name, model) in enumerate(models):
                ax = axes[i, j]
                if not hasattr(model, "predict_proba"):
                    warnings.warn(f"[{name}] model has no predict_proba. Skipping LIME for this model.")
                    ax.text(0.5, 0.5, f"No predict_proba\n({name})", ha="center", va="center")
                    continue

                predict_fn = model.predict_proba
                try:
                    exp = explainer.explain_instance(x_row.values, predict_fn,
                                                    num_features=num_features, num_samples=num_samples)
                    items = exp.as_list()
                    feats, weights = zip(*items) if items else ([], [])
                    y_pos = np.arange(len(feats))
                    ax.barh(y_pos, weights)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(feats)
                    ax.invert_yaxis()
                    ax.set_title(f"{name} — LIME (idx={idx})")
                except Exception as e:
                    warnings.warn(f"[{name}] LIME explain failed: {e}")
                    ax.text(0.5, 0.5, "LIME failed", ha="center", va="center")

        plt.tight_layout()
        plt.show()