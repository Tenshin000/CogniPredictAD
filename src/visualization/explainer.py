import copy
import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import warnings

from IPython.display import Image, display
from lime.lime_tabular import LimeTabularExplainer
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.tree import export_graphviz, plot_tree, _tree
from sklearn.utils.validation import check_is_fitted
from typing import Union, Optional, List, Sequence, Tuple, Any


class ModelExplainer:
    """
    ModelExplainer
    --------------
    Utilities to produce model-agnostic and model-specific explanations for one or more classifiers.
    Supports SHAP (global and local plots), LIME (local explanations), and visualization of DecisionTreeClassifier objects.
    Designed to accept an iterable of (name, model) pairs and a representative dataset (X_train) used as background / reference data for the explainers.

    Key features:
      - Robust attempt to construct a SHAP explainer using several fallbacks (callable, predict_proba, predict,
        or background-based fallback). Returns both the explainer and a short description of the method used.
      - Convenience methods to produce grids of SHAP summary / waterfall / force plots for multiple models.
      - Per-model SHAP summary-only plotting, waterfall-only, and force-only functions to let the user pick
        a specific visualization mode and set of instances.
      - LIME explanations laid out side-by-side for quick per-instance comparisons across models.
      - Decision tree unwrapping and visualization that attempts to find an inner DecisionTreeClassifier inside
        common wrappers (Pipeline, GridSearchCV, ensemble wrappers, etc.), reorder classes to a desired ordering,
        and adjust leaf ties consistently.
      - Defensive programming: many try/except blocks and warnings for graceful degradation when a particular
        explainer or plot type is not available for a model.

    Notes:
      - The class expects models to be already fitted when used with plotting functions that access internals.
      - SHAP behavior may vary across SHAP versions; the implementation tries multiple APIs (shap.plots.beeswarm,
        shap.summary_plot, shap.plots.waterfall, shap.plots.force) and falls back sensibly.
      - LIME requires models to expose predict_proba; the method warns and skips models that lack it.
    """
    # ----------------------------#
    #        INITIALIZATION       #
    # ----------------------------#
    def __init__(self, models: Sequence[Tuple[str, Any]], X_train: pd.DataFrame, y_train: Optional[pd.Series] = None,
                 feature_names: Optional[List[str]] = None, class_names: Optional[List[str]] = None,
                 background_size: int = 100, random_state: int = 42):
        """
        Parameters:
            models : sequence of (name, model)
                Iterable of (string name, fitted estimator) pairs to explain.
            X_train : pd.DataFrame
                Representative feature data used by explainers (background/feature matrix).
            y_train : pd.Series, optional
                Optional labels; may be useful when selecting instances or for LIME class_names.
            feature_names : list[str], optional
                Names of features; defaults to X_train.columns when X_train is a DataFrame.
            class_names : list[str], optional
                Optional class labels (strings) for multiclass outputs.
            background_size : int
                Maximum number of background samples to use when constructing fallback SHAP explainers.
            random_state : int
                Seed used when subsampling the background data.
        """
        # Store models as list of (name, model) pairs
        self.models = list(models)
        # Copy of training data (used for explanations)
        self.X_train = X_train.copy()
        # copy y_train only if provided
        self.y_train = y_train.copy() if y_train is not None else None
        # Feature names for visualization
        self.feature_names = feature_names or list(X_train.columns)
        # Optional class names (useful for multiclass problems)
        self.class_names = class_names
        # Background sample for SHAP explanations
        self.background = self.X_train.sample(min(background_size, len(self.X_train)), random_state=random_state)

    # ----------------------------#
    #         SHAP HELPERS        #
    # ----------------------------#
    def _get_shap_explainer(self, name: str, model: Any):
        """Try multiple ways to create a shap.Explainer. 
        Returns (explainer, used_method) or (None, reason)."""
        # Always use X_train (copied) for explainer construction
        X_for_model = self.X_train.copy()
        # Decide output names to pass to SHAP (prefer user-provided class_names, otherwise try to infer from the model if available)
        output_names = None
        if self.class_names is not None:
            output_names = self.class_names
        elif hasattr(model, "classes_"):
            output_names = [str(c) for c in getattr(model, "classes_")]

        # 1) If model is callable directly
        if callable(model):
            try:
                expl = shap.Explainer(model, X_for_model, feature_names=self.feature_names, output_names=output_names)
                return expl, "callable(model)"
            except Exception as e:
                warnings.warn(f"[{name}] shap.Explainer(callable model) failed: {e}")

        # 2) predict_proba
        if hasattr(model, "predict_proba") and callable(getattr(model, "predict_proba")):
            try:
                predict_proba = lambda X_input: model.predict_proba(X_input)
                expl = shap.Explainer(predict_proba, X_for_model, feature_names=self.feature_names, output_names=output_names)
                return expl, "model.predict_proba"
            except Exception as e:
                warnings.warn(f"[{name}] shap.Explainer(model.predict_proba) failed: {e}")

        # 3) predict
        if hasattr(model, "predict") and callable(getattr(model, "predict")):
            try:
                predict = lambda X_input: model.predict(X_input)
                expl = shap.Explainer(predict, X_for_model, feature_names=self.feature_names, output_names=output_names)
                return expl, "model.predict"
            except Exception as e:
                warnings.warn(f"[{name}] shap.Explainer(model.predict) failed: {e}")

        # 4) Fallback: build a generic shap.Explainer letting SHAP pick TreeExplainer if appropriate
        try:
            expl = shap.Explainer(model, self.background, feature_names=self.feature_names, output_names=output_names)
            return expl, "shap.Explainer(fallback_with_background)"
        except Exception as e:
            warnings.warn(f"[{name}] fallback shap.Explainer attempt failed: {e}")

        return None, "no usable callable or explainer"

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

        # Create grid reliably: n_models rows, 3 cols
        fig, axes = plt.subplots(nrows=n_models, ncols=3,
                                figsize=(figsize_per_row[0] * 3 if n_models == 1 else figsize_per_row[0] * 3,
                                        figsize_per_row[1] * n_models))
        # Normalize to 2D array shape (n_models, 3)
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

            # Compute shap values
            try:
                shap_vals = explainer(self.X_train)
            except Exception as e:
                warnings.warn(f"[{name}] Explainer(...) on X_train failed: {e}. Skipping this model.")
                for a in (ax_summary, ax_water, ax_force):
                    a.text(0.5, 0.5, f"SHAP failed\n({name})", ha="center", va="center")
                continue

            # SUMMARY (beeswarm / summary)
            try:
                plt.sca(ax_summary)   # set current axis
                # Try to use the SHAP matplotlib backend (many versions respect current axis with matplotlib=True)
                # Prefer shap.plots.beeswarm or summary_plot with show=False 
                try:
                    # shap.plots.beeswarm accepts an Explanation or array 
                    shap.plots.beeswarm(shap_vals, show=False)
                except Exception:
                    # Fallback to summary_plot (older API)
                    shap.summary_plot(shap_vals, features=self.X_train, feature_names=self.feature_names, show=False)
                ax_summary.set_title(f"{name} — SHAP summary")
            except Exception as e:
                warnings.warn(f"[{name}] summary_plot failed: {e}")
                ax_summary.text(0.5, 0.5, "summary_plot failed", ha="center", va="center")

            # Validate sample index
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
            try:
                shap_vals = explainer(self.X_train)
            except Exception as e:
                warnings.warn(f"[{name}] Explainer(...) failed: {e}. Skipping.")
                continue

            vals_shape = getattr(shap_vals, "values", None).shape if hasattr(shap_vals, "values") else None

            # Determine class names
            class_names = self.class_names or list(getattr(model, "classes_", range(vals_shape[2] if vals_shape else 1)))

            # Convert to array if needed
            vals = shap_vals.values if hasattr(shap_vals, "values") else np.array(shap_vals)

            if vals.ndim == 3:  # multiclass case: (samples, features, classes) or (samples, features, outputs)
                # SHAP sometimes gives (samples, features, classes) or (samples, features, outputs)
                n_classes = vals.shape[2]
                for cls_idx, cname in enumerate(class_names[:n_classes]):
                    plt.figure(figsize=figsize)
                    try:
                        shap.summary_plot(vals[:, :, cls_idx], features=self.X_train, feature_names=self.feature_names, show=False)
                        plt.title(f"{name} — SHAP summary — {cname}")
                    except Exception:
                        # Fallback: beeswarm with Explanation object
                        try:
                            single_expl = shap.Explanation(values=vals[:, :, cls_idx], feature_names=self.feature_names)
                            shap.plots.beeswarm(single_expl, show=False)
                            plt.title(f"{name} — SHAP summary — {cname}")
                        except Exception as e:
                            warnings.warn(f"[{name}] per-class summary failed for {cname}: {e}")
                    plt.tight_layout()
                    plt.show()
            else:
                # Single-output case
                plt.figure(figsize=figsize)
                try:
                    shap.summary_plot(shap_vals, features=self.X_train, feature_names=self.feature_names, show=False)
                    plt.title(f"{name} — SHAP summary")
                except Exception as e:
                    warnings.warn(f"[{name}] summary plot failed: {e}")
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

    # ----------------------------#
    #            TREES            #
    # ----------------------------#
    def plot_decision_trees(self, max_depth: Optional[int] = None, fontsize: int = 10):
        """
        Display Decision Tree models (whose name starts with 'Decision_Tree')
        using Matplotlib. This version unwraps common wrappers (Pipeline, GridSearchCV,
        AdaBoost/Bagging, Voting, etc.) to find the underlying DecisionTreeClassifier
        instance that actually contains `.tree_`.
        """
        def _unwrap_to_tree(est):
            """Recursively try to locate an object that has .tree_ (DecisionTreeClassifier)."""
            seen = set()
            queue = [est]
            while queue:
                cur = queue.pop(0)
                if cur is None:
                    continue
                cid = id(cur)
                if cid in seen:
                    continue
                seen.add(cid)

                # Direct DecisionTreeClassifier-like
                if hasattr(cur, "tree_"):
                    return cur

                # Common wrapper attributes that may contain an estimator
                if hasattr(cur, "best_estimator_"):
                    queue.append(cur.best_estimator_)
                if hasattr(cur, "estimator_"):
                    queue.append(cur.estimator_)
                if hasattr(cur, "base_estimator_"):
                    queue.append(cur.base_estimator_)
                # Sklearn Pipelines
                if hasattr(cur, "steps") and isinstance(getattr(cur, "steps"), (list, tuple)) and len(cur.steps) > 0:
                    queue.append(cur.steps[-1][1])
                if hasattr(cur, "named_steps") and isinstance(getattr(cur, "named_steps"), dict) and len(cur.named_steps) > 0:
                    queue.append(list(cur.named_steps.values())[-1])
                # Ensembles like VotingClassifier, BaggingClassifier, AdaBoostClassifier
                if hasattr(cur, "estimators_") and getattr(cur, "estimators_", None):
                    # estimators_ is usually a list of estimators or list of (name, estimator)
                    for e in cur.estimators_:
                        queue.append(e[1] if isinstance(e, tuple) else e)
                if hasattr(cur, "estimators") and getattr(cur, "estimators", None):
                    for e in cur.estimators:
                        queue.append(e[1] if isinstance(e, tuple) else e)
                # Sometimes "base_estimator" (without trailing underscore)
                if hasattr(cur, "base_estimator") and getattr(cur, "base_estimator", None):
                    queue.append(cur.base_estimator)
            return None

        found = False
        desired_order = ["CN", "EMCI", "LMCI", "AD"]

        for name, model in self.models:
            if not name.startswith("Decision_Tree"):
                continue
            found = True

            try:
                inner = _unwrap_to_tree(model)
                if inner is None:
                    print(f"[{name}] could not find underlying DecisionTreeClassifier (no .tree_ in object or wrappers).")
                    continue

                # Ensure fitted
                try:
                    check_is_fitted(inner, attributes="tree_")
                except Exception:
                    print(f"[{name}] underlying DecisionTreeClassifier is not fitted (no .tree_).")
                    continue

                # Work on a deep copy of the real tree estimator
                tree_copy = copy.deepcopy(inner)

                # Obtain original class ordering from the estimator if possible
                orig_classes = list(getattr(tree_copy, "classes_", []))
                if not orig_classes:
                    # Fallback: try outer model classes_
                    orig_classes = list(getattr(model, "classes_", [])) or orig_classes

                # Reorder classes according to desired_order, preserving any other classes
                display_classes = [c for c in desired_order if c in orig_classes] + [c for c in orig_classes if c not in desired_order]
                if display_classes:
                    # Map indices. If mapping fails, skip reordering
                    try:
                        idx_map = [orig_classes.index(c) for c in display_classes]
                        # tree_.value shape: (node_count, n_outputs, n_classes) in sklearn
                        try:
                            tree_copy.tree_.value[:] = tree_copy.tree_.value[:, :, idx_map]
                        except Exception:
                            # Some sklearn versions have different shapes, attempt a safer reorder if possible
                            val = tree_copy.tree_.value
                            if val.ndim == 3 and val.shape[2] == len(orig_classes):
                                reordered = val[:, :, idx_map]
                                tree_copy.tree_.value[:] = reordered
                        tree_copy.classes_ = np.array(display_classes)
                    except Exception:
                        # If any issue, leave classes as-is
                        pass

                tree_copy.n_classes_ = len(getattr(tree_copy, "classes_", []))

                # Adjust leaf nodes to pick worst class if tied (keep original logic but robust)
                try:
                    node_count = tree_copy.tree_.node_count
                    children_left = tree_copy.tree_.children_left
                    children_right = tree_copy.tree_.children_right
                    values = tree_copy.tree_.value  # shape: (node_count, n_outputs, n_classes)
                    for i in range(node_count):
                        if children_left[i] == children_right[i] == -1:  # leaf
                            vals = values[i, 0] if values.ndim == 3 else values[i]
                            if vals is None or len(vals) == 0:
                                continue
                            max_val = np.max(vals)
                            tied_idx = [k for k, v in enumerate(vals) if v == max_val]
                            if len(tied_idx) > 1 and getattr(tree_copy, "classes_", None) is not None:
                                # Choose worst class based on desired_order priority
                                display = list(tree_copy.classes_)
                                def severity_index(c):
                                    try:
                                        return desired_order.index(c)
                                    except ValueError:
                                        return len(desired_order)  # less severe
                                tied_classes = [display[k] for k in tied_idx]
                                worst_class = max(tied_classes, key=severity_index)
                                # Construct a new vector: set all to max_val - 1 and worst to max_val
                                new_vec = np.full_like(vals, fill_value=max_val - 1)
                                new_vec[display.index(worst_class)] = max_val
                                if values.ndim == 3:
                                    tree_copy.tree_.value[i, 0, :] = new_vec
                                else:
                                    tree_copy.tree_.value[i, :] = new_vec
                except Exception:
                    pass

                # Plot
                plt.figure(figsize=(40, 40))
                plot_tree(
                    tree_copy,
                    feature_names=[str(f) for f in self.feature_names],
                    class_names=[str(c) for c in getattr(tree_copy, "classes_", [])],
                    filled=True,
                    fontsize=fontsize,
                    max_depth=max_depth,
                    label='all'
                )
                plt.title(f"Decision Tree visualization — {name}")
                plt.show()

            except Exception as e:
                print(f"[{name}] Error while generating the decision tree: {e}")

        if not found:
            print("No models found with a name starting with 'Decision_Tree'.")
