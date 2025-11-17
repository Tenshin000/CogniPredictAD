import os
import copy
import numpy as np
import graphviz
import matplotlib.pyplot as plt
import pickle

from imblearn.pipeline import Pipeline
from sklearn.tree import export_text, export_graphviz, plot_tree


def save_xai_models(X_train, models_dir="../results/all_models"):
    """
    Load Decision Tree models from `models_dir` (even if inside Pipelines)
    and save XAI outputs (rules and tree images) as TXT and PDF.
    Only models whose filenames start with "Decision_Tree" are processed.
    """
    try:
        trees_dir = os.path.join(models_dir, "trees")
        rules_dir = os.path.join(models_dir, "rules")
        os.makedirs(trees_dir, exist_ok=True)
        os.makedirs(rules_dir, exist_ok=True)

        feature_names = list(X_train.columns) if hasattr(X_train, "columns") else [f"f{i}" for i in range(X_train.shape[1])]

        if not os.path.isdir(models_dir):
            raise FileNotFoundError(f"models_dir not found: {models_dir}")

        models = []
        for fname in os.listdir(models_dir):
            if fname.endswith(".pkl") and fname.startswith("Decision_Tree"):
                with open(os.path.join(models_dir, fname), "rb") as f:
                    model = pickle.load(f)
                models.append((os.path.splitext(fname)[0], model))
        print(f"Found {len(models)} Decision_Tree models:", [n for n, _ in models])

        desired_order = ["CN", "EMCI", "LMCI", "AD"] # from least to most severe

        for model_name, model in models:
            tree_model = None
            if isinstance(model, Pipeline):
                tree_model = model.named_steps.get('clf', None) or next(
                    (s for s in model.named_steps.values() if hasattr(s, "tree_")), None
                )
                tree_model = tree_model or model
            else:
                tree_model = model

            if not hasattr(tree_model, "tree_"):
                raise ValueError(f"{model_name} is not a DecisionTree model.")

            orig_classes = list(getattr(tree_model, "classes_", []))
            display_classes = [c for c in desired_order if c in orig_classes] + [c for c in orig_classes if c not in desired_order]
            idx_map = [orig_classes.index(c) for c in display_classes]

            tree_copy = copy.deepcopy(tree_model)

            # Reorder values along class axis
            try:
                tree_copy.tree_.value[:] = tree_copy.tree_.value[:, :, idx_map]
            except Exception:
                pass
            tree_copy.classes_ = np.array(display_classes)
            tree_copy.n_classes_ = len(display_classes)

            # --- Adjust leaf nodes to pick worst class if tied ---
            for i in range(tree_copy.tree_.node_count):
                if tree_copy.tree_.children_left[i] == tree_copy.tree_.children_right[i] == -1:  # leaf node
                    vals = tree_copy.tree_.value[i, 0]  # shape: (n_classes,)
                    max_val = vals.max()
                    tied_classes = [display_classes[j] for j, v in enumerate(vals) if v == max_val]
                    if len(tied_classes) > 1:
                        # pick the class with highest severity
                        worst_class = max(tied_classes, key=lambda c: desired_order.index(c))
                        tree_copy.tree_.value[i, 0, :] = 0
                        tree_copy.tree_.value[i, 0, display_classes.index(worst_class)] = max_val

            # --- Export textual rules ---
            rules_text = export_text(tree_copy, feature_names=feature_names)
            rules_text_ifthen = []
            for line in rules_text.splitlines():
                stripped = line.lstrip()
                indent = (len(line) - len(stripped)) // 4
                if stripped.startswith("|---"):
                    rules_text_ifthen.append("    " * indent + "IF " + stripped.replace("|---", "").strip())
                else:
                    rules_text_ifthen.append("    " * indent + stripped.replace("class:", "THEN class ="))
            rules_text = "\n".join(rules_text_ifthen)

            with open(os.path.join(rules_dir, f"{model_name}_rules.txt"), "w", encoding="utf-8") as f:
                f.write(rules_text)

            # --- Export tree diagram ---
            class_names = [str(c) for c in getattr(tree_copy, "classes_", [])] or None
            try:
                dot_data = export_graphviz(
                    tree_copy, out_file=None, feature_names=feature_names,
                    class_names=class_names, filled=True, rounded=True,
                    special_characters=True, label='all'
                )
                graph = graphviz.Source(dot_data)
                graph.render(os.path.join(trees_dir, f"{model_name}_tree"), format="pdf", cleanup=True)
            except Exception:
                fig, ax = plt.subplots(figsize=(16, 12))
                plot_tree(tree_copy, feature_names=feature_names, class_names=class_names,
                          filled=True, rounded=True, ax=ax, label='all')
                fig.savefig(os.path.join(trees_dir, f"{model_name}_tree.pdf"), bbox_inches="tight", dpi=600)
                plt.close(fig)

        print("XAI outputs saved.")

    except Exception as e:
        print(f"Error in save_xai_models: {e}")
