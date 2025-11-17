"""
main.py

Medical prediction GUI using customtkinter.

- Comments and printed outputs are in English.
- UI flow:
  * Blue start page with "Start"
  * White selection page to choose one of the 4 models (Model/XAIModel/AltModel/AltXAIModel)
  * After selection, descriptions disappear and the input fields appear
  * Prediction is displayed in red; user can Confirm or Contest diagnosis
  * Confirm/Contest appends a row to data/NEWADNIMERGE.csv (creates folder/file if needed)

Author: Francesco Panattoni
"""

import os
import customtkinter as ctk
import joblib
import pandas as pd
import numpy as np
import tkinter as tk
import tkinter.messagebox as messagebox

# ----------------------------#
#       CONFIGURATIONS        #
# ----------------------------#
ctk.set_appearance_mode("Light")   # "System", "Dark", "Light"
ctk.set_default_color_theme("blue")  

RESULTS_DIR = "results"
DATA_DIR = "data"
NEW_CSV = os.path.join(DATA_DIR, "NEWADNIMERGE.csv")

# Column names and order used in the CSV (DX first). These remain the ratio/normalized names
COLUMN_ORDER = [
    "DX", "AGE", "PTGENDER", "PTEDUCAT", "APOE4", "MMSE", "CDRSB", "ADAS13",
    "LDELTOTAL", "FAQ", "MOCA", "TRABSCOR", "RAVLT_immediate", "RAVLT_learning",
    "RAVLT_perc_forgetting", "mPACCdigit", "EcogPtMem", "EcogPtLang", "EcogPtVisspat",
    "EcogPtPlan", "EcogPtOrgan", "EcogPtDivatt", "EcogSPMem", "EcogSPLang",
    "EcogSPVisspat", "EcogSPPlan", "EcogSPOrgan", "EcogSPDivatt", "FDG",
    "TAU/ABETA", "PTAU/ABETA", "Hippocampus/ICV", "Entorhinal/ICV",
    "Fusiform/ICV", "MidTemp/ICV", "Ventricles/ICV", "WholeBrain/ICV"
]

# ALL_PARAMS now contains raw biomarker and raw structural columns (not ratios)
ALL_PARAMS = [
    "AGE", "PTGENDER", "PTEDUCAT", "APOE4", "MMSE", "CDRSB", "ADAS13",
    "LDELTOTAL", "FAQ", "MOCA", "TRABSCOR", "RAVLT_immediate", "RAVLT_learning",
    "RAVLT_perc_forgetting", "mPACCdigit", "EcogPtMem", "EcogPtLang", "EcogPtVisspat",
    "EcogPtPlan", "EcogPtOrgan", "EcogPtDivatt", "EcogSPMem", "EcogSPLang",
    "EcogSPVisspat", "EcogSPPlan", "EcogSPOrgan", "EcogSPDivatt", "FDG",
    # raw biomarkers
    "TAU", "PTAU", "ABETA",
    # raw structural volumes and ICV
    "Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain", "ICV"
]

# The three features removed for AltModel/AltXAIModel (they remain in the CSV though)
REMOVE_FOR_ALT_MODEL = ["CDRSB", "LDELTOTAL", "mPACCdigit"]

# type/range validation map (min, max, type)
# For biomarker/structural we use wide but positive ranges
VALIDATION = {
    "AGE": (0, 120, int),
    "APOE4": (0, 2, int),
    "MMSE": (0, 30, int),
    "LDELTOTAL": (0, 30, int),
    "FAQ": (0, 30, int),
    "MOCA": (0, 30, int),
    "CDRSB": (0, 18, float),
    "ADAS13": (0, 85, int),
    "TRABSCOR": (15, 300, int),
    "RAVLT_immediate": (0, 75, int),
    "RAVLT_learning": (-15, 15, int),
    "RAVLT_perc_forgetting": (-100, 100, float),
    "EcogPtMem": (1, 4, float),
    "EcogPtLang": (1, 4, float),
    "EcogPtVisspat": (1, 4, float),
    "EcogPtPlan": (1, 4, float),
    "EcogPtOrgan": (1, 4, float),
    "EcogPtDivatt": (1, 4, float),
    "EcogSPMem": (1, 4, float),
    "EcogSPLang": (1, 4, float),
    "EcogSPVisspat": (1, 4, float),
    "EcogSPPlan": (1, 4, float),
    "EcogSPOrgan": (1, 4, float),
    "EcogSPDivatt": (1, 4, float),
    # biomarkers: allow non-negative floats with very wide upper bound
    "TAU": (0, 1e5, float),
    "PTAU": (0, 1e5, float),
    "ABETA": (0, 1e5, float),
    # structures & ICV: allow positive floats (volumes)
    "Hippocampus": (1e-6, 1e7, float),
    "Entorhinal": (1e-6, 1e7, float),
    "Fusiform": (1e-6, 1e7, float),
    "MidTemp": (1e-6, 1e7, float),
    "Ventricles": (1e-6, 1e7, float),
    "WholeBrain": (1e-6, 1e10, float),
    "ICV": (1e-6, 1e10, float)  # ICV must be positive and not zero
}

# Diagnosis mapping using string labels
DIAG_MAP = {
    "CN": "Cognitively Normal (CN)",
    "EMCI": "Early Mild Cognitive Impairment (EMCI)",
    "LMCI": "Late Mild Cognitive Impairment (LMCI)",
    "AD": "Alzheimer's Disease (AD)"
}

# Reverse mapping numeric->string for compatibility
NUM_TO_LABEL = {
    0: "CN",
    1: "EMCI",
    2: "LMCI",
    3: "AD"
}

# ----------------------------#
#           UTILITY           #
# ----------------------------#
class ModelLoader:
    """Class to load a scikit-learn style pickle and run predictions."""
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def load(self):
        """Load the model from file using joblib."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.model = joblib.load(self.model_path)
        if not hasattr(self.model, "predict"):
            raise ValueError("Loaded object does not have a predict() method.")
        return self.model

    def predict(self, X_df):
        """
        Predict using the loaded model.
        X_df: pandas DataFrame with the features expected by the model.
        Returns string label ("CN","EMCI","LMCI","AD").

        Handles models that return either numeric labels (0..3) or string labels
        and normalizes them to canonical string codes.
        """
        if self.model is None:
            self.load()

        # helper to normalize a raw class value to canonical label (e.g. 0 -> "CN", "Cognitively Normal (CN)" -> "CN")
        def _normalize_raw_label(raw):
            # bytes -> str
            if isinstance(raw, (bytes, bytearray)):
                try:
                    raw = raw.decode("utf-8")
                except Exception:
                    raw = str(raw)
            # numeric -> map via NUM_TO_LABEL
            try:
                num = int(float(raw))
                if num in NUM_TO_LABEL:
                    return NUM_TO_LABEL[num]
            except Exception:
                pass
            # string -> try tokens
            if isinstance(raw, str):
                key = raw.strip().upper()
                if key in DIAG_MAP:
                    return key
                for token in DIAG_MAP.keys():
                    if token in key:
                        return token
                if "NORMAL" in key or "COGNITIVELY" in key:
                    return "CN"
                if "EARLY" in key and "MCI" in key:
                    return "EMCI"
                if "LATE" in key and "MCI" in key:
                    return "LMCI"
                if "ALZ" in key or "ALZHEIMER" in key:
                    return "AD"
            raise ValueError(f"Unrecognized class label from model: {repr(raw)}")

        # severity order: higher index = worse
        severity_order = ["CN", "EMCI", "LMCI", "AD"]
        severity_rank = {lab: i for i, lab in enumerate(severity_order)}

        # If model supports predict_proba, use it to decide (and handle ties)
        if hasattr(self.model, "predict_proba") and hasattr(self.model, "classes_"):
            try:
                probs = self.model.predict_proba(X_df)  # shape (n_samples, n_classes)
                probs_row = np.asarray(probs).ravel()[0]  # first sample
                classes = list(self.model.classes_)
                # normalize class names
                norm_classes = []
                for c in classes:
                    try:
                        norm = _normalize_raw_label(c)
                    except Exception:
                        # fallback: try direct string conversion
                        norm = str(c).strip().upper()
                    norm_classes.append(norm)

                # pair classes with probs
                pairs = list(zip(norm_classes, probs_row))
                # find maximum probability
                max_prob = float(np.max(probs_row))
                # find classes tied for max (use isclose for safety)
                tied = [lab for lab, p in pairs if np.isclose(p, max_prob, atol=1e-12, rtol=0)]
                if len(tied) == 1:
                    return tied[0]
                # tie -> choose the worst according to severity_rank
                # filter tied labels that are known, otherwise keep as-is
                tied_known = [t for t in tied if t in severity_rank]
                if tied_known:
                    # pick the one with largest rank (worst)
                    worst = max(tied_known, key=lambda x: severity_rank[x])
                    return worst
                # if none of the tied labels are in severity map, normalize first tied
                return tied[0]
            except Exception:
                # on any error, fall back to predict()
                pass

        # fallback: use predict() and normalize single output (handles array-like too)
        preds = self.model.predict(X_df)
        if isinstance(preds, (list, tuple, np.ndarray, pd.Series)):
            raw = np.asarray(preds).ravel()[0]
        else:
            raw = preds
        return _normalize_raw_label(raw)

class PageManager:
    """Class to manage customtkinter pages and flows."""
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.title("CogniPredictAD: Medical Classifier for Alzheimer")
        self.root.geometry("1000x760")
        # State variables
        self.model_choice = None
        self.model_loader = None
        self.model_descs = {
            "Model1.pkl": "This model is based on an Extra Trees classifier, which builds many decision trees using randomized feature splits to improve robustness and reduce overfitting. It draws on multiple clinical and cognitive measures, including CDRSB, LDELTOTAL, and mPACCdigit, to estimate the patient's cognitive status. Its primary strength is the ability to capture complex, non-linear relationships in the data, though the internal decision process can be difficult to interpret.",
            "XAIModel1.pkl": "This model uses a Decision Tree, a simpler method where predictions are made by following a clear series of if-then rules based on the patient's test scores. It includes CDRSB, LDELTOTAL, and mPACCdigit in its analysis. Because of its structure, the model is easily explainable: doctors can see exactly which variables and thresholds lead to the final diagnosis. While it may be less accurate than more complex models, it provides valuable transparency for clinical decision-making.",
            "AltModel.pkl": "This model uses Adaptive Boosting (AdaBoost), an ensemble technique that combines many weak learners, typically shallow decision trees, by iteratively reweighting the training samples to focus on harder cases. It analyzes several clinical and cognitive features but excludes CDRSB, LDELTOTAL, and mPACCdigit from the prediction, so it may be less accurate.",
            "AltXAIModel.pkl": "This model is also based on a Decision Tree, but it makes predictions without using CDRSB, LDELTOTAL, and mPACCdigit. Like XAIModel1, it follows a transparent rule-based structure, making its decisions easy to trace and understand. The absence of those three variables makes it useful in clinical contexts where those specific measures are not available, while still allowing doctors to follow the diagnostic reasoning step by step."
        }
        self.inputs = {}  # param -> widget
        self.last_parsed = None
        self.last_prediction_label = None 
        self.last_feature_row = None

        # Frames for pages
        self.start_frame = None
        self.selection_frame = None
        self.input_frame = None

        # For cancelling an insert
        self.last_saved_row_index = None

        # Build UI
        self.build_start_page()

    def build_start_page(self):
        """Blue start page with Start button and appearance mode selector."""
        if self.selection_frame:
            self.selection_frame.pack_forget()
        if self.input_frame:
            self.input_frame.pack_forget()

        # Create start frame
        self.start_frame = ctk.CTkFrame(self.root, fg_color="#87CEEB")
        self.start_frame.pack(fill="both", expand=True, padx=20, pady=40)

        # Top spacer
        spacer_top = ctk.CTkLabel(self.start_frame, text="")
        spacer_top.pack(pady=12)

        # Title
        title = ctk.CTkLabel(self.start_frame, text="CogniPredictAD", font=("Helvetica", 24, "bold"))
        title.pack(pady=(0, 8))

        # Appearance selector (placed directly under the title)
        appearance_frame = ctk.CTkFrame(self.start_frame, fg_color="transparent")
        appearance_frame.pack(pady=(0, 10))

        appearance_label = ctk.CTkLabel(appearance_frame, text="Appearance mode:", anchor="w")
        appearance_label.grid(row=0, column=0, padx=(0,8))

        # Store appearance var on the instance so other methods can access it if needed
        self.appearance_var = ctk.StringVar(value=ctk.get_appearance_mode())

        # Use CTkComboBox when available; fallback to OptionMenu otherwise
        try:
            combo = ctk.CTkComboBox(appearance_frame, values=["Light", "Dark"], variable=self.appearance_var, width=140)
            combo.set(self.appearance_var.get())
            combo.grid(row=0, column=1, padx=(0,8))
            # apply immediately when value changes
            def _on_appearance_change(*_):
                mode = self.appearance_var.get()
                try:
                    ctk.set_appearance_mode(mode)
                except Exception:
                    pass
            # trace variable changes
            self.appearance_var.trace_add("write", _on_appearance_change)
        except Exception:
            # fallback to tkinter OptionMenu
            var = self.appearance_var
            om = tk.OptionMenu(appearance_frame, var, "Light", "Dark", command=lambda _: ctk.set_appearance_mode(var.get()))
            om.config(width=12)
            om.grid(row=0, column=1, padx=(0,8))

        # Subtitle / instruction (placed after appearance selector)
        subtitle = ctk.CTkLabel(self.start_frame, text="Press Start to continue", font=("Helvetica", 12))
        subtitle.pack(pady=(6, 18))

        # Start button
        start_btn = ctk.CTkButton(self.start_frame, text="Start", width=160, height=48,
                                  command=self.on_start_pressed)
        start_btn.pack(pady=6)

        # Bottom spacer
        spacer_bot = ctk.CTkLabel(self.start_frame, text="")
        spacer_bot.pack(expand=True)

    def on_start_pressed(self):
        """Handle Start pressed."""
        self.start_frame.pack_forget()
        self.build_selection_page()

    def build_selection_page(self):
        """White page to select the model and show descriptions."""
        if self.input_frame:
            self.input_frame.pack_forget()
        if self.start_frame:
            self.start_frame.pack_forget()

        self.selection_frame = ctk.CTkFrame(self.root)
        self.selection_frame.pack(fill="both", expand=True, padx=12, pady=12)

        title = ctk.CTkLabel(self.selection_frame, text="Model selection", font=("Helvetica", 18, "bold"))
        title.pack(pady=(6, 8))

        # Radio buttons for models
        self.model_var = ctk.StringVar(value="Model1.pkl")
        models = ["Model.pkl", "XAIModel.pkl", "AltModel.pkl", "AltXAIModel.pkl"]
        radios_frame = ctk.CTkFrame(self.selection_frame, fg_color="transparent")
        radios_frame.pack(pady=(4,8))
        for m in models:
            r = ctk.CTkRadioButton(radios_frame, text=m, variable=self.model_var, value=m)
            r.pack(anchor="w", padx=8, pady=2)

        # Multi-line description
        desc_label = ctk.CTkLabel(self.selection_frame, text="Model descriptions:", anchor="w")
        desc_label.pack(fill="x", padx=8, pady=(10,0))
        self.desc_text = ctk.CTkTextbox(self.selection_frame, width=960, height=375)
        # Populate with combined descriptions
        combined = self.get_combined_descriptions()
        self.desc_text.insert("0.0", combined)
        self.desc_text.pack(padx=8, pady=(4,8))

        instr = ctk.CTkLabel(self.selection_frame, text="Select a model and then click 'Load Model' to continue.")
        instr.pack(pady=(4,8))

        btns_frame = ctk.CTkFrame(self.selection_frame, fg_color="transparent")
        btns_frame.pack(pady=6)
        load_btn = ctk.CTkButton(btns_frame, text="Load Model", command=self.on_load_model)
        load_btn.grid(row=0, column=0, padx=8)
        back_btn = ctk.CTkButton(btns_frame, text="Back", command=self.on_back_to_start)
        back_btn.grid(row=0, column=1, padx=8)
        exit_btn = ctk.CTkButton(btns_frame, text="Exit", command=self.root.quit)
        exit_btn.grid(row=0, column=2, padx=8)

        placeholder = ctk.CTkLabel(self.selection_frame, text="Model not loaded yet.", fg_color="transparent")
        placeholder.pack(pady=20)

    def on_back_to_start(self):
        """Back to start page"""
        self.selection_frame.pack_forget()
        self.build_start_page()

    def get_combined_descriptions(self):
        """Return a combined description string for the four models."""
        lines = []
        for k, v in self.model_descs.items():
            lines.append(f"{k}:\n{v}\n")
        return "\n".join(lines)

    def on_load_model(self):
        """Load model selected and proceed to build inputs."""
        chosen = self.model_var.get()
        if not chosen:
            messagebox.showerror("Load Model", "Please select a model.")
            return
        self.model_choice = chosen
        model_path = os.path.join(RESULTS_DIR, chosen)
        try:
            self.model_loader = ModelLoader(model_path)
            self.model_loader.load()
        except Exception as e:
            messagebox.showerror("Load Model", f"Error loading model:\n{e}")
            return

        if self.selection_frame:
            self.selection_frame.pack_forget()

        self.build_input_form()

    def build_input_form(self):
        """Build the long input form and action buttons with scroll support."""
        if hasattr(self, "input_frame") and self.input_frame:
            try:
                self.input_frame.destroy()
            except Exception:
                pass
        self.input_frame = ctk.CTkFrame(self.root)
        self.input_frame.pack(fill="both", expand=True, padx=12, pady=12)

        header = ctk.CTkLabel(self.input_frame, text=f"Inputs for {self.model_choice}", font=("Helvetica", 16, "bold"))
        header.pack(pady=(4,6))

        scrollable_available = hasattr(ctk, "CTkScrollableFrame")
        if scrollable_available:
            scroll_container = ctk.CTkScrollableFrame(self.input_frame, width=960, height=520)
            scroll_container.pack(padx=6, pady=6, fill="both", expand=False)
            content_parent = scroll_container
        else:
            outer = tk.Frame(self.input_frame)
            outer.pack(fill="both", expand=True)
            canvas = tk.Canvas(outer, width=960, height=520)
            vscroll = tk.Scrollbar(outer, orient="vertical", command=canvas.yview)
            canvas.configure(yscrollcommand=vscroll.set)
            vscroll.pack(side="right", fill="y")
            canvas.pack(side="left", fill="both", expand=True)
            inner_frame = tk.Frame(canvas)
            canvas.create_window((0, 0), window=inner_frame, anchor='nw')

            def _on_frame_config(event, c=canvas):
                c.configure(scrollregion=c.bbox("all"))
            inner_frame.bind("<Configure>", _on_frame_config)
            content_parent = inner_frame

        left_frame = ctk.CTkFrame(content_parent, fg_color="transparent") if scrollable_available else tk.Frame(content_parent)
        right_frame = ctk.CTkFrame(content_parent, fg_color="transparent") if scrollable_available else tk.Frame(content_parent)

        if scrollable_available:
            left_frame.grid(row=0, column=0, padx=(6,12), pady=6, sticky="n")
            right_frame.grid(row=0, column=1, padx=(12,6), pady=6, sticky="n")
        else:
            left_frame.pack(side="left", padx=(6,12), pady=6, anchor="n")
            right_frame.pack(side="left", padx=(12,6), pady=6, anchor="n")

        # populate inputs into left/right
        half = (len(ALL_PARAMS) + 1) // 2

        # structural fields that should show units (mm³)
        STRUCT_MM3 = {"Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain", "ICV"}

        for i, param in enumerate(ALL_PARAMS):
            target = left_frame if i < half else right_frame

            # display label: append unit for structural volumes
            display_name = f"{param} (mm³)" if param in STRUCT_MM3 else param

            if scrollable_available:
                label = ctk.CTkLabel(target, text=display_name, anchor="w")
                label.grid(row=(i if i<half else i-half), column=0, padx=6, pady=4, sticky="w")
            else:
                label = tk.Label(target, text=display_name, anchor="w")
                label.grid(row=(i if i<half else i-half), column=0, padx=6, pady=4, sticky="w")

            # create widget
            if param == "PTGENDER":
                if scrollable_available:
                    widget = ctk.CTkComboBox(target, values=["Female", "Male"], width=160)
                    widget.set("Female")
                    widget.grid(row=(i if i<half else i-half), column=1, padx=6, pady=4, sticky="w")
                else:
                    widget = ctk.StringVar(value="Female")
                    combo = tk.OptionMenu(target, widget, "Female", "Male")
                    combo.grid(row=(i if i<half else i-half), column=1, padx=6, pady=4, sticky="w")
                    widget = widget  # store the StringVar
            elif param == "APOE4":
                if scrollable_available:
                    widget = ctk.CTkComboBox(target, values=["0", "1", "2"], width=100)
                    widget.set("0")
                    widget.grid(row=(i if i<half else i-half), column=1, padx=6, pady=4, sticky="w")
                else:
                    widget = ctk.StringVar(value="0")
                    combo = tk.OptionMenu(target, widget, "0", "1", "2")
                    combo.grid(row=(i if i<half else i-half), column=1, padx=6, pady=4, sticky="w")
                    widget = widget
            else:
                if scrollable_available:
                    widget = ctk.CTkEntry(target, width=160)
                    widget.grid(row=(i if i<half else i-half), column=1, padx=6, pady=4, sticky="w")
                else:
                    widget = tk.Entry(target, width=22)
                    widget.grid(row=(i if i<half else i-half), column=1, padx=6, pady=4, sticky="w")

            self.inputs[param] = widget

        actions = ctk.CTkFrame(self.input_frame, fg_color="transparent")
        actions.pack(pady=(12,6))

        predict_btn = ctk.CTkButton(actions, text="Predict", command=self.on_predict)
        predict_btn.grid(row=0, column=0, padx=8)
        back_selection_btn = ctk.CTkButton(actions, text="Back to Selection", command=self.on_back_to_selection)
        back_selection_btn.grid(row=0, column=1, padx=8)

        self.prediction_label = ctk.CTkLabel(self.input_frame, text="", font=("Helvetica", 14, "bold"), text_color="red")
        self.prediction_label.pack(pady=(8,6))

        confirm_frame = ctk.CTkFrame(self.input_frame, fg_color="transparent")
        confirm_frame.pack(pady=(6,8))

        confirm_btn = ctk.CTkButton(confirm_frame, text="Confirm Diagnosis", command=self.on_confirm)
        confirm_btn.grid(row=0, column=0, padx=8)

        contest_btn = ctk.CTkButton(confirm_frame, text="Contest Diagnosis", command=self.on_contest_toggle)
        contest_btn.grid(row=0, column=1, padx=8)

        undo_btn = ctk.CTkButton(confirm_frame, text="Undo Last", command=self.on_undo_last)
        undo_btn.grid(row=0, column=3, padx=8)

        self.contest_choice = ctk.CTkComboBox(confirm_frame, values=["CN","EMCI","LMCI","AD"], width=140)
        self.contest_choice.grid(row=0, column=2, padx=8)
        self.contest_choice.set("")
        self.contest_choice.grid_remove()

        self.status_label = ctk.CTkLabel(self.input_frame, text="", anchor="w")
        self.status_label.pack(fill="x", padx=8, pady=(6,4))

    def on_back_to_selection(self):
        """Return to model selection page (destroy input frame)."""
        if self.input_frame:
            try:
                self.input_frame.pack_forget()
                self.input_frame.destroy()
            except Exception:
                pass
            self.input_frame = None

        if self.selection_frame:
            try:
                self.selection_frame.pack_forget()
                self.selection_frame.destroy()
            except Exception:
                pass
            self.selection_frame = None
        self.build_selection_page()

    def _create_derived_features(self, parsed_raw):
        """
        Given parsed_raw (values for ALL_PARAMS), compute derived features required by models and CSV:
        - TAU/ABETA
        - PTAU/ABETA
        - hippocampus/ICV, entorhinal/ICV, fusiform/ICV, midtemp/ICV, ventricles/ICV, wholebrain/ICV

        Returns a dict keyed by the feature columns used in COLUMN_ORDER[1:].
        """
        row = {}

        # start with all feature columns initialized as NaN
        for col in COLUMN_ORDER[1:]:
            row[col] = np.nan

        # copy direct mappings for those features that exist in parsed_raw
        for k, v in parsed_raw.items():
            if k in row:
                row[k] = v
        # compute ratios for biomarkers
        TAU = parsed_raw.get("TAU", np.nan)
        PTAU = parsed_raw.get("PTAU", np.nan)
        ABETA = parsed_raw.get("ABETA", np.nan)

        # TAU/ABETA
        try:
            if pd.isna(TAU) or pd.isna(ABETA) or float(ABETA) == 0.0:
                row["TAU/ABETA"] = np.nan
            else:
                row["TAU/ABETA"] = float(TAU) / float(ABETA)
        except Exception:
            row["TAU/ABETA"] = np.nan

        # PTAU/ABETA
        try:
            if pd.isna(PTAU) or pd.isna(ABETA) or float(ABETA) == 0.0:
                row["PTAU/ABETA"] = np.nan
            else:
                row["PTAU/ABETA"] = float(PTAU) / float(ABETA)
        except Exception:
            row["PTAU/ABETA"] = np.nan

        # structural normalizations /ICV
        ICV = parsed_raw.get("ICV", np.nan)
        struct_names = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
        for s in struct_names:
            colname = f"{s}/ICV"
            val = parsed_raw.get(s, np.nan)
            try:
                if pd.isna(val) or pd.isna(ICV) or float(ICV) == 0.0:
                    row[colname] = np.nan
                else:
                    row[colname] = float(val) / float(ICV)
            except Exception:
                row[colname] = np.nan

        return row

    def on_predict(self):
        """Validate inputs, prepare X, compute derived features, possibly drop columns, and predict using model."""
        valid, parsed = self.validate_and_parse_inputs()
        if not valid:
            return

        # Build features dict with derived columns
        derived_row = self._create_derived_features(parsed)

        # The model expects the same order as COLUMN_ORDER[1:]
        feature_columns = COLUMN_ORDER[1:]
        X_df = pd.DataFrame([derived_row], columns=feature_columns)

        # Drop columns for AltModel/AltXAIModel
        if self.model_choice in ("AltModel.pkl", "AltXAIModel.pkl"):
            X_for_model = X_df.drop(columns=[c for c in REMOVE_FOR_ALT_MODEL if c in X_df.columns], errors='ignore')
        else:
            X_for_model = X_df

        try:
            label = self.model_loader.predict(X_for_model)
        except Exception as e:
            messagebox.showerror("Prediction error", f"Prediction error:\n{e}")
            return

        # Label is a string like "CN","EMCI","LMCI","AD"
        diag_text = f"{label} [{DIAG_MAP.get(label, 'Unknown')}]"
        self.prediction_label.configure(text=diag_text)
        self.last_parsed = parsed  # Raw parsed values
        self.last_prediction_label = label
        self.last_feature_row = derived_row  # Store derived row ready for saving
        self.status_label.configure(text=f"Model {self.model_choice} predicted: {diag_text}")

    def on_confirm(self):
        """Save the last prediction into the NEWADNIMERGE.csv (DX as string label)."""
        if self.last_prediction_label is None:
            messagebox.showerror("Confirm Diagnosis", "No prediction available. Please click Predict first.")
            return
        dx_to_save = self.last_prediction_label  # Already a string label
        try:
            self.append_row_to_csv(self.last_feature_row, dx_to_save)
        except Exception as e:
            messagebox.showerror("Save error", f"Error saving data:\n{e}")
            return
        messagebox.showinfo("Saved", "Data saved to NEWADNIMERGE.csv (confirmed diagnosis).")
        try:
            df = pd.read_csv(NEW_CSV)
            self.last_saved_row_index = len(df) - 1
        except Exception:
            self.last_saved_row_index = None
        self.prediction_label.configure(text="")
        self.status_label.configure(text="Saved confirmed diagnosis.")
        # Reset last prediction
        self.last_prediction_label = None
        self.last_parsed = None
        self.last_feature_row = None

    def on_contest_toggle(self):
        """Show/hide contest dropdown and handle saving if a choice is selected."""
        if self.contest_choice.winfo_viewable():
            chosen = self.contest_choice.get()
            if not chosen:
                self.contest_choice.grid_remove()
                return
            dx_to_save = chosen  # Store string label directly
            if self.last_feature_row is None:
                valid, parsed = self.validate_and_parse_inputs()
                if not valid:
                    return
                derived = self._create_derived_features(parsed)
                try:
                    self.append_row_to_csv(derived, dx_to_save)
                except Exception as e:
                    messagebox.showerror("Save error", f"Error saving data:\n{e}")
                    return
            else:
                try:
                    self.append_row_to_csv(self.last_feature_row, dx_to_save)
                except Exception as e:
                    messagebox.showerror("Save error", f"Error saving data:\n{e}")
                    return
            messagebox.showinfo("Saved", "Data saved to NEWADNIMERGE.csv (contested diagnosis recorded).")
            try:
                df = pd.read_csv(NEW_CSV)
                self.last_saved_row_index = len(df) - 1
            except Exception:
                self.last_saved_row_index = None
            self.contest_choice.set("")
            self.contest_choice.grid_remove()
            self.prediction_label.configure(text="")
            self.status_label.configure(text="Saved contested diagnosis.")
            self.last_prediction_label = None
            self.last_parsed = None
            self.last_feature_row = None
        else:
            self.contest_choice.grid()

    def on_undo_last(self):
        """Undo the most recent saved row, without touching older rows."""
        if self.last_saved_row_index is None:
            messagebox.showinfo("Undo", "No recent entry to undo.")
            return
        if not os.path.exists(NEW_CSV):
            messagebox.showerror("Undo", "No dataset file found.")
            return

        try:
            df = pd.read_csv(NEW_CSV)
            if df.empty:
                messagebox.showinfo("Undo", "The dataset is already empty.")
                return
            if len(df) - 1 != self.last_saved_row_index:
                messagebox.showinfo("Undo", "Cannot undo: another entry was added afterwards.")
                return
            df = df.iloc[:-1, :]
            df.to_csv(NEW_CSV, index=False)
            self.last_saved_row_index = None
            self.status_label.configure(text="Last entry undone.")
            messagebox.showinfo("Undo", "The last entry was undone successfully.")
        except Exception as e:
            messagebox.showerror("Undo error", f"Error while undoing:\n{e}")

    def validate_and_parse_inputs(self):
        """
        Validate all inputs based on VALIDATION rules.
        Returns (True, parsed_dict) or (False, None) and shows popup on error.
        """
        parsed = {}
        errors = []

        for param in ALL_PARAMS:
            widget = self.inputs.get(param)
            if widget is None:
                errors.append(f"{param}: widget missing.")
                continue
            raw = None
            try:
                raw = widget.get()
            except Exception:
                try:
                    raw = widget.get()
                except Exception:
                    raw = None

            if raw is None or (isinstance(raw, str) and raw.strip() == ""):
                errors.append(f"{param}: no value provided.")
                continue

            if param == "PTGENDER":
                if raw not in ("Female", "Male"):
                    errors.append("PTGENDER must be 'Female' or 'Male'.")
                else:
                    parsed[param] = 1 if raw == "Male" else 0
                continue

            if param == "APOE4":
                try:
                    a = int(raw)
                except Exception:
                    errors.append("APOE4 must be 0, 1 or 2.")
                    continue
                if a not in (0,1,2):
                    errors.append("APOE4 must be 0, 1 or 2.")
                    continue
                parsed[param] = int(a)
                continue

            # Otherwise numeric
            try:
                val = float(raw)
            except Exception:
                errors.append(f"{param}: invalid numeric value.")
                continue

            if param in VALIDATION:
                minv, maxv, expected_type = VALIDATION[param]
                if val < minv or val > maxv:
                    errors.append(f"{param}: value {val} outside allowed range [{minv}, {maxv}].")
                    continue
                if expected_type is int:
                    parsed[param] = int(val)
                else:
                    parsed[param] = float(val)
            else:
                parsed[param] = float(val)

        if errors:
            messagebox.showerror("Input validation errors", "\n".join(errors))
            return False, None
        return True, parsed

    def append_row_to_csv(self, feature_row_dict, dx_value):
        """
        Append a single row to NEWADNIMERGE.csv.
        feature_row_dict: dict keyed by feature column names (columns after DX)
        dx_value: string label ("CN","EMCI","LMCI","AD")
        """
        os.makedirs(DATA_DIR, exist_ok=True)

        full_row = {}
        for col in COLUMN_ORDER:
            if col == "DX":
                full_row["DX"] = str(dx_value)
            else:
                v = feature_row_dict.get(col, np.nan)
                # Ensure PTGENDER/APOE4 stored as ints if present
                if col == "PTGENDER":
                    v = int(v) if not (pd.isna(v)) else v
                if col == "APOE4":
                    v = int(v) if not (pd.isna(v)) else v
                full_row[col] = v

        df_row = pd.DataFrame([full_row], columns=COLUMN_ORDER)
        if os.path.exists(NEW_CSV):
            df_row.to_csv(NEW_CSV, mode='a', header=False, index=False)
        else:
            df_row.to_csv(NEW_CSV, mode='w', header=True, index=False)

# ----------------------------#
#            MAIN             #
# ----------------------------#
def main():
    # Check results dir presence (non-blocking)
    if not os.path.exists(RESULTS_DIR):
        # Show a simple info message when starting the app
        print(f"Warning: results directory '{RESULTS_DIR}' does not exist. Please create it and add your .pkl models.")
    root = ctk.CTk()
    app = PageManager(root)
    root.mainloop()

if __name__ == "__main__":
    main()
