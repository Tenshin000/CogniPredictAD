"""
main.py

Medical prediction GUI using customtkinter.

- Comments and printed outputs are in English.
- UI flow:
  * Blue start page with "Start"
  * White selection page to choose one of the 4 models (Model1/Model2/XAIModel1/XAIModel2)
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
ctk.set_appearance_mode("System")   # "System", "Dark", "Light"
ctk.set_default_color_theme("blue")  # can be changed to another built-in theme

RESULTS_DIR = "results"
DATA_DIR = "data"
NEW_CSV = os.path.join(DATA_DIR, "NEWADNIMERGE.csv")

# Column names and order based on your provided snippet
COLUMN_ORDER = [
    "DX", "AGE", "PTGENDER", "PTEDUCAT", "APOE4", "MMSE", "CDRSB", "ADAS13",
    "LDELTOTAL", "FAQ", "MOCA", "TRABSCOR", "RAVLT_immediate", "RAVLT_learning",
    "RAVLT_perc_forgetting", "mPACCdigit", "EcogPtMem", "EcogPtLang", "EcogPtVisspat",
    "EcogPtPlan", "EcogPtOrgan", "EcogPtDivatt", "EcogSPMem", "EcogSPLang",
    "EcogSPVisspat", "EcogSPPlan", "EcogSPOrgan", "EcogSPDivatt", "FDG",
    "PTAU/ABETA", "Hippocampus/ICV", "Entorhinal/ICV", "Fusiform/ICV",
    "MidTemp/ICV", "Ventricles/ICV", "WholeBrain/ICV"
]

# All parameters to ask the user (in the UI). They match the dataset columns except DX.
ALL_PARAMS = [
    "AGE", "PTGENDER", "PTEDUCAT", "APOE4", "MMSE", "CDRSB", "ADAS13",
    "LDELTOTAL", "FAQ", "MOCA", "TRABSCOR", "RAVLT_immediate", "RAVLT_learning",
    "RAVLT_perc_forgetting", "mPACCdigit", "EcogPtMem", "EcogPtLang", "EcogPtVisspat",
    "EcogPtPlan", "EcogPtOrgan", "EcogPtDivatt", "EcogSPMem", "EcogSPLang",
    "EcogSPVisspat", "EcogSPPlan", "EcogSPOrgan", "EcogSPDivatt", "FDG",
    "PTAU/ABETA", "Hippocampus/ICV", "Entorhinal/ICV", "Fusiform/ICV",
    "MidTemp/ICV", "Ventricles/ICV", "WholeBrain/ICV"
]

# The three features removed for Model2/XAIModel2 (they remain in the CSV though)
REMOVE_FOR_MODEL2 = ["CDRSB", "LDELTOTAL", "mPACCdigit"]

# type/range validation map (min, max, type)
VALIDATION = {
    "AGE": (0, 120, int),
    "APOE4": (0, 2, int),
    "MMSE": (0, 30, int),
    "LDELTOTAL": (0, 30, int),
    "FAQ": (0, 30, int),
    "MOCA": (0, 30, int),
    # NOTE: you previously set CDRSB 0..18; keep that if desired (was 0..10 initially)
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
}

# diagnosis mapping
DIAG_MAP = {
    0: "Cognitively Normal (CN)",
    1: "Early Mild Cognitive Impairment (EMCI)",
    2: "Late Mild Cognitive Impairment (LMCI)",
    3: "Alzheimer's Disease (AD)"
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
        Returns integer label (0,1,2,3).
        """
        if self.model is None:
            self.load()
        preds = self.model.predict(X_df)
        if isinstance(preds, (list, tuple, np.ndarray, pd.Series)):
            label = int(np.asarray(preds).ravel()[0])
        else:
            label = int(preds)
        return label


class PageManager:
    """Class to manage customtkinter pages and flows."""
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.title("Medical Classifier")
        self.root.geometry("1000x760")
        # State variables
        self.model_choice = None
        self.model_loader = None
        self.model_descs = {
            "Model1.pkl": "Description for Model1 (edit later).",
            "Model2.pkl": "Description for Model2 (edit later).",
            "XAIModel1.pkl": "Description for XAIModel1 (edit later).",
            "XAIModel2.pkl": "Description for XAIModel2 (edit later)."
        }
        self.inputs = {}  # param -> widget
        self.last_parsed = None
        self.last_prediction_label = None
        self.last_feature_row = None

        # Frames for pages
        self.start_frame = None
        self.selection_frame = None
        self.input_frame = None

        # Build UI
        self.build_start_page()
    

    def build_start_page(self):
        """Blue start page with Start button."""
        if self.selection_frame:
            self.selection_frame.pack_forget()
        if self.input_frame:
            self.input_frame.pack_forget()

        self.start_frame = ctk.CTkFrame(self.root, fg_color="#87CEEB")
        self.start_frame.pack(fill="both", expand=True)

        spacer = ctk.CTkLabel(self.start_frame, text="")
        spacer.pack(pady=80)

        start_btn = ctk.CTkButton(self.start_frame, text="Start", width=160, height=48,
                                  command=self.on_start_pressed)
        start_btn.pack(pady=20)


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
        models = ["Model1.pkl", "Model2.pkl", "XAIModel1.pkl", "XAIModel2.pkl"]
        radios_frame = ctk.CTkFrame(self.selection_frame, fg_color="transparent")
        radios_frame.pack(pady=(4,8))
        for m in models:
            r = ctk.CTkRadioButton(radios_frame, text=m, variable=self.model_var, value=m)
            r.pack(anchor="w", padx=8, pady=2)

        # Multi-line description (editable)
        desc_label = ctk.CTkLabel(self.selection_frame, text="Model descriptions (editable):", anchor="w")
        desc_label.pack(fill="x", padx=8, pady=(10,0))
        self.desc_text = ctk.CTkTextbox(self.selection_frame, width=940, height=160)
        # populate with combined descriptions
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

        # placeholder area for inputs (we will pack input_frame when ready)
        placeholder = ctk.CTkLabel(self.selection_frame, text="Model not loaded yet.", fg_color="transparent")
        placeholder.pack(pady=20)


    def on_back_to_start(self):
        """Back to start page"""
        # save description edits back to self.model_descs ? keep for now
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
        # Update model descriptions from textbox (user might have edited)
        # Keep current text but we don't parse it into model_descs structure
        self.model_choice = chosen
        model_path = os.path.join(RESULTS_DIR, chosen)
        try:
            self.model_loader = ModelLoader(model_path)
            self.model_loader.load()
        except Exception as e:
            messagebox.showerror("Load Model", f"Error loading model:\n{e}")
            return

        # Hide the entire selection frame so it disappears
        if self.selection_frame:
            self.selection_frame.pack_forget()

        # Build input form (this will create and pack input_frame)
        self.build_input_form()


    def build_input_form(self):
        """Build the long input form and action buttons with scroll support."""
        # Create input_frame (ensure any previous one is destroyed)
        if hasattr(self, "input_frame") and self.input_frame:
            try:
                self.input_frame.destroy()
            except Exception:
                pass
        self.input_frame = ctk.CTkFrame(self.root)
        self.input_frame.pack(fill="both", expand=True, padx=12, pady=12)

        header = ctk.CTkLabel(self.input_frame, text=f"Inputs for {self.model_choice}", font=("Helvetica", 16, "bold"))
        header.pack(pady=(4,6))

        # Try to use CTkScrollableFrame if available
        scrollable_available = hasattr(ctk, "CTkScrollableFrame")
        if scrollable_available:
            # Create CTkScrollableFrame (modern and simple)
            scroll_container = ctk.CTkScrollableFrame(self.input_frame, width=960, height=520)
            scroll_container.pack(padx=6, pady=6, fill="both", expand=False)
            content_parent = scroll_container
        else:
            # Fallback implementation using tkinter Canvas + Frame + scrollbar
            outer = tk.Frame(self.input_frame)
            outer.pack(fill="both", expand=True)
            canvas = tk.Canvas(outer, width=960, height=520)
            vscroll = tk.Scrollbar(outer, orient="vertical", command=canvas.yview)
            canvas.configure(yscrollcommand=vscroll.set)
            vscroll.pack(side="right", fill="y")
            canvas.pack(side="left", fill="both", expand=True)
            inner_frame = tk.Frame(canvas)
            # make tkinter.Frame compatible with CTk geometry (we will embed CTk widgets on tk.Frame by using ctks inside it)
            canvas.create_window((0, 0), window=inner_frame, anchor='nw')

            # update scrollregion when inner_frame changes
            def _on_frame_config(event, c=canvas):
                c.configure(scrollregion=c.bbox("all"))
            inner_frame.bind("<Configure>", _on_frame_config)
            content_parent = inner_frame

        # Build two-column layout inside content_parent
        left_frame = ctk.CTkFrame(content_parent, fg_color="transparent") if scrollable_available else tk.Frame(content_parent)
        right_frame = ctk.CTkFrame(content_parent, fg_color="transparent") if scrollable_available else tk.Frame(content_parent)

        if scrollable_available:
            # If using CTkScrollableFrame, we can grid CTkFrames inside it
            left_frame.grid(row=0, column=0, padx=(6,12), pady=6, sticky="n")
            right_frame.grid(row=0, column=1, padx=(12,6), pady=6, sticky="n")
        else:
            # Using tk fallback, pack left and right frames side by side
            left_frame.pack(side="left", padx=(6,12), pady=6, anchor="n")
            right_frame.pack(side="left", padx=(12,6), pady=6, anchor="n")

        # populate inputs into left/right
        half = (len(ALL_PARAMS) + 1) // 2
        for i, param in enumerate(ALL_PARAMS):
            target = left_frame if i < half else right_frame
            # create label
            if scrollable_available:
                label = ctk.CTkLabel(target, text=param, anchor="w")
                label.grid(row=(i if i<half else i-half), column=0, padx=6, pady=4, sticky="w")
            else:
                label = ctk.Label(target, text=param, anchor="w")
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
                    widget = widget  # store the StringVar (we will call .get() on it in validate)
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

        # Actions frame (pack below the scroll area)
        actions = ctk.CTkFrame(self.input_frame, fg_color="transparent")
        actions.pack(pady=(12,6))

        predict_btn = ctk.CTkButton(actions, text="Predict", command=self.on_predict)
        predict_btn.grid(row=0, column=0, padx=8)
        back_selection_btn = ctk.CTkButton(actions, text="Back to Selection", command=self.on_back_to_selection)
        back_selection_btn.grid(row=0, column=1, padx=8)

        # Prediction display
        self.prediction_label = ctk.CTkLabel(self.input_frame, text="", font=("Helvetica", 14, "bold"), text_color="red")
        self.prediction_label.pack(pady=(8,6))

        # Confirm / Contest buttons
        confirm_frame = ctk.CTkFrame(self.input_frame, fg_color="transparent")
        confirm_frame.pack(pady=(6,8))

        confirm_btn = ctk.CTkButton(confirm_frame, text="Confirm Diagnosis", command=self.on_confirm)
        confirm_btn.grid(row=0, column=0, padx=8)
        contest_btn = ctk.CTkButton(confirm_frame, text="Contest Diagnosis", command=self.on_contest_toggle)
        contest_btn.grid(row=0, column=1, padx=8)

        # Contest dropdown (hidden initially)
        self.contest_choice = ctk.CTkComboBox(confirm_frame, values=["CN","EMCI","LMCI","AD"], width=140)
        self.contest_choice.grid(row=0, column=2, padx=8)
        self.contest_choice.set("")
        self.contest_choice.grid_remove()

        # Status label
        self.status_label = ctk.CTkLabel(self.input_frame, text="", anchor="w")
        self.status_label.pack(fill="x", padx=8, pady=(6,4))


    def on_back_to_selection(self):
        """Return to model selection page (destroy input frame)."""
        # destroy input frame
        if self.input_frame:
            try:
                self.input_frame.pack_forget()
                self.input_frame.destroy()
            except Exception:
                pass
            self.input_frame = None

        # show selection frame again: destroy and rebuild to get a clean state
        if self.selection_frame:
            try:
                self.selection_frame.pack_forget()
                self.selection_frame.destroy()
            except Exception:
                pass
            self.selection_frame = None
        self.build_selection_page()


    def on_predict(self):
        """Validate inputs, prepare X, possibly drop columns, and predict using model."""
        valid, parsed = self.validate_and_parse_inputs()
        if not valid:
            return
        # Build feature row following COLUMN_ORDER[1:]
        feature_columns = COLUMN_ORDER[1:]
        row_dict = {}
        for col in feature_columns:
            if col in parsed:
                row_dict[col] = parsed[col]
            else:
                row_dict[col] = np.nan

        X_df = pd.DataFrame([row_dict], columns=feature_columns)

        # Drop columns for Model2/XAIModel2
        if self.model_choice in ("Model2.pkl", "XAIModel2.pkl"):
            X_for_model = X_df.drop(columns=[c for c in REMOVE_FOR_MODEL2 if c in X_df.columns], errors='ignore')
        else:
            X_for_model = X_df

        try:
            label = self.model_loader.predict(X_for_model)
        except Exception as e:
            messagebox.showerror("Prediction error", f"Prediction error:\n{e}")
            return

        diag_text = f"{label} [{DIAG_MAP.get(label, 'Unknown')}]"
        self.prediction_label.configure(text=diag_text)
        self.last_parsed = parsed
        self.last_prediction_label = label
        self.last_feature_row = row_dict
        self.status_label.configure(text=f"Model {self.model_choice} predicted: {diag_text}")

    def on_confirm(self):
        """Save the last prediction into the NEWADNIMERGE.csv (DX from model)."""
        if self.last_prediction_label is None:
            messagebox.showerror("Confirm Diagnosis", "No prediction available. Please click Predict first.")
            return
        dx_to_save = int(self.last_prediction_label)
        try:
            self.append_row_to_csv(self.last_feature_row, dx_to_save)
        except Exception as e:
            messagebox.showerror("Save error", f"Error saving data:\n{e}")
            return
        messagebox.showinfo("Saved", "Data saved to NEWADNIMERGE.csv (confirmed diagnosis).")
        self.prediction_label.configure(text="")
        self.status_label.configure(text="Saved confirmed diagnosis.")
        # reset last prediction
        self.last_prediction_label = None
        self.last_parsed = None
        self.last_feature_row = None


    def on_contest_toggle(self):
        """Show/hide contest dropdown and handle saving if a choice is selected."""
        if self.contest_choice.winfo_viewable():
            # If visible and selection present, save
            chosen = self.contest_choice.get()
            if not chosen:
                # hide it if empty
                self.contest_choice.grid_remove()
                return
            mapping = {"CN":0, "EMCI":1, "LMCI":2, "AD":3}
            dx_to_save = mapping.get(chosen)
            # If we don't have a parsed set yet, validate now
            if self.last_feature_row is None:
                valid, parsed = self.validate_and_parse_inputs()
                if not valid:
                    return
                row_dict = {col: parsed.get(col, np.nan) for col in COLUMN_ORDER[1:]}
                try:
                    self.append_row_to_csv(row_dict, dx_to_save)
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
            self.contest_choice.set("")
            self.contest_choice.grid_remove()
            self.prediction_label.configure(text="")
            self.status_label.configure(text="Saved contested diagnosis.")
            self.last_prediction_label = None
            self.last_parsed = None
            self.last_feature_row = None
        else:
            # show it
            self.contest_choice.grid()


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
            # Extract raw value depending on widget type
            raw = None
            # CTkComboBox has .get(), CTkEntry has .get()
            try:
                raw = widget.get()
            except Exception:
                # fallback: attempt to read .get() anyway
                try:
                    raw = widget.get()
                except Exception:
                    raw = None

            # Empty string check
            if raw is None or (isinstance(raw, str) and raw.strip() == ""):
                errors.append(f"{param}: no value provided.")
                continue

            # PTGENDER and APOE4 special handling
            if param == "PTGENDER":
                if raw not in ("Female", "Male"):
                    errors.append("PTGENDER must be 'Female' or 'Male'.")
                else:
                    parsed[param] = 1 if raw == "Male" else 0
                continue

            if param == "APOE4":
                # combobox might return int or string; force int
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
                # no rule -> keep float
                parsed[param] = float(val)

        if errors:
            messagebox.showerror("Input validation errors", "\n".join(errors))
            return False, None
        return True, parsed

    
    def append_row_to_csv(self, feature_row_dict, dx_value):
        """
        Append a single row to NEWADNIMERGE.csv.
        feature_row_dict: dict keyed by feature column names (columns after DX)
        dx_value: integer label
        """
        os.makedirs(DATA_DIR, exist_ok=True)

        full_row = {}
        for col in COLUMN_ORDER:
            if col == "DX":
                full_row["DX"] = int(dx_value)
            else:
                v = feature_row_dict.get(col, np.nan)
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
    # check results dir presence (non-blocking)
    if not os.path.exists(RESULTS_DIR):
        # show a simple info message when starting the app
        print(f"Warning: results directory '{RESULTS_DIR}' does not exist. Please create it and add your .pkl models.")
    root = ctk.CTk()
    app = PageManager(root)
    root.mainloop()


if __name__ == "__main__":
    main()
