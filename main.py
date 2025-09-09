"""
Medical prediction GUI using PySimpleGUI.

- Comments and printed outputs are in English as requested.
- UI flow:
  * Blue start page with "Start"
  * White selection page to choose one of the 4 models (Model1/Model2/XAIModel1/XAIModel2)
  * After selection, descriptions disappear and the input fields appear
  * Prediction is displayed in red; user can Confirm or Contest diagnosis
  * Confirm/Contest appends a row to data/NEWADNIMERGE.csv (creates folder/file if needed)

Author: Francesco Panattoni
"""

import os
import joblib
import pandas as pd
import numpy as np
import PySimpleGUI as sg


# ----------------------------#
#       CONFIGURATIONS        #
# ----------------------------# 
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
        # joblib is robust for scikit-learn pickles
        self.model = joblib.load(self.model_path)
        # simple check
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
        # Ensure shape is (1, n_features)
        # Convert to numpy array for safety
        preds = self.model.predict(X_df)
        # If predict returns array-like
        if isinstance(preds, (list, tuple, np.ndarray, pd.Series)):
            label = int(np.asarray(preds).ravel()[0])
        else:
            # maybe returns scalar
            label = int(preds)
        return label


class PageManager:
    """Class to manage PySimpleGUI pages and flows."""
    def __init__(self):
        sg.theme("LightBlue")  # default theme; we'll set colors per layout
        self.window = None
        self.model_choice = None
        self.model_loader = None
        self.model_descs = {
            "Model1.pkl": "Description for Model1 (edit later).",
            "Model2.pkl": "Description for Model2 (edit later).",
            "XAIModel1.pkl": "Description for XAIModel1 (edit later).",
            "XAIModel2.pkl": "Description for XAIModel2 (edit later)."
        }
        # prepare elements dictionary for easy update
        self.input_elements = {}

    def start(self):
        """Start the GUI main loop."""
        self.show_start_page()

    # LAYOUTS
    def show_start_page(self):
        """Blue start page with Start button."""
        layout = [
            [sg.Text("", size=(40, 3))],
            [sg.Push(), sg.Button("Start", size=(12,2), button_color=('white','dark blue')), sg.Push()]
        ]
        self.window = sg.Window("Medical Classifier", layout, background_color="#87CEEB", finalize=True, element_justification='center')
        # event loop for start page
        while True:
            event, values = self.window.read()
            if event == sg.WIN_CLOSED:
                self.window.close()
                return
            if event == "Start":
                # move to selection page
                self.window.close()
                self.show_selection_page()
                return

    def show_selection_page(self):
        """White page to select the model and show descriptions. After selection show inputs."""
        sg.theme("DefaultNoMoreNagging")
        # radio buttons for models
        models = ["Model1.pkl", "Model2.pkl", "XAIModel1.pkl", "XAIModel2.pkl"]
        radio_row = [sg.Radio(m, "MODEL", key=f"MODEL_{m}", default=(i==0)) for i,m in enumerate(models)]
        desc_area = sg.Multiline(self.get_combined_descriptions(), size=(80,10), key="DESC_AREA")
        # instructions
        instr = sg.Text("Select a model and then click 'Load Model' to continue.", size=(60,1))
        load_button = sg.Button("Load Model")
        # placeholder for inputs area (will be replaced after model load)
        inputs_col = sg.Column([[sg.Text("Model not loaded yet.", key="INPUT_PLACEHOLDER")]], key="INPUT_COL", visible=True)
        layout = [
            [sg.Text("Model selection", font=("Any", 14, "bold"))],
            radio_row,
            [instr],
            [desc_area],
            [load_button],
            [sg.HorizontalSeparator()],
            [inputs_col],
            [sg.Button("Back"), sg.Button("Exit")]
        ]
        self.window = sg.Window("Model Selection", layout, finalize=True, size=(900,700), element_justification='left')
        # event loop
        while True:
            event, values = self.window.read()
            if event in (sg.WIN_CLOSED, "Exit"):
                self.window.close()
                return
            if event == "Back":
                self.window.close()
                self.show_start_page()
                return
            if event == "Load Model":
                # detect selected model
                chosen = None
                for m in models:
                    if values.get(f"MODEL_{m}"):
                        chosen = m
                        break
                if chosen is None:
                    sg.popup_error("Please select a model.")
                    continue
                self.model_choice = chosen
                # optionally allow user to edit descriptions before they disappear (we keep them editable)
                # remove description area and show inputs
                self.window["DESC_AREA"].update(visible=False)
                self.window["INPUT_PLACEHOLDER"].update(visible=False)
                # load model file from results folder
                model_path = os.path.join(RESULTS_DIR, chosen)
                try:
                    self.model_loader = ModelLoader(model_path)
                    self.model_loader.load()
                except Exception as e:
                    sg.popup_error(f"Error loading model: {e}")
                    # restore descriptions visible in case of error
                    self.window["DESC_AREA"].update(visible=True)
                    self.window["INPUT_PLACEHOLDER"].update(visible=True)
                    continue
                # show input form
                self.build_input_form()
            # additional events handled by input form (they will bubble up here)

    def get_combined_descriptions(self):
        """Return a combined description string for the four models."""
        lines = []
        for k, v in self.model_descs.items():
            lines.append(f"{k}:\n{v}\n")
        return "\n".join(lines)

    def build_input_form(self):
        """Construct the inputs form in the window (replaces INPUT_COL)."""
        # Build rows: for each param create Label + Input
        # For PTGENDER use Combo [Female, Male] -> will map Female->0, Male->1
        # For APOE4 use Combo 0,1,2
        form_rows = []
        self.input_elements = {}

        # Two-column layout for compactness
        left_cols = []
        right_cols = []
        half = (len(ALL_PARAMS) + 1) // 2
        for i, param in enumerate(ALL_PARAMS):
            if param == "PTGENDER":
                elem = sg.Combo(["Female", "Male"], key=f"IN_{param}", default_value="Female", readonly=True, size=(12,1))
            elif param == "APOE4":
                elem = sg.Combo([0,1,2], key=f"IN_{param}", default_value=0, readonly=True, size=(6,1))
            else:
                # numeric input: allow text input but will validate later
                elem = sg.Input(key=f"IN_{param}", size=(12,1))
            # save element key
            self.input_elements[param] = f"IN_{param}"
            row = [sg.Text(param, size=(18,1)), elem]
            if i < half:
                left_cols.append(row)
            else:
                right_cols.append(row)

        # Compose column layout
        left_col = [[r[0], r[1]] for r in left_cols]
        right_col = [[r[0], r[1]] for r in right_cols]

        inputs_layout = [
            [sg.Column(left_col, vertical_alignment='top'),
             sg.VerticalSeparator(),
             sg.Column(right_col, vertical_alignment='top')]
        ]

        # buttons for prediction and actions
        actions = [sg.Button("Predict"), sg.Button("Back to Selection")]
        result_row = [sg.Text("Prediction: ", size=(10,1)), sg.Text("", key="PREDICTION_TEXT", text_color="red", font=("Any", 12, "bold"))]
        confirm_buttons = [sg.Button("Confirm Diagnosis"), sg.Button("Contest Diagnosis")]
        # Contest choice element (hidden until Contest pressed)
        contest_choice = sg.Combo(["CN","EMCI","LMCI","AD"], key="CONTEST_CHOICE", visible=False, readonly=True)
        # Put everything into INPUT_COL by updating it
        new_col_layout = [
            [sg.Frame("Inputs (fill all fields)", inputs_layout, relief=sg.RELIEF_SUNKEN)],
            [sg.HorizontalSeparator()],
            result_row,
            [sg.Column([actions + confirm_buttons + [contest_choice]])],
            [sg.Text("", key="STATUS", size=(80,2))]
        ]

        self.window["INPUT_COL"].update([[sg.Column(new_col_layout)]], visible=True)
        # Resize window to fit new layout
        self.window.refresh()

        # Now update event loop to handle input events
        while True:
            event, values = self.window.read()
            if event in (sg.WIN_CLOSED, "Exit"):
                self.window.close()
                return
            if event == "Back to Selection":
                # close and go back to model selection
                self.window.close()
                self.show_selection_page()
                return
            if event == "Predict":
                # Validate inputs and run prediction
                valid, parsed = self.validate_and_parse_inputs(values)
                if not valid:
                    # validate_and_parse_inputs shows popup with details
                    continue
                # build DataFrame row according to COLUMN_ORDER (excluding DX)
                # DX is not fed into model; prepare feature vector for model: columns after DX
                feature_columns = COLUMN_ORDER[1:]  # everything except DX
                # build base row dict
                row_dict = {}
                for col in feature_columns:
                    # Note: parsed contains user values for ALL_PARAMS only; some columns like DX not present
                    if col in parsed:
                        row_dict[col] = parsed[col]
                    else:
                        # if some columns not present (shouldn't happen) set NaN
                        row_dict[col] = np.nan

                X_df = pd.DataFrame([row_dict], columns=feature_columns)

                # if model is Model2 or XAIModel2, drop the specified columns before passing to model
                if self.model_choice in ("Model2.pkl", "XAIModel2.pkl"):
                    X_for_model = X_df.drop(columns=[c for c in REMOVE_FOR_MODEL2 if c in X_df.columns], errors='ignore')
                else:
                    X_for_model = X_df

                # predict
                try:
                    label = self.model_loader.predict(X_for_model)
                except Exception as e:
                    sg.popup_error(f"Prediction error: {e}")
                    continue

                # Display prediction in red
                diag_text = f"{label} [{DIAG_MAP.get(label, 'Unknown')}]"
                self.window["PREDICTION_TEXT"].update(diag_text)
                # store last parsed and label in window metadata (values dict won't persist across reads reliably)
                self.last_parsed = parsed
                self.last_prediction_label = label
                self.last_feature_row = row_dict  # to be saved
                self.window["STATUS"].update(f"Model {self.model_choice} predicted: {diag_text}")
            if event == "Confirm Diagnosis":
                # need to have a prediction first
                if not hasattr(self, "last_prediction_label"):
                    sg.popup_error("No prediction available. Please click Predict first.")
                    continue
                # Save to CSV with DX = model prediction
                dx_to_save = int(self.last_prediction_label)
                try:
                    self.append_row_to_csv(self.last_feature_row, dx_to_save)
                except Exception as e:
                    sg.popup_error(f"Error saving data: {e}")
                    continue
                sg.popup_ok("Data saved to NEWADNIMERGE.csv (confirmed diagnosis).")
                # Reset prediction text
                self.window["PREDICTION_TEXT"].update("")
                self.window["STATUS"].update("Saved confirmed diagnosis.")
            if event == "Contest Diagnosis":
                # Show the contest dropdown if not visible
                if not values.get("CONTEST_CHOICE"):
                    self.window["CONTEST_CHOICE"].update(visible=True)
                    continue
                # If visible and value selected, perform saving with selected diagnosis
                chosen = values.get("CONTEST_CHOICE")
                if not chosen:
                    sg.popup_error("Please select a contest diagnosis from the dropdown.")
                    continue
                # Map to integer code
                mapping = {"CN":0, "EMCI":1, "LMCI":2, "AD":3}
                dx_to_save = mapping[chosen]
                # require parsed inputs exist. If user hasn't clicked predict, parse now:
                if not hasattr(self, "last_feature_row"):
                    valid, parsed = self.validate_and_parse_inputs(values)
                    if not valid:
                        continue
                    # build row_dict for saving
                    row_dict = {}
                    for col in COLUMN_ORDER[1:]:
                        if col in parsed:
                            row_dict[col] = parsed[col]
                        else:
                            row_dict[col] = np.nan
                    try:
                        self.append_row_to_csv(row_dict, dx_to_save)
                    except Exception as e:
                        sg.popup_error(f"Error saving data: {e}")
                        continue
                else:
                    try:
                        self.append_row_to_csv(self.last_feature_row, dx_to_save)
                    except Exception as e:
                        sg.popup_error(f"Error saving data: {e}")
                        continue
                sg.popup_ok("Data saved to NEWADNIMERGE.csv (contested diagnosis recorded).")
                self.window["CONTEST_CHOICE"].update(visible=False, value="")
                self.window["PREDICTION_TEXT"].update("")
                self.window["STATUS"].update("Saved contested diagnosis.")

    # HELPERS: VALIDATION AND SAVING
    def validate_and_parse_inputs(self, values):
        """
        Validate all inputs based on VALIDATION rules.
        Returns (True, parsed_dict) or (False, None) and shows popup on error.
        """
        parsed = {}
        errors = []
        # Parse all parameters
        for param in ALL_PARAMS:
            key = f"IN_{param}"
            raw = values.get(key)
            if raw is None:
                errors.append(f"{param}: no value provided.")
                continue
            # special handling for PTGENDER and APOE4
            if param == "PTGENDER":
                # raw should be "Female" or "Male"
                if raw not in ("Female", "Male"):
                    errors.append("PTGENDER must be 'Female' or 'Male'.")
                else:
                    parsed[param] = 1 if raw == "Male" else 0
                continue
            if param == "APOE4":
                # ensure 0/1/2
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

            # Otherwise numeric input
            # attempt to convert to float
            try:
                val = float(raw)
            except Exception:
                errors.append(f"{param}: invalid numeric value.")
                continue
            
            # Check validation ranges if present
            if param in VALIDATION:
                minv, maxv, expected_type = VALIDATION[param]
                if val < minv or val > maxv:
                    errors.append(f"{param}: value {val} outside allowed range [{minv}, {maxv}].")
                    continue
                # Cast to int if expected
                if expected_type is int:
                    parsed[param] = int(val)
                else:
                    parsed[param] = float(val)
            else:
                # no range rule: keep as float
                parsed[param] = float(val)

        if errors:
            sg.popup_error("Input validation errors:\n" + "\n".join(errors))
            return False, None
        # Everything parsed successfully
        return True, parsed

    def append_row_to_csv(self, feature_row_dict, dx_value):
        """
        Append a single row to NEWADNIMERGE.csv.
        feature_row_dict: dict keyed by feature column names (columns after DX)
        dx_value: integer label
        """
        # Ensure data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)

        # Build a full dict including DX
        full_row = {}
        # Ensure we follow COLUMN_ORDER
        for col in COLUMN_ORDER:
            if col == "DX":
                full_row["DX"] = int(dx_value)
            else:
                # fill with value from feature_row_dict if present; else NaN
                v = feature_row_dict.get(col, np.nan)
                # cast PTGENDER to int (we stored 0/1), cast APOE4 int
                if col == "PTGENDER":
                    v = int(v) if not (pd.isna(v)) else v
                if col == "APOE4":
                    v = int(v) if not (pd.isna(v)) else v
                full_row[col] = v

        # Create DataFrame with one row
        df_row = pd.DataFrame([full_row], columns=COLUMN_ORDER)

        # If the file exists, append; otherwise create with header
        if os.path.exists(NEW_CSV):
            # append without header
            df_row.to_csv(NEW_CSV, mode='a', header=False, index=False)
        else:
            df_row.to_csv(NEW_CSV, mode='w', header=True, index=False)

# ----------------------------#
#            MAIN             #
# ----------------------------# 
def main():
    """
    Entry point to run the PySimpleGUI application.
    """
    # Ensure results directory exists (we just warn if missing)
    if not os.path.exists(RESULTS_DIR):
        sg.popup_ok(f"Warning: results directory '{RESULTS_DIR}' does not exist. Please create it and add your .pkl models.")
    pm = PageManager()
    pm.start()


if __name__ == "__main__":
    main()
