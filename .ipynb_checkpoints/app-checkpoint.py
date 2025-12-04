# app.py
from flask import Flask, render_template, request
import pickle, numpy as np, os

app = Flask(__name__)
app.config["SECRET_KEY"] = "replace-this-with-a-secure-secret"

# Paths (ensure these exist after running train_save.py)
MODEL_PATH = "reg_model.pkl"
FEATURES_PATH = "feature_names.pkl"
ENCODERS_PATH = "label_encoders.pkl"

# Load artifacts
reg_model = None
feature_names = []
label_encoders = {}

try:
    with open(MODEL_PATH, "rb") as f:
        reg_model = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f:
        feature_names = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        label_encoders = pickle.load(f)
    print("Loaded model and metadata.")
except Exception as e:
    print("Error loading artifacts:", e)

# Helper
def is_categorical(col):
    return col in label_encoders

def preprocess_form(form):
    """
    Convert form (ImmutableMultiDict) -> numpy array shaped (1, n_features)
    Categorical features are encoded using saved LabelEncoders.
    Unknown categories are mapped to -1 (you can change this behavior).
    """
    row = []
    for col in feature_names:
        raw = form.get(col, "").strip()
        if is_categorical(col):
            encoder = label_encoders[col]
            classes = list(encoder.classes_)
            if raw in classes:
                encoded = int(encoder.transform([raw])[0])
            else:
                # try case-insensitive match
                match = None
                for c in classes:
                    if c.lower() == raw.lower():
                        match = c
                        break
                if match is not None:
                    encoded = int(encoder.transform([match])[0])
                else:
                    # fallback mapping for unseen label -1 (you may prefer to reject)
                    encoded = -1
            row.append(encoded)
        else:
            try:
                val = float(raw)
            except:
                val = 0.0
            row.append(val)
    return np.array([row], dtype=float)

@app.route("/", methods=["GET","POST"])
def index():
    if reg_model is None or not feature_names:
        return "<h3>Model not loaded. Run training script and ensure artifacts exist.</h3>"

    # Build inputs metadata for template
    inputs_meta = []
    for col in feature_names:
        if is_categorical(col):
            options = list(label_encoders[col].classes_)
            inputs_meta.append({"name": col, "type": "categorical", "options": options})
        else:
            inputs_meta.append({"name": col, "type": "numeric", "options": None})

    prediction = None
    error = None

    if request.method == "POST":
        try:
            X = preprocess_form(request.form)
            pred = reg_model.predict(X)
            prediction = float(pred[0])
        except Exception as e:
            error = str(e)

    return render_template("predict.html",
                           inputs_meta=inputs_meta,
                           prediction=prediction,
                           error=error)

if __name__ == "__main__":
    app.run(debug=True)
