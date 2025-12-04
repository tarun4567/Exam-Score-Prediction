from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'abc123'


try:
    with open("reg_model.pkl", "rb") as f:
        reg_model = pickle.load(f)

    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    with open("label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)

    print("Model & encoders loaded successfully.")

except Exception as e:
    print("Error loading files:", e)



def is_categorical(col):
    return col in label_encoders


def preprocess_form_input(form):
    row = []
    for col in feature_names:
        raw = form.get(col, "").strip()

        if is_categorical(col):
            encoder = label_encoders[col]
            classes = list(encoder.classes_)

            # exact match
            if raw in classes:
                val = encoder.transform([raw])[0]
            else:
               
                raw_lower = raw.lower()
                match = next((c for c in classes if c.lower() == raw_lower), None)

                if match:
                    val = encoder.transform([match])[0]
                else:
                    val = -1  
        else:
            try:
                val = float(raw)
            except:
                val = 0.0

        row.append(val)

    return np.array([row])


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    input_fields = []
    for col in feature_names:
        if is_categorical(col):
            options = list(label_encoders[col].classes_)
            input_fields.append({
                "name": col,
                "type": "categorical",
                "options": options
            })
        else:
            input_fields.append({
                "name": col,
                "type": "numeric",
                "options": None
            })

    if request.method == "POST":
        try:
            processed_input = preprocess_form_input(request.form)
            predicted_value = reg_model.predict(processed_input)[0]
            prediction = round(float(predicted_value), 2)

        except Exception as e:
            error = str(e)

    return render_template("predict.html",
                           input_fields=input_fields,
                           prediction=prediction,
                           error=error)


if __name__ == "__main__":
    app.run(debug=True)
