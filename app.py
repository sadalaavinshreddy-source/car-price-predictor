from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# =========================
# LOAD MODEL + ENCODERS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "car_price_model.pkl"))
le_company = joblib.load(os.path.join(BASE_DIR, "company_encoder.pkl"))
le_fuel = joblib.load(os.path.join(BASE_DIR, "fuel_encoder.pkl"))
le_trans = joblib.load(os.path.join(BASE_DIR, "trans_encoder.pkl"))
le_cond = joblib.load(os.path.join(BASE_DIR, "condition_encoder.pkl"))


# =========================
# HOME PAGE
# =========================
@app.route("/")
def home():
    return render_template("index.html")


# =========================
# PREDICTION
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:

        # INPUTS
        company = request.form.get("company")
        year = int(request.form.get("year"))
        kms = int(request.form.get("kms"))
        fuel = request.form.get("fuel")
        transmission = request.form.get("transmission")
        condition = request.form.get("condition")

        # ENCODING (IMPORTANT FIX)
        company_val = le_company.transform([company])[0]
        fuel_val = le_fuel.transform([fuel])[0]
        trans_val = le_trans.transform([transmission])[0]
        cond_val = le_cond.transform([condition])[0]

        # FINAL INPUT ARRAY (MUST MATCH TRAINING ORDER)
        input_data = np.array([[
            company_val,
            year,
            kms,
            fuel_val,
            trans_val,
            cond_val
        ]])

        # PREDICTION
        prediction = model.predict(input_data)[0]
        prediction = round(float(prediction), 2)

        return render_template(
            "index.html",
            prediction_text=f"Predicted Price: ₹ {prediction}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(debug=True)