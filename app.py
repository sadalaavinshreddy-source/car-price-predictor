from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# ======================
# LOAD MODEL + ENCODERS
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "car_price_model.pkl"))
le_company = joblib.load(os.path.join(BASE_DIR, "company_encoder.pkl"))
le_fuel = joblib.load(os.path.join(BASE_DIR, "fuel_encoder.pkl"))
le_trans = joblib.load(os.path.join(BASE_DIR, "transmission_encoder.pkl"))
le_cond = joblib.load(os.path.join(BASE_DIR, "condition_encoder.pkl"))


# ======================
# HOME PAGE
# ======================
@app.route("/")
def home():
    return render_template("index.html")


# ======================
# PREDICTION ROUTE
# ======================
@app.route("/predict", methods=["POST"])
def predict():
    try:

        # ===== INPUTS =====
        company = request.form.get("company")
        year = int(request.form.get("year") or 2000)
        kms = int(request.form.get("kms") or 0)
        fuel = request.form.get("fuel")
        transmission = request.form.get("transmission")
        condition = request.form.get("condition")

        # ===== SAFE ENCODING =====
        def safe_encode(le, value):
            try:
                return le.transform([value])[0]
            except:
                return 0

        company_val = safe_encode(le_company, company)
        fuel_val = safe_encode(le_fuel, fuel)
        trans_val = safe_encode(le_trans, transmission)
        cond_val = safe_encode(le_cond, condition)

        # ===== FINAL INPUT (6 FEATURES ONLY) =====
        input_data = np.array([[
            company_val,
            year,
            kms,
            fuel_val,
            trans_val,
            cond_val
        ]])

        # ===== PREDICTION =====
        prediction = model.predict(input_data)[0]
        prediction = round(float(prediction), 2)

        return render_template(
            "index.html",
            prediction_text=f"🚗 Predicted Car Price: ₹ {prediction}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )


# ======================
# RUN APP
# ======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)