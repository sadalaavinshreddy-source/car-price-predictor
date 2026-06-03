from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("car_price_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # SAFE INPUT (NO CRASH VERSION)
        car_name = request.form.get('car_name')
        model_name = request.form.get('model')
        ad_id = request.form.get('ad_id')

        year = request.form.get('year')
        kms = request.form.get('kms')

        fuel = request.form.get('fuel')
        documents = request.form.get('documents')
        assembly = request.form.get('assembly')
        transmission = request.form.get('transmission')
        condition = request.form.get('condition')
        reg_type = request.form.get('registration_type')
        reg_city = request.form.get('registration_city')

        # convert safe int
        year = int(year) if year else 0
        kms = int(kms) if kms else 0

        # encoding maps
        fuel_map = {"Petrol": 0, "Diesel": 1}
        trans_map = {"Manual": 0, "Automatic": 1}
        doc_map = {"Original": 0, "Not Original": 1}
        assembly_map = {"Local": 0, "Imported": 1}
        cond_map = {"Used": 0, "Not Used": 1}
        reg_map = {"Registered": 0, "Not Registered": 1}

        # convert safely
        fuel_val = fuel_map.get(fuel, 0)
        trans_val = trans_map.get(transmission, 0)
        doc_val = doc_map.get(documents, 0)
        assembly_val = assembly_map.get(assembly, 0)
        cond_val = cond_map.get(condition, 0)
        reg_val = reg_map.get(reg_type, 0)

        # final input
        input_data = np.array([[
            fuel_val,
            trans_val,
            doc_val,
            assembly_val,
            cond_val,
            reg_val,
            year,
            kms
        ]])

        # prediction
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


if __name__ == "__main__":
    app.run(debug=True)