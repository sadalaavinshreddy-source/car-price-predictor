from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("car_price_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # ---------------- INPUT FROM FORM ----------------
    car_name = request.form['car_name']
    model_name = request.form['model']
    ad_id = request.form['ad_id']
    year = int(request.form['year'])
    kms = int(request.form['kms'])
    fuel = request.form['fuel']
    documents = request.form['documents']
    assembly = request.form['assembly']
    transmission = request.form['transmission']
    condition = request.form['condition']
    reg_type = request.form['registration_type']
    reg_city = request.form['registration_city']

    # ---------------- ENCODING ----------------
    fuel_map = {"Petrol": 0, "Diesel": 1}
    trans_map = {"Manual": 0, "Automatic": 1}
    doc_map = {"Original": 0, "Not Original": 1}
    assembly_map = {"Local": 0, "Imported": 1}
    cond_map = {"Used": 0, "Not Used": 1}
    reg_map = {"Registered": 0, "Not Registered": 1}

    fuel_val = fuel_map.get(fuel, 0)
    trans_val = trans_map.get(transmission, 0)
    doc_val = doc_map.get(documents, 0)
    assembly_val = assembly_map.get(assembly, 0)
    cond_val = cond_map.get(condition, 0)
    reg_val = reg_map.get(reg_type, 0)

    # ---------------- FINAL INPUT ARRAY ----------------
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

    # ---------------- PREDICTION ----------------
    prediction = model.predict(input_data)[0]

    # Convert to Indian Rupees format
    prediction = round(float(prediction), 2)

    output = f"Predicted Car Price: {prediction}"

    return render_template("index.html", prediction_text=output)


if __name__ == "__main__":
    app.run(debug=True)