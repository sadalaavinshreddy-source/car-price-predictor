from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load Model
model = joblib.load("car_price_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    data = {
        'Ad ID': [int(request.form['ad_id'])],
        'Car Name': [request.form['car_name']],
        'Make': [request.form['make']],
        'Model': [request.form['model']],
        'Year': [int(request.form['year'])],
        "KM's driven": [int(request.form['kms'])],
        'Fuel': [request.form['fuel']],
        'Registration city': [request.form['registration_city']],
        'Car documents': [request.form['documents']],
        'Assembly': [request.form['assembly']],
        'Transmission': [request.form['transmission']],
        'Condition': [request.form['condition']],
        'Seller Location': [request.form['seller_location']]
    }

    input_df = pd.DataFrame(data)

    prediction = model.predict(input_df)

    predicted_price = round(float(prediction[0]), 2)

    return render_template(
        'index.html',
        prediction=predicted_price
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)