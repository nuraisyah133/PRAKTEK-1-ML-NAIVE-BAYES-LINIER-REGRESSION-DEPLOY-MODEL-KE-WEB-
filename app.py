from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("model/hasil_pelatihan_model.pkl", "rb") as mul_reg:
    ml_model = joblib.load(mul_reg)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    print("Prediksi dimulai")
    if request.method == 'POST':
        try:
            # Get values from the form
            RnD_Spend = float(request.form.get('RnD_Spend'))
            Admin_Spend = float(request.form.get('Admin_Spend'))
            Market_Spend = float(request.form.get('Market_Spend'))
            
            # Prepare data for prediction
            pred_args = [RnD_Spend, Admin_Spend, Market_Spend]
            pred_args_arr = np.array(pred_args).reshape(1, -1)
            
            # Make the prediction
            model_prediction = ml_model.predict(pred_args_arr)
            model_prediction = round(float(model_prediction[0]), 2)
            
            return render_template('predict.html', prediction=model_prediction)
        except ValueError:
            return "Please check if the values are entered correctly"
    return render_template('home.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0')
