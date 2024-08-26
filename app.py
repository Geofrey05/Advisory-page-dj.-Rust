from flask import Flask, render_template, request
import numpy as np
import pickle

# Load the trained model and scalers
model = pickle.load(open('model.pkl', 'rb'))
minmax_scaler = pickle.load(open('MinMaxScaler.pkl', 'rb'))
standard_scaler = pickle.load(open('StandardScaler.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input data from the form
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        
        # Prepare data for prediction
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        scaled_features = minmax_scaler.transform(features)
        standardized_features = standard_scaler.transform(scaled_features)
        
        # Make a prediction
        prediction = model.predict(standardized_features)
        
        # Map the prediction to the crop name
        crop_dict = {
            1: 'rice',
            2: 'maize',
            3: 'jute',
            4: 'cotton',
            5: 'coconut',
            6: 'papaya',
            7: 'orange',
            8: 'apple',
            9: 'muskmelon',
            10: 'watermelon',
            11: 'grapes',
            12: 'mango',
            13: 'banana',
            14: 'pomegranate',
            15: 'lentil',
            16: 'blackgram',
            17: 'mungbean',
            18: 'mothbeans',
            19: 'pigeonpeas',
            20: 'kidneybeans',
            21: 'chickpea',
            22: 'coffee'
        }
        
        result = crop_dict[int(prediction[0])]
        
        return render_template('./index.html', prediction_text=f'Recommended Crop: {result}')

if __name__ == "__main__":
    app.run(debug=True, port=5001)
