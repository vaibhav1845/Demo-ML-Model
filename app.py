from flask import Flask, request, render_template
import pickle
import numpy as  np

# Load the trained model
model_path = 'C:/Users/akash/OneDrive/Desktop/demo/model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    int_features = [int(x) for x in  request.form.values()]
    final_features = [np.array(int_features)]

    # Make prediction
    prediction = model.predict(final_features)
    output = 'placed' if prediction[0] == 1 else "not placed"

    return render_template('index.html', prediction_text = 'Prediction: {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)