from flask import Flask, render_template, request
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
import joblib

app = Flask(__name__)

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Train a simple model (Random Forest) for demonstration purposes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'iris_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get feature inputs from the form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Make a prediction using the loaded model
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(features)

        # Map the numeric prediction to the actual species
        species_mapping = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
        predicted_species = species_mapping[prediction[0]]

        return render_template('result.html', species=predicted_species)

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return render_template('error.html', error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
