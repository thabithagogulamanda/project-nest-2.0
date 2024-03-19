from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("Training Dataset.csv")
data.dropna(inplace=True)

# Select features and target variable
selected_features = ['ApplicantIncome', 'LoanAmount']
X = data[selected_features]
y = data["Loan_Status"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ApplicantIncome = float(request.form['ApplicantIncome'])
    LoanAmount = float(request.form['LoanAmount'])

    user_input = np.array([[ApplicantIncome, LoanAmount]])

    # Make prediction
    predicted_approval = model.predict(user_input)

    # Render the prediction result template with the prediction
    return render_template('predict.html', prediction=predicted_approval[0])

if __name__ == '__main__':
    app.run(debug=True)
