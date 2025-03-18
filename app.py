from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)

# Load dataset
data = pd.read_csv('cost-revenue-clean.csv')

# Train Model
X = data[['production_budget_usd']]
y = data[['worldwide_gross_usd']]
regression = LinearRegression()
regression.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        budget = float(request.form['budget'])
        predicted_revenue = regression.predict([[budget]])[0][0]
        predicted_revenue = max(0, predicted_revenue)  # Prevent negative predictions
        return render_template('index.html', prediction=f"${predicted_revenue:,.2f}")
    except ValueError:
        return render_template('index.html', error="❌ Invalid input! Please enter a valid number.")

# This is for local development
if __name__ == '__main__':
    app.run(debug=True)
    
# This is for Vercel
app = app
