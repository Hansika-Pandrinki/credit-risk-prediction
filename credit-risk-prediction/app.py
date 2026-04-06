from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Sample dataset
data = {
    'Income': ['High', 'Low', 'Medium', 'High', 'Low', 'Medium', 'High', 'Low', 'Medium'],
    'Loan_Amount': ['Low', 'High', 'Medium', 'Low', 'High', 'Medium', 'Low', 'High', 'Medium'],
    'Payment_History': ['Good', 'Bad', 'Average', 'Good', 'Bad', 'Average', 'Good', 'Bad', 'Average'],
    'Age': ['Young', 'Old', 'Middle', 'Middle', 'Young', 'Old', 'Old', 'Middle', 'Young'],
    'Credit_Score': ['Good', 'Poor', 'Average', 'Good', 'Poor', 'Average', 'Good', 'Poor', 'Average']
}

df = pd.DataFrame(data)

# Encode data
encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df[['Income', 'Loan_Amount', 'Payment_History', 'Age']]
y = df['Credit_Score']

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    income = request.form['income'].capitalize()
    loan_amount = request.form['loan_amount'].capitalize()
    payment_history = request.form['payment_history'].capitalize()
    age = request.form['age'].capitalize()

    new_data = pd.DataFrame([[income, loan_amount, payment_history, age]],
                            columns=['Income', 'Loan_Amount', 'Payment_History', 'Age'])

    try:
        for col in new_data.columns:
            new_data[col] = encoders[col].transform(new_data[col])

        prediction = model.predict(new_data)
        result = encoders['Credit_Score'].inverse_transform(prediction)[0]

        if result == "Good":
            prediction_text = "Credit Score: Good"
        elif result == "Average":
            prediction_text = "Credit Score: Average"
        else:
            prediction_text = "Credit Score: Poor"

    except Exception:
        prediction_text = "Please enter valid values only."

    return render_template('index.html', prediction_text=prediction_text)


if __name__ == "__main__":
    app.run(debug=True)