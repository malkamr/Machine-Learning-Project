from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load('churn_model.pkl')
model_columns = joblib.load('model_columns.pkl')


def build_features(data):
    balance = float(data['Balance'])
    estimated_salary = float(data['EstimatedSalary'])
    credit_score = float(data['CreditScore'])
    age = float(data['Age'])

    input_dict = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': float(data['Tenure']),
        'Balance': balance,
        'NumOfProducts': float(data['NumOfProducts']),
        'HasCrCard': int(data['HasCrCard']),
        'IsActiveMember': int(data['IsActiveMember']),
        'EstimatedSalary': estimated_salary,
        'BalanceSalaryRatio': balance / (estimated_salary + 1),
        'CreditScoreAge': credit_score / age,
        'Geography_Germany': 1 if data['Geography'] == 'Germany' else 0,
        'Geography_Spain': 1 if data['Geography'] == 'Spain' else 0,
        'Gender_Male': 1 if data['Gender'] == 'Male' else 0
    }

    return pd.DataFrame([input_dict])[model_columns]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True) if request.is_json else request.form

    required_fields = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                        'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography', 'Gender']

    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400

    try:
        features = build_features(data)
    except (ValueError, KeyError) as e:
        return jsonify({'error': str(e)}), 400

    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])

    result = {
        'churn_prediction': prediction,
        'churn_probability': round(probability, 4),
        'label': 'Likely to churn' if prediction == 1 else 'Likely to stay'
    }

    if request.is_json:
        return jsonify(result)
    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
