import pandas as pd
import joblib
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)
model = joblib.load(r'C:\Users\MASTER-IS\Desktop\google play apps\model\best_model.joblib')

# تحويل التاريخ
def process_input(last_updated):
    df = pd.DataFrame({'Last Updated': [last_updated]})
    df['Last Updated'] = pd.to_datetime(df['Last Updated'], errors='coerce')
    df['Last_Day'] = df['Last Updated'].dt.day
    df['Last_Weekday'] = df['Last Updated'].dt.weekday
    df['Last_Week'] = df['Last Updated'].dt.isocalendar().week.astype(int)
    df['Last_Month'] = df['Last Updated'].dt.month
    df['Last_Year'] = df['Last Updated'].dt.year
    return df.iloc[:, 1:]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_app_installs():
    data = request.form

    
    rating = float(data.get('Rating'))
    reviews = int(data.get('Reviews'))
    price = float(data.get('Price'))

   
    features = pd.DataFrame({
        'Category': [data.get('Category')],
        'Type': [data.get('Type')],
        'Content Rating': [data.get('Content_Rating')],
        'Genres': [data.get('Genres')],
        'Current Ver': [data.get('Current_Ver')],
        'Android Ver': [data.get('Android_Ver')],
        'Rating': [rating],
        'Reviews': [reviews],
        'Price': [price]
    })

    
    last_updated = data.get('Last_Updated')
    date_features = process_input(last_updated)
    features = pd.concat([features, date_features], axis=1)

    
    categorical_cols = ['Category', 'Type', 'Content Rating', 'Genres', 'Current Ver', 'Android Ver']
    features = pd.get_dummies(features, columns=categorical_cols)

    
    try:
        model_cols = model.feature_names_in_
        for col in model_cols:
            if col not in features.columns:
                features[col] = 0
        features = features[model_cols]  
    except AttributeError:
        pass  


    pred = int(model.predict(features)[0])
    return jsonify({'predicted_installs': pred})

if __name__ == "__main__":
    app.run(debug=True)