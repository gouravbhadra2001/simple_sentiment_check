from flask import Flask, render_template, request, redirect, url_for

import pickle

app = Flask(__name__)
model = pickle.load(open('sentiment_analyser.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html', prediction_text="")

@app.route('/predict', methods=['POST'])
def predict():
    # Get the user input from the form
    user_text = request.form['userText']

    # Perform sentiment analysis using the loaded model
    sentiment_scores = model.polarity_scores(user_text)
    compound_score = sentiment_scores['compound']

    # Classify the sentiment based on the compound score
    if compound_score >= 0.05:
        prediction = "POSITIVE"
    elif compound_score <= -0.05:
        prediction = "NEGATIVE"
    else:
        prediction = "NEUTRAL"

    # Redirect to a new route with the prediction
    return redirect(url_for('result', prediction=prediction, user_input = user_text))

@app.route('/result/<prediction>')
def result(prediction):
    user_input = request.args.get('user_input', '')  # Retrieve user_input from query parameters
    return render_template('index.html', prediction_text=f"{prediction}", user_text=f"{user_input}")
