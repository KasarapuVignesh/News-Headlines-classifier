from flask import Flask, render_template, request
import pickle
import os

# Load models and vectorizer
nb_model = pickle.load(open('models/nb_model.pkl', 'rb'))
lr_model = pickle.load(open('models/lr_model.pkl', 'rb'))
vectorizer = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))

# Category labels


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    selected_model = None

    if request.method == 'POST':
        headline = request.form['headline']
        selected_model = request.form['model']

        # Vectorize input
        vect_input = vectorizer.transform([headline])

        # Predict
        if selected_model == 'naive_bayes':
            pred = nb_model.predict(vect_input)[0]
        elif selected_model == 'logistic_regression':
            pred = lr_model.predict(vect_input)[0]
        else:
            pred = "Invalid model"

        prediction = pred

    return render_template('index.html', prediction=prediction, selected_model=selected_model)

if __name__ == '__main__':
    app.run(debug=True)
