from flask import Flask, render_template, request
import nltk
import pickle
from nltk.corpus import stopwords
import re
import os
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)
ps = PorterStemmer()
nltk.data.path.append('./nltk_data')

model = pickle.load(open('model.pkl', 'rb'))
tfidfvect = pickle.load(open('tfidfvect.pkl', 'rb'))

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    original_text = request.form['text']
    review = re.sub('[^a-zA-Z]', ' ', original_text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    review_vect = tfidfvect.transform([review]).toarray()
    prediction = 'FAKE' if model.predict(review_vect) == 0 else 'REAL'
    return render_template('index.html', text=original_text, result=prediction)


if __name__ == "__main__":
    app.run(port=3000, debug=True)
