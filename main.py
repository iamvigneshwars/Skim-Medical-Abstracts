from flask import Flask, render_template, request, redirect
from nltk import sent_tokenize
from predict import classify

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main_page():
    
    if request.method == 'POST':
        abstract = request.form['abstract']
        # sentences = sent_tokenize(abstract)
        predicted_labels = classify(abstract)
        return render_template('main.html', results = predicted_labels)
    
    return render_template('main.html')


if __name__ == "__main__":
    app.run(debug=True)