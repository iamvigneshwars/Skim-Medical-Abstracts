from flask import Flask, render_template, request, redirect
from nltk import sent_tokenize
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from model import transformer
import tensorflow as tf
# from tensorflow.keras import layers
from preprocess import create_data
global model

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main_page():
    
    model = transformer()
    if request.method == 'POST':
        abstract = request.form['abstract']
        classes = ["BACKGROUND", "CONCLUSIONS", "METHODS", "OBJECTIVE", "RESULTS"]
        data = create_data(abstract)
        abs_pred_probs = model.predict(x = data)
        abs_preds = tf.argmax(abs_pred_probs, axis=1)
        abs_pred_classes = [classes[i] for i in abs_preds]
        
        results = []
        for i , line in enumerate(data[0]):
            predicted = {
                    'label':abs_pred_classes[i],
                    'sentence':line
                    }

            results.append(predicted)
        return render_template('main.html', results = results)
    
    return render_template('main.html')

if __name__ == "__main__":
    app.run(debug=True)
