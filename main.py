from flask import Flask, render_template, request, redirect
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from model import transformer
import tensorflow as tf
from preprocess import create_data

app = Flask(__name__)

model = transformer()
@app.route('/', methods=['GET', 'POST'])
def main_page():
    
    if request.method == 'POST':
        if request.form['abstract']:
            abstract = request.form['abstract']
            global results
            results = []
            classes = ["BACKGROUND", "CONCLUSIONS", "METHODS", "OBJECTIVE", "RESULTS"]
            data = create_data(abstract)
            abs_pred_probs = model.predict(x = data)
            abs_preds = tf.argmax(abs_pred_probs, axis=1)
            abs_pred_classes = [classes[i] for i in abs_preds]
            
            for i , line in enumerate(data[0]):
                predicted = {
                        'label':abs_pred_classes[i],
                        'sentence':line
                        }

                results.append(predicted)
            return redirect('/skim-abstracts')
        return redirect('/')
    
    return render_template('main.html')

@app.route('/skim-abstracts', methods=['GET', 'POST'])
def prediction_page():

    return render_template('prediction_page.html',  results = results)

if __name__ == "__main__":
    app.run(debug=True)
