from flask import Flask, render_template, request, redirect
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from model import hybridModel
import numpy as np
import tensorflow as tf
from preprocess import createData


model = hybridModel()
classes = ["BACKGROUND", "CONCLUSION", "METHOD", "OBJECTIVE", "RESULT"]

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main_page():
    
    try:

        if request.method == 'POST':
            if request.form['abstract']:
                abstract = request.form['abstract']
                global results
                results = []
                data = createData(abstract)
                abs_pred_probs = model.predict(x = data)
                pred_prob = np.max(abs_pred_probs, axis = 1)
                abs_preds = tf.argmax(abs_pred_probs, axis=1).numpy()
                abs_pred_classes = [classes[i] for i in abs_preds]
                
                for i , line in enumerate(data[0]):
                    predicted = {
                            'label':abs_pred_classes[i],
                            'sentence':line,
                            'prob':round(pred_prob[i] * 100, 2) 
                            }

                    results.append(predicted)
                return redirect('/skim-abstracts=5')
            return redirect('/')
        
        return render_template('main.html')

    except:
        return render_template('main.html', error = True)

@app.route('/skim-abstracts=<int:id>', methods=['GET', 'POST'])
def prediction_page(id):

    return render_template('prediction_page.html',classes = classes,  results = results, id = id)

if __name__ == "__main__":
    app.run(debug=True)
