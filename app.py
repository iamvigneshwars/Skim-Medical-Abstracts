from flask import Flask, render_template, request, redirect
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from model import hybridModel
import numpy as np
import tensorflow as tf
from preprocess import createData


# Inialize the model 
MODEL = hybridModel()
CLASSES = ["BACKGROUND", "CONCLUSION", "METHOD", "OBJECTIVE", "RESULT"]

app = Flask(__name__)
# global RESULTS
# RESULTS = []

@app.route('/', methods=['GET', 'POST'])
def main_page():
    
    try:

        # If the user enters the abstracts then the model makes
        # predictions and the prediction page will be rendered.
        if request.method == 'POST':
            if request.form['abstract']:
                abstract = request.form['abstract']
                # Store the predicted labels
                global RESULTS 
                RESULTS = []
                # Preprocess the data
                data = createData(abstract)
                abs_pred_probs = MODEL.predict(x = data)
                pred_prob = np.max(abs_pred_probs, axis = 1)
                abs_preds = tf.argmax(abs_pred_probs, axis=1).numpy()
                abs_pred_classes = [CLASSES[i] for i in abs_preds]
                
                for i , line in enumerate(data[0]):
                    predicted = {
                            'label':abs_pred_classes[i],
                            'sentence':line,
                            'prob':round(pred_prob[i] * 100, 2) 
                            }

                    RESULTS.append(predicted)
                
                # Redirects to the prediction page
                return redirect('/skim-abstracts=5')

            # Blank submit
            return redirect('/')
        
        # Initial page. 
        return render_template('index.html')

    except:
        # If the input is less than three lines, or if the prediction pages breaks for some
        # reason, renders the initial page. If you want to add to this code or debug this code, it is better not to 
        # use try except block.
        return render_template('index.html', error = True)

@app.route('/skim-abstracts=<int:id>', methods=['GET', 'POST'])
def prediction_page(id):

    # Renders the prediction page with predicted labels.
    return render_template('prediction_page.html',classes = CLASSES,  results = RESULTS, id = id)

if __name__ == "__main__":
    app.run(debug=True)
