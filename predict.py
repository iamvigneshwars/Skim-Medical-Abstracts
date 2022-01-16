import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from model import transformer
import tensorflow as tf
from tensorflow.keras import layers
from preprocess import create_data

def classify(data):
    classes = ["BACKGROUND", "CONCLUSIONS", "METHODS", "OBJECTIVE", "RESULTS"]
    model= transformer()
    data = create_data(data)
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

    return results
