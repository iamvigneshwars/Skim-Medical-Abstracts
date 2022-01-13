from model import transformer
import tensorflow as tf
from tensorflow.keras import layers
import os
from preprocess import create_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("\n----------INITIALIZING MODEL----------\n")
model= transformer()
def classify(data, model):
    classes = ["BACKGROUND", "CONCLUSIONS", "METHODS", "OBJECTIVE", "RESULTS"]

    abs_preb_probs = model.predict(x = data)
    abs_preds = tf.argmax(abs_pred_probs, axis=1)
    abs_pred_classes = [classes[i] for i in abs_preds]
    
    for i , line in enumerate(data):
        print(abs_pred_classes[i],": ")
        print(line, "\n")

abstract = input("Enter Unstructured Medical Abstract: \n")
abstract = create_data(abstract)

print("\n----------CLASSIFYING----------\n")

classify(abstract, model)
