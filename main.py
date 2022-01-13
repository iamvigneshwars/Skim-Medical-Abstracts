import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from model import transformer
import tensorflow as tf
from tensorflow.keras import layers
from nltk import sent_tokenize
import pandas as pd
import numpy as np
import joblib

print("----------INITIALIZING MODEL----------\n")
model= transformer()
one_hot = joblib.load('Model/one_hot.joblib')
def classify(data, model):
    classes = ["BACKGROUND", "CONCLUSIONS", "METHODS", "OBJECTIVE", "RESULTS"]
    data = sent_tokenize(data)
    abstracts = [] 
    for line_no, abst_lines in enumerate(data):
        each_line = {} 
        each_line['position'] = str(line_no+1) +"_of_"+ str(len(data))
        each_line["text"] = abst_lines # to get the text of sentence in convert to lower
        abstracts.append(each_line) # add dictionary to list of abstracts.
    # reset the sample lines for next abstract.
    def split(text):
        return ' '.join(list(text))

    abstract = pd.DataFrame(abstracts)  
    abs_sent = abstract.text
    abs_char = abstract.text.apply(split)
    abs_pos = one_hot.transform(np.expand_dims(abstract.position, axis = 1)).toarray()
    abs_pred_probs = model.predict(x = (abs_sent,
                                    abs_char,
                                    abs_pos))
    abs_preds = tf.argmax(abs_pred_probs, axis=1)
    abs_pred_classes = [classes[i] for i in abs_preds]
    
    for i , line in enumerate(data):
        print(abs_pred_classes[i],": ")
        print(line, "\n")

abstract = input("Enter Unstructured Medical Abstract: \n")
print("----------CLASSIFYING----------\n")

classify(abstract, model)
