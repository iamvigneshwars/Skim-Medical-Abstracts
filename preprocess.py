from nltk import sent_tokenize
import pandas as pd
import numpy as np
import joblib

def create_data(abstract):

    data = sent_tokenize(abstract)
    abstracts = []

    for line_no, abst_lines in enumerate(data):
        each_line = {}
        each_line['position'] = str(line_no+1) +"_of_"+ str(len(data))
        each_line["text"] = abst_lines 
        abstracts.append(each_line)

    def split(text):
        return ' '.join(list(text))

    abstract = pd.DataFrame(abstracts)
    abs_sent = abstract.text
    abs_char = abstract.text.apply(split)
    one_hot = joblib.load('Model/one_hot.joblib')
    abs_pos = one_hot.transform(np.expand_dims(abstract.position, axis = 2)).toarray()
    abs_pred_probs = model.predict(x = (abs_sent,
                                    abs_char,
                                    abs_pos))

    return (abs_sent, abs_char, abs_pos)


