from nltk import sent_tokenize
import pandas as pd
import numpy as np
import joblib

def createData(abstract):

    data = sent_tokenize(abstract)
    abstracts = []

    def scaleNumberOfLines(num, maximum):
        new_value = ( (num - 1) / (maximum - 1) ) * (5 - 1) + 1
        return round(new_value)

    position = ['#', 'FIRST', 'SECOND', 'THIRD', 'FOURTH', 'FIFTH']
    for line_no, abst_lines in enumerate(data):
        each_line = {}
        each_line["text"] = abst_lines 
        each_line['position'] = position[scaleNumberOfLines(line_no+1, len(data))]
        abstracts.append(each_line)

    abstract = pd.DataFrame(abstracts)
    abs_sent = abstract.text
    one_hot = joblib.load('Model/one_hot.joblib')
    abs_pos = one_hot.transform(np.expand_dims(abstract.position, axis = 1)).toarray()

    return (abs_sent, abs_pos)


