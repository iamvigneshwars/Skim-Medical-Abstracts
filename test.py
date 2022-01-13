from model import transformer
from spacy .lang.en import English

model = transformer()

def classify(data, model):

    nlp = English()
    sentencizer = nlp.create_pipe("sentencizer")
    nlp.add_pipe(sentencizer)

    data = nlp(data)
    data = [str(sent) for sent in list(data.sents)]

    abstracts = [] # To store the dictonaries

    for line_no, abst_lines in enumerate(data):
        each_line = {}
        each_line['position'] = str(line_no+1) +"_of_"+ str(len(data))
        each_line["text"] = abst_lines # to get the text of sentence in convert to lower
        abstracts.append(each_line) # add dictionary to list of abstracts.
    # reset the sample lines for next abstract.

    abstract = pd.DataFrame(abstracts)

    abs_sent = abstract.text
    abs_char = abstract.text.apply(split)
    abs_pos = one_hot.transform(np.expand_dims(abstract.position, axis = 1)).toarray()

    abs_sent= vectorizer(abs_sent)
    abs_char = vectorizer_char(abs_char)

    abs_pred_probs = model.predict(x = (abs_sent,
                                    abs_char,
                                    abs_pos))

    abs_preds = tf.argmax(abs_pred_probs, axis=1)
    abs_pred_classes = [classes[i] for i in abs_preds]

    for i , line in enumerate(data):
        print(abs_pred_classes[i],": ")
        print(line, "\n")

abstract = input("Enter the Medical Abstract: \n")
classify(abstract, model)
