# Import Libraries
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# from termcolor import colored
# import matplotlib.pyplot as plt
# import itertools
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

# Useful Custom Functions that will be useful on ipython notebook 
def model_results(y_true, y_pred):
    
    """
    This function calculates the Accuracy, precision, recall, f1 scores of the classification
    model
    
    Args:
        y_true : True labels of the data use for prediction.
        y_pred : Predicted labels by the model.
    
    Returns:
        A dictionary containing accuracy, precision, recall and f1 score. 
    
    """
    
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision,recall,f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    results = {"accuracy": round(accuracy, 5),
               "precision": round(precision,5),
               "recall": round(recall, 5),
               "f1": round(f1,5)}
    return results


def class_report(true_y, pred_y):
    cl_report = classification_report(true_y, pred_y,target_names=classes, output_dict=True)
    cl_report = pd.DataFrame(cl_report)
    cl_report= cl_report.T
    cl_report = cl_report*100
    cl_report = cl_report.round(decimals = 2)
    cl_report.drop('support', axis = 1, inplace = True)
    cl_report.drop('accuracy', axis = 0, inplace = True)
    
    return cl_report

def conf_matrix(true_y, pred_y):
    
    conf_mat = confusion_matrix(true_y, pred_y)
    cm = conf_mat
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] 
    n_classes = cm.shape[0]

    fig, ax = plt.subplots(figsize=(12, 12))
    cax = ax.matshow(cm_norm, cmap=plt.cm.Blues)
    fig.colorbar(cax,  shrink=0.8)
    labels = classes
    ax.set(title="Confusion Matrix",
             xlabel="Predicted label",
             ylabel="True label",
             xticks=np.arange(n_classes),
             yticks=np.arange(n_classes), 
             xticklabels=labels, 
             yticklabels=labels)
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    threshold = 75
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
              horizontalalignment="center",
              color="white" if cm_norm[i, j]*100 > threshold else "black",
              size=9)
    

def print_compare(current_model, prev_model = None):
    
    """
    This function prints the results of current model and the difference between the results
    of current model and the previous model.
    
    Args:
        current_model: dictionary contining the results of current model.
        prev_model: dictionary containing the results of the previous model.
        
    Returns:
        A dictionary contining the difference between the current model and previous model.
    """
    if not(prev_model):
        print("ACCURACY  : ", round(current_model['accuracy'],2), end = " ")
        print("\tRECALL   :", round(current_model['recall']*100, 2))
        print("PRECISION : ", round(current_model['precision']*100, 2), end = " ")
        print("\tF1 SCORE :", round(current_model['f1']*100, 2))
   
    if(prev_model):
        comp = {}
        
        for key in prev_model.keys():
            if not(isinstance(prev_model[key], str)):
                comp[key] = current_model[key] - prev_model[key]
        # Accuracy
        if(round(comp['accuracy'], 2) > 0):
            print("ACCURACY :", 
            round(current_model['accuracy'],2), 
            colored(str(abs(round(comp['accuracy'], 2))) + "%↑\t",'green'), 
            end = " ")
        elif(round(comp['accuracy'], 2) < 0):
            print("ACCURACY :", 
                  round(current_model['accuracy'],2), 
                  colored(str(abs(round(comp['accuracy'], 2))) + "%↓\t",'red'), 
                  end = " ")
        else:
            print("ACCURACY :", 
                  round(current_model['accuracy'],2), 
                  colored(str(abs(round(comp['accuracy'], 2))) + "%\t",'yellow'), 
                  end = " ")
        
        # Recall
        if(round(comp['recall'], 2) > 0):
            print("RECALL:", 
                round(current_model['recall']*100,2), 
                colored(str(abs(round(comp['recall'], 2))) + "%↑",'green'))
        elif(round(comp['recall'], 2) < 0):
            print("RECALL:", 
                round(current_model['recall']*100,2), 
                colored(str(abs(round(comp['recall'], 2))) + "%↓",'red'))
        else:
            print("RECALL:", 
                round(current_model['recall']*100,2), 
                colored(str(abs(round(comp['recall'], 2))) + "%",'yellow'))
            
        # Precision
        if(round(comp['precision'], 2) > 0):
            print("PRECISION:", 
                round(current_model['precision']*100,2), 
                colored(str(abs(round(comp['precision'], 2))) + "%↑\t",'green'), 
                end = " ")
        elif(round(comp['precision'], 2) < 0):
            print("PRECISION:", 
                round(current_model['precision']*100,2), 
                colored(str(abs(round(comp['precision'], 2))) + "%↓\t",'red'), 
                end = " ")
        else:
            print("PRECISION:", 
                round(current_model['precision']*100,2), 
                colored(str(abs(round(comp['precision'], 2))) + "%\t",'yellow'), 
                end = " ")    
            
        # F1 SCORE
        if(round(comp['f1'], 2) > 0):
            print("F1    :", 
                round(current_model['f1']*100,2), 
                colored(str(abs(round(comp['f1'], 2))) + "%↑",'green'))
        elif(round(comp['f1'], 2) < 0):
            print("F1    :", 
                  round(current_model['f1']*100,2), 
                  colored(str(abs(round(comp['f1'], 2))) + "%↓",'red'))
        else:
            print("F1    :", 
                  round(current_model['f1']*100,2), 
                  colored(str(abs(round(comp['f1'], 2))) + "%",'yellow'))     
                    
        return comp
    
def split(text):
    return ' '.join(list(text))

# Load and create dataset
def load_data(dataset):
    """    
    Reads the file and returns a list of lines in the file.
    
    Args: 
        dataset: target filepath.
        
    Returns:
        A list of strings.
    
    """
    with open(dataset, "r") as data:
        return data.readlines()

def convert(num, maximum):
    new_value = ( (num - 1) / (maximum - 1) ) * (5 - 1) + 1
    return round(new_value)
    
def create_dataset(data):
    
    """
    
    Takes in the filename, reads the content and extracts the text in 
    the sentence, target label of the sentence, and the position of the sentence
    in an abstract.
    
    Args:
        data: filepath of target text file.
        
    Returns:
        A list of dictionaries. Each dictionary with key values containing
        the ID of an abstract, position of the sentence in an abstract,
        text containing the sentence, and the target label.
    
    
    """
    # To store all the lines in a abstract except, 
    # first(abstract ID) and last line (space).
    abstract_lines = []
    abstracts = [] # To store the dictonaries
    position = ['#', 'FIRST', 'SECOND', 'THIRD', 'FOURTH', 'FIFTH']
    for line in data:
        # If the is it the first line or last line, do not add it to the samples.
        if not(line.isspace()  or line.startswith("###")):
            abstract_lines.append(line)
        
        if(line.startswith("###")):
            line_id = line.strip()[3:]
        # If the line is a space('\n'), then all the lines from a
        # abstract has been stored in abstract_lines.
        if(line.isspace()):
            # To store each line into the dictonary.
            for line_no, abst_lines in enumerate(abstract_lines):
                each_line = {} 
                lines = abst_lines.splitlines() # split into seperate lines.
                each_line["ID"] = line_id
                each_line['position'] = position[convert(line_no+1, len(abstract_lines))]
                #each_line['position'] = str(line_no+1) +"_of_"+ str(len(abstract_lines))
                each_line["text"] = lines[0].split("\t")[1].lower() # to get the text of sentence in convert to lower.
                each_line["target"] = lines[0].split("\t")[0] # to get the label
                abstracts.append(each_line) # add dictionary to list of abstracts.
            # reset the sample lines for next abstract.
            abstract_lines = []
    return abstracts 

# Data: https://github.com/Franck-Dernoncourt/pubmed-rct.git

dataset_dir_20k = "pubmed-rct/PubMed_20k_RCT_numbers_replaced_with_at_sign/"

# Load PubMed_20k_RCT_numbers_replaced_with_at_sign dataset
train_abstract_20k = load_data(dataset_dir_20k + "train.txt")
test_abstract_20k = load_data(dataset_dir_20k + "test.txt")
dev_abstract_20k = load_data(dataset_dir_20k + "dev.txt")

# Create a list of dictionaries for 20k datasets.
train_20k = create_dataset(train_abstract_20k)
test_20k = create_dataset(test_abstract_20k)
dev_20k = create_dataset(dev_abstract_20k)

# Create 20K DataFrame
train_20k_df= pd.DataFrame(train_20k)
test_20k_df= pd.DataFrame(test_20k)
dev_20k_df= pd.DataFrame(dev_20k)

# Get the sentence text
train_sentences = train_20k_df.text.to_list()
val_sentences = dev_20k_df.text.to_list()
test_sentences = test_20k_df.text.to_list()

# Split the sentence into characters
train_chars = train_20k_df.text.apply(split)
val_chars = dev_20k_df.text.apply(split)
test_chars = test_20k_df.text.apply(split)

# Create one hot encoded labels: 
one_hot = OneHotEncoder(sparse = False)
train_y_onehot = one_hot.fit_transform(train_20k_df.target.to_numpy().reshape(-1, 1))
val_y_onehot = one_hot.fit_transform(dev_20k_df.target.to_numpy().reshape(-1, 1))
test_y_onehot = one_hot.fit_transform(test_20k_df.target.to_numpy().reshape(-1, 1))

# Create a onehot encoder for position feature
one_hot = OneHotEncoder()
one_hot.fit(np.expand_dims(train_20k_df.position, axis = 1))
joblib.dump(one_hot,"one_hot.joblib")

train_pos = one_hot.transform(np.expand_dims(train_20k_df.position, axis = 1)).toarray()
val_pos = one_hot.transform(np.expand_dims(dev_20k_df.position, axis = 1)).toarray()
test_pos = one_hot.transform(np.expand_dims(test_20k_df.position, axis = 1)).toarray()

# Create label encoder
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
train_y_encoded = labelencoder.fit_transform(train_20k_df.target.to_numpy())
test_y_encoded = labelencoder.fit_transform(test_20k_df.target.to_numpy())
val_y_encoded = labelencoder.fit_transform(dev_20k_df.target.to_numpy())
classes = ["BACKGROUND", "CONCLUSIONS", "METHODS", "OBJECTIVE", "RESULTS"]

physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Initializing vectorizer layers.
vectorizer = layers.experimental.preprocessing.TextVectorization(max_tokens=68000,
                                                                 output_sequence_length=55) # 95% sentences contain 55 words as seen in data analysis.

vectorizer.adapt(train_sentences)
vectorizer_char = layers.experimental.preprocessing.TextVectorization(max_tokens =60,
                                    output_sequence_length = 300, # 95% of the sentences have ~300 chars
                                    name = 'Character_vectorizer')

vectorizer_char.adapt(train_chars.to_list())


# Pretrained Embeddings:  http://nlp.stanford.edu/data/glove.6B.zip

path_to_glove_file = "glove.6B/glove.6B.300d.txt"
voc = vectorizer.get_vocabulary()
word_index = dict(zip(voc, range(len(voc))))

embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))

num_tokens = len(voc) + 2
embedding_dim = 300
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))

# Create TensorFlow dataset
# Training data (PubMed 20K)
train = tf.data.Dataset.from_tensor_slices((train_sentences,
                                            tf.cast(train_pos, dtype = tf.int64)))

train_y = tf.data.Dataset.from_tensor_slices(train_y_onehot)
train_ds = tf.data.Dataset.zip((train, train_y)).batch(32).prefetch(tf.data.AUTOTUNE)

# Validation data (PubMed 20k)
val = tf.data.Dataset.from_tensor_slices((val_sentences, 
                                          tf.cast(val_pos, dtype = tf.int64)))
val_y = tf.data.Dataset.from_tensor_slices(val_y_onehot)
val_ds = tf.data.Dataset.zip((val, val_y)).batch(32).prefetch(tf.data.AUTOTUNE)

# Test Data (PubMed 20k)
test = tf.data.Dataset.from_tensor_slices((test_sentences,
                                           tf.cast(test_pos, dtype = tf.int64)))
test_y = tf.data.Dataset.from_tensor_slices(test_y_onehot)
test_ds = tf.data.Dataset.zip((test, test_y)).batch(32).prefetch(tf.data.AUTOTUNE)



# THE MODEL:
# Pretrained Embedding layer
embedding_layer = layers.Embedding(
    64843,
    300,
    trainable=False,
    name = "Pre_trained"
)

# Character Embeddings layer 
char_layer = layers.Embedding(input_dim = 28,
                             output_dim = 30,
                             name="char_layer")

# Custom Attention layer
class attention(layers.Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
        super(attention, self).build(input_shape)

    def call(self,intput_emb):
        et=tf.keras.backend.squeeze(tf.keras.backend.tanh(tf.keras.backend.dot(intput_emb,self.W)+self.b),axis=-1)
        at=tf.keras.backend.softmax(et)
        at=tf.keras.backend.expand_dims(at,axis=-1)
        return intput_emb*at


# Word Embeddings Model
sent_inputs = layers.Input(shape=[], dtype=tf.string)
sent_vec = vectorizer(sent_inputs)
word_embeddings = embedding_layer(sent_vec)
attention_layer=attention()(word_embeddings)
word_layer_2= layers.Bidirectional(layers.LSTM(128, return_sequences = True))(attention_layer)
word_layer_3= layers.Bidirectional(layers.LSTM(128, return_sequences = False))(word_layer_2)
word_model = tf.keras.Model(inputs=sent_inputs,
                            outputs=word_layer_3)


# Position model
position_inputs = layers.Input(shape=(5,))
pos_model = tf.keras.Model(position_inputs,
                           position_inputs)



concat_layer = layers.Concatenate(name="word_char_pos")([word_model.output,
                                                        pos_model.output])


concat_dense = layers.Dense(128, activation='relu')(concat_layer)

output = layers.Dense(5, activation = 'softmax')(concat_dense)
model = tf.keras.Model(inputs = [word_model.input,
                                 pos_model.input],
                       outputs =  output)

model.compile(loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing= 0.3),
                optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
                metrics = ['accuracy'])

early_stopping  = tf.keras.callbacks.EarlyStopping(monitor= 'val_loss' , 
                                                   patience = 3,
                                                   min_delta = 0.5,
                                                   verbose = 1)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",  
                                                 factor=0.2, 
                                                 patience=3,
                                                 verbose=1, 
                                                 min_lr=1e-7)

model_history = model.fit(train_ds, 
                            epochs = 3,
                            validation_data = val_ds)

model.save_weights("Model/model")