import tensorflow as tf
from tensorflow.keras import layers
# physical_devices = tf.config.list_physical_devices('GPU') 
# for device in physical_devices:
#     tf.config.experimental.set_memory_growth(device, True)

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

def hybridModel():

    vectorizer = layers.experimental.preprocessing.TextVectorization(max_tokens=68000,
                                                                     output_sequence_length=55)

    embedding_layer = layers.Embedding(
        input_dim = 64843,
        output_dim = 300,
        trainable=False,
        name = "Pre_trained"
    )

    sent_inputs = layers.Input(shape=[], dtype=tf.string) # Text input layer
    # Vectorizer layer
    sent_vec = vectorizer(sent_inputs) 
    # Pre trained word embeddings layers
    word_embeddings = embedding_layer(sent_vec) 
    # Attention layer
    attention_layer=attention()(word_embeddings) 
    # Bi-LSTM layer 
    word_layer_2= layers.Bidirectional(layers.LSTM(128, return_sequences = True))(attention_layer) 
    # Bi-LSTM layer
    word_layer_3= layers.Bidirectional(layers.LSTM(128, return_sequences = False))(word_layer_2) 
    word_model = tf.keras.Model(inputs=sent_inputs, 
                                outputs=word_layer_3)

    # Sentence Position feature
    # one-hot encoded position feature
    position_inputs = layers.Input(shape=(5,)) 
    pos_model = tf.keras.Model(position_inputs,
                               position_inputs)

    # Concatinate one-hot position features with output of Bi-LSTM layer
    concat_layer = layers.Concatenate(name="word_char_pos")([word_model.output, 
                                                            pos_model.output])  

    # Fully connected layer
    concat_dense = layers.Dense(128, activation='relu')(concat_layer) 

    # Model output layer
    output = layers.Dense(5, activation = 'softmax')(concat_dense) 
    model = tf.keras.Model(inputs = [word_model.input,
                                     pos_model.input],
                           outputs =  output)

    model.compile(loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing= 0.3),
                    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
                    metrics = ['accuracy'])

    # Load pretrained weights
    model.load_weights("Model/model").expect_partial() 

    return model


