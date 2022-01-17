import tensorflow as tf
from tensorflow.keras import layers
physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)



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

def transformer():
    vectorizer = layers.experimental.preprocessing.TextVectorization(max_tokens=68000,
                                                                     output_sequence_length=55)

    vectorizer_char = layers.experimental.preprocessing.TextVectorization(max_tokens =60,
                                        output_sequence_length = 300,
                                        name = 'Character_vectorizer')

    embedding_layer = layers.Embedding(
        input_dim = 64843,
        output_dim = 300,
        trainable=False,
        name = "Pre_trained"
    )

    char_layer = layers.Embedding(input_dim = 28,
                                 output_dim = 30,
                                 name="char_layer")


    # THE MODEL:
    # Word Embeddings Model
    sent_inputs = layers.Input(shape=[], dtype=tf.string)
    sent_vec = vectorizer(sent_inputs)
    word_embeddings = embedding_layer(sent_vec)
    word_layer_2= layers.Bidirectional(layers.LSTM(128, return_sequences = True))(word_embeddings)
    attention_layer=attention()(word_layer_2)
    word_model = tf.keras.Model(inputs=sent_inputs,
                                outputs=attention_layer)

    # Character Embeddings Model
    char_inputs = layers.Input(shape=[], dtype=tf.string)
    char_vectorizer = vectorizer_char(char_inputs)
    char_embeddings = char_layer(char_vectorizer)
    char_layer_1= layers.Bidirectional(layers.LSTM(128, return_sequences=True))(char_embeddings) 
    char_model = tf.keras.Model(inputs=char_inputs,
                              outputs=char_layer_1)

    # Position model
    position_inputs = layers.Input(shape=(460,), dtype = tf.int64)
    pos_dense = layers.Dense(64, activation = 'relu')(position_inputs)
    pos_model = tf.keras.Model(position_inputs, pos_dense)

    word_char_layer = layers.Concatenate(axis =1)([attention_layer,
                                            char_layer_1])

    word_char_lstm = layers.Bidirectional(layers.LSTM(128))(word_char_layer)
    word_char_dropout = layers.Dropout(0.5)(word_char_lstm)

    hybrid_layer = layers.Concatenate(name="word_char_pos")([word_char_dropout,
                                                            pos_model.output])

    output = layers.Dense(5, activation = 'softmax')(hybrid_layer)
    model = tf.keras.Model(inputs = [word_model.input,
                                     char_model.input,
                                     pos_model.input],
                           outputs =  output)

    model.compile(loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing= 0.3),
                    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
                    metrics = ['accuracy'])

    model.load_weights("Model/Model").expect_partial()

    return model


