import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from model import transformer
import tensorflow as tf
from tensorflow.keras import layers
from preprocess import create_data

def classify(data, model):
    classes = ["BACKGROUND", "CONCLUSIONS", "METHODS", "OBJECTIVE", "RESULTS"]
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


model = transformer()
cont = 'y'
while cont == 'y':
    abstract = input("\nEnter the Abstract: \n\n")
    result = classify(abstract, model)
    
    for r in result:
        print(r['label'], " : ", r['sentence'], "\n")

# if __name__ == "__main__":
#     model= transformer()
#     try:
#         cont = 'y'
#         while cont == 'y':
#             abstract = input("\nEnter the Abstract: \n\n")
#             result = classify(abstract, model)
            
#             for r in result:
#                 print(r['label'], " : ", r['sentence'], "\n")

#             cont = str(input("\nWant to skim another unstructured abstract? [y/n] : ").lower())

#     except:

#         print("Error !")

