import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from model import hybridModel
import tensorflow as tf
from tensorflow.keras import layers
from preprocess import createData

def classify(data, model):

    """
    This function makes predictions for each sentences in the input text with its'
    appropriate headings or sections. 

    Args : 
    
            data  - Unstructured medical abstracts.
            model - Model that trained to classify abstract sentences.

    Returns:

            A list of dictionary that contains the sentence and label of
            the sentence. 

            Example : 

                results = [
                    {
                        label : BACKGROUND, 
                        sentence : Most cancer patients are treated with some combination of surgery, radiation, and chemotherapy. 
                    }.
                    {
                        label : METHODS,
                        sentence : We retrospectively analyzed the data of 867 COVID-19 cases. 
                    }
                ]

    """


    classes = ["BACKGROUND", "CONCLUSIONS", "METHODS", "OBJECTIVE", "RESULTS"]
    data = createData(data)
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



#Only runs when this file is executed directly. 
if __name__ == "__main__":

    model= hybridModel()

    try:
        cont = 'y'
        while cont == 'y' or cont =='':
            abstract = input("\nEnter the Abstract: \n\n")
            result = classify(abstract, model)
            
            for r in result:
                print(r['label'], " : ", r['sentence'], "\n")

            cont = str(input("\nWant to skim another unstructured abstract? [Y/n] : ").lower())

    except:

        print("Please Enter only unstructured medical abstracts with atleat 5 lines!")

