"""CS410 Text Classification - Twitter Sarcasm
tmcnu3 - Tom McNulty - Captain
mh54 - Michael Huang
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from transformers import TFRobertaForSequenceClassification, TFBertForSequenceClassification
from transformers import RobertaTokenizerFast, BertTokenizerFast
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split

#Read Data
def load_data():
    train = pd.read_json("data/train.jsonl", lines=True)
    test =  pd.read_json("data/test.jsonl", lines=True)
    train = train.sample(frac=1, random_state = 5).reset_index(drop=True) #shuffle dataset
    return train, test

#prepare context for input
def prepare_context(text_lst):
    res = ""
    for i in range(len(text_lst)):
        res += " # CONTEXT " + str(i)+ " # " + text_lst[i]
    return res

#prepare input and return pandas dataframe
def prepare_input(df, test_flag = False):
    #we are adding in the "#RESPONSE#" string to the fron of the input. and then adding in the new context labled tweets right after the response.
    df["input"] = df["response"].apply(lambda x: " # RESPONSE # "+x) + df["context"].apply(prepare_context) 
    df["input"] = " ** TWEET ** " + df["input"] #(not really important, used this to differentiate amongst external data but not needed in the end)
    #we are mapping the 1/0 lables to the lable_numeric column
    if not test_flag:
        df["label_numeric"] = df["label"].map({"SARCASM": 1, "NOT_SARCASM": 0})
    return df

#tokenizes data and return tf dataset object for training
def tokenize_data(texts, tokenizer):
    #tokenize data (prepare inputs, attention masks, and special tokens)
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
    return encodings

#train and return model
def start_train(train_encodings, train_labels, val_encodings, val_labels):
    train_dataset = tf.data.Dataset.from_tensor_slices(( #creates a tensorflow dataset object that can be used to train
        dict(train_encodings),
        train_labels
    ))

    val_dataset = tf.data.Dataset.from_tensor_slices((
        dict(val_encodings),
        val_labels
    ))

    K.clear_session() #initializes random parameters
    model = TFRobertaForSequenceClassification.from_pretrained('roberta-large')
    # this established the learning rate. Adam optimization is a stochastic gradient descent method that is based on
    # adaptive estimation of first-order and second-order moments.
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    #compiles the model to be ready to train
    model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])
    #starts training
    model.fit(train_dataset.shuffle(1000).batch(16), epochs=3, batch_size=16, validation_data=val_dataset.shuffle(100).batch(16))
    return model

#predict on validation set with trained model and evaluate with F1 score
def validate_model(model, tokenizer, val_texts, val_labels):
    val_pred = []
    for text in tqdm(val_texts): #predict each validation row with progress bar
        val_encodings = tokenizer.encode(text,
                                truncation=True,
                                padding=True,
                                max_length=128, 
                                return_tensors="tf")
        # predict() function enables us to predict the labels of the data values on the basis of the trained model.
        logits = model.predict(val_encodings)[0] #outputs logits
        val_pred.append(tf.nn.softmax(logits, axis=1).numpy()[0]) #converts logits to probabilities
    val_pred_labels = np.argmax(val_pred, axis=-1) #outputs higher probable class
    print("validation set f1 score: ", f1_score(val_labels, val_pred_labels)) #prints f1 score

#predict on test set and write output to answer.txt
def predict_on_test(model, tokenizer, test_df):
    test_texts = list(test_df["input"].values)
    test_pred = []
    for text in tqdm(test_texts):
        test_encodings = tokenizer.encode(text,
                                truncation=True,
                                padding=True,
                                max_length=128,
                                return_tensors="tf")
        logits = model.predict(test_encodings)[0]
        test_pred.append(tf.nn.softmax(logits, axis=1).numpy()[0])
    test_pred_labels = np.argmax(test_pred, axis=-1)

    #write test output to answer.txt
    with open('data/answer.txt', 'w') as the_file:
        for i in range(len(test_df)):
            if test_pred_labels[i] == 1:
                the_file.write(test_df.loc[i, "id"]+",SARCASM\n")
            else:
                the_file.write(test_df.loc[i, "id"]+",NOT_SARCASM\n")

def run():
    #load and prepare data
    train, test = load_data() 
    train, test = prepare_input(train), prepare_input(test, True)

    #train-test split
    train_texts, val_texts, train_labels, val_labels = train_test_split(list(train["input"].values), list(train["label_numeric"].values), test_size=.2, random_state = 5)

    #tokenize and train
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
    train_encodings, val_encodings = tokenize_data(train_texts, tokenizer), tokenize_data(val_texts, tokenizer)
    model = start_train(train_encodings, train_labels, val_encodings, val_labels)

    #validate and predict on test and write test output 
    validate_model(model, tokenizer, val_texts, val_labels)
    predict_on_test(model, tokenizer, test)

    #save model
    model.save_pretrained("data/roberta_model")

run()