from flask import Flask, render_template, request
import pickle
from tensorflow import keras
import numpy as np
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import nltk
from tensorflow.keras.models import load_model
nltk.download('punkt')


app = Flask(__name__)

cols = ['user_name', 'post_message', 'timestamp_post', 'num_like_post', 'num_comment_post', 'num_share_post', 'label']
df = pd.read_csv("./public_train.csv", usecols = cols ,encoding = "utf-8")
df["post_message"] = df["post_message"].str.lower() # Chuyển text về lowercasse
df = df.dropna()

def create_corpus(df):
    corpus = []
    for tweet in tqdm(df['post_message']):
        words = [word.lower() for word in word_tokenize(tweet)]
        corpus.append(words)
    return corpus

df['post_message']=df['post_message'].apply(str)
corpus = create_corpus(df)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_matrix(corpus, mode='tfidf')
tweet_pad = pad_sequences(sequences, maxlen=200, truncating='post', padding='post')



def predictor(text_input):
    model = load_model("model.h5")
    prediction = model.predict(text_input)
    return prediction[6]

@app.route("/")
def index():
    return render_template("index.html") 

@app.route("/handle")
def demo_template():
    return render_template("handler.html")

@app.route("/handler", methods=["GET", "POST"])
def handler():
    text = request.form["text-input"]
    sequences = tokenizer.texts_to_matrix(text, mode="tfidf")
    tweet_pad = pad_sequences(sequences, maxlen=200, truncating='post', padding='post')
    result = predictor(tweet_pad) 
    if result >= 0.06:
        prediction = "Tin thật"
    else:
        prediction = "Tin giả"
    return render_template("handler.html", result=prediction)

if __name__ == "__main__":
    app.run(debug=True)
