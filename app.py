from flask import Flask, render_template, request
import pickle
from tensorflow import keras
import numpy as np
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)


def tokenizer_fun(raw_text):
    tokenizer = Tokenizer(num_words=15922+1)
    sequences = tokenizer.texts_to_matrix(raw_text, mode="tf-idf")
    tweet_pad = pad_sequences(sequences, maxlen=200, truncating='post', padding='post')
    return tweet_pad


def news_predict(input):
    model = keras.models.load_model("model.h5")
    Ypredict = model.predict(input)
    return Ypredict[0][0]

@app.route("/")
def index():
    return render_template("index.html") 

@app.route("/handle")
def demo_template():
    return render_template("handler.html")

@app.route("/handler", methods=["GET", "POST"])
def handler():
    if request.method == "POST":
        text = request.form["text-input"]
        tokenizer_text = tokenizer_fun(text)
        result = news_predict(tokenizer_text)
        if result >= 0.5:
            prediction = "Tin giả"
        else:
            prediction = "Tin thật"
    return render_template("handler.html", result=prediction)

if __name__ == "__main__":
    app.run(debug=True)
