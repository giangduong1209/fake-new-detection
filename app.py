from audioop import ulaw2lin
from re import S
from flask import Flask, render_template, request
import pickle
from tensorflow import keras
import numpy as np
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)


tokenizer = Tokenizer(num_words=15922+1)

def news_predict(input):
    # with open("model.sav", "rb") as file:
    #     pickle_model = pickle.load(file)
    # Ypredict = np.array(input)
    # return Ypredict
    model = keras.models.load_model("model.h5")
    sequences = tokenizer.texts_to_matrix(input, mode="tf-idf")
    tweet_pad = pad_sequences(sequences, maxlen=200, truncating='post', padding='post')
    Ypredict = model.predict(tweet_pad)
    return Ypredict[0][0]

@app.route("/")
def index():
    return render_template("index.html") 

@app.route("/handle", methods=["GET", "POST"])
def handler():
    if request.method == "POST":
        text = request.form["text-input"]
    result = news_predict(text)
    if result >= 0.5:
        prediction = "Tin giả"
    else:
        prediction = "Tin thật"
    return prediction

if __name__ == "__main__":
    app.run(debug=True)
