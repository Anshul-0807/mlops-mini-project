from flask import Flask, render_template,request
import mlflow
from preprocessing_utility import normalize_text
import dagshub
import pickle
import os
import pandas as pd

import numpy as np
import re
import nltk
import string
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer


mlflow.set_tracking_uri("https://dagshub.com/Anshul-0807/mlops-mini-project.mlflow")
dagshub.init(repo_owner='Anshul-0807', repo_name='mlops-mini-project', mlflow=True)

# load the model from model registry

model_name = "my_model"
model_version = 1

model_uri = f'models:/{model_name}/{model_version}'
model = mlflow.pyfunc.load_model(model_uri)

vectorizer = pickle.load(open('models/vectorizer.pkl','rb'))



app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html',result=None)

@app.route('/predict', methods=['POST'])
def predict():

    text = request.form['text']

    # clean
    text = normalize_text(text)

    # bow
    features = vectorizer.transform([text])

    # # Convert sparse matrix to DataFrame
    # features_df = pd.DataFrame.sparse.from_spmatrix(features)
    # features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])

    # prediction
    result = model.predict(features)

    # show
    return render_template('index.html', result=result[0]) 
   

app.run(debug=True)