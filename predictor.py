import re
import os
import pickle

import pandas as pd
import nltk
nltk.download('wordnet',quiet=True)
nltk.download('omw-1.4',quiet=True)
nltk.download('punkt',quiet=True)

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import NearestNeighbors

LE = LabelEncoder()
TFIDF = TfidfVectorizer()

data_file = os.path.join(os.getcwd(), "yummly.json")
models = {
    "svc": os.path.join(os.getcwd(), "models", "svc.model"),
    "nn": os.path.join(os.getcwd(), "models", "nn.model")
}
RAW_DATA = pd.read_json(data_file)

def get_models():
    if not os.path.exists(models['svc']) or not os.path.exists(models['nn']):
        models_fldr = os.path.join(os.getcwd(), "models")
        if not os.path.exists(models_fldr):
            os.mkdir(models_fldr)

        svc, nn = fit_prediction_models()
        with open(models['svc'], 'wb') as svc_f, open(models['nn'], 'wb') as nn_f:
            pickle.dump(svc, svc_f)
            pickle.dump(nn, nn_f)
        
        return svc, nn
    else:
        _, _ = get_yummly_data()
        with open(models['svc'], 'rb') as svc_f, open(models['nn'], 'rb') as nn_f:
            return pickle.load(svc_f), pickle.load(nn_f)


def fit_prediction_models():
    x, y = get_yummly_data()
    
    svc = CalibratedClassifierCV(LinearSVC()).fit(x, y)
    nn = NearestNeighbors(n_neighbors=5, metric='cosine').fit(x)

    return svc, nn

def find_cuisine(svc, ingredient):
    processed_ingreds = pre_process_ingredients(ingredient)
    inp_X = TFIDF.transform([processed_ingreds])
    probs = svc.predict_proba(inp_X)

    scores = [(cuisine,  score) for cuisine, score in zip(LE.classes_, probs[0])]
    scores.sort(key=lambda x: x[1])

    return scores[-1][0], round(scores[-1][1], 2)

def find_closest(nn, ingredient, N):
    processed_ingreds = pre_process_ingredients(ingredient)
    inp_X = TFIDF.transform([processed_ingreds])
    distances, indices = nn.kneighbors(inp_X, n_neighbors=N)

    closest = [(RAW_DATA.iloc[ind, 0], 1 - dist) for ind, dist in zip(indices[0], distances[0])]
    closest = [{"id": str(id), "score": round(score, 2)} for id, score in closest]
    return closest

def get_yummly_data():
    df = RAW_DATA.copy()
    df.ingredients = df.ingredients.map(pre_process_ingredients)
    
    x = TFIDF.fit_transform(df.ingredients)
    y = LE.fit_transform(df.cuisine)

    return x, y

def pre_process_ingredients(ingredients):
    # lemma = WordNetLemmatizer()
    # ingredients = [ingredient.lower() for ingredient in ingredients]
    # ingredients = [re.sub(r" +", "_", i.strip()) for i in ingredients]
    # ingredients = [lemma.lemmatize(i) for i in ingredients]    
    # ingredients = ",".join(ingredients)
    lemma = WordNetLemmatizer()
    ingredients = [ingredient.lower() for ingredient in ingredients]
    ingredients = [word_tokenize(i) for i in ingredients]
    ingredients = [" ".join([lemma.lemmatize(j) for j in i]) for i in ingredients]
    ingredients = [re.sub(r" +", "_", i.strip()) for i in ingredients]
    ingredients = ",".join(ingredients)

    return ingredients
