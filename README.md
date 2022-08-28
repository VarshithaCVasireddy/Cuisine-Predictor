# Cusine Predictor
## Author: Varshitha Choudary Vasireddy

## Description of the project:
The goal of the project is to create an application that take a list of ingredients from a user and attempts to predict the type of cuisine and similar meals. Consider a chef who has a list of ingredients and would like to change the current meal without changing the ingredients. The steps to develop the application should proceed as follows.

1. Pre-train or index all necessary classifiers using the existing datasets.
2. Ask the user to input all the ingredients that they are interested in.
3. Use the model to predict the type of cuisine and tell the user.
4. Find the top N closest foods (you can define N). Return the IDs of those dishes to the user. If a dataset does not have IDs associated with them you may add them arbitrarily.

## Setting up the initial installations
Below packages are to be installed in the project's virtual environment to successfully run the project. The below command has to be followed.
~~~
pipenv install nltk, scikit-learn ,pandas , pytest
~~~

## Structure of project

2 python files are present in this project, **project2.py** and **predictor.py**

### **predictor.py**
Below Packages imported are in order to run this program code and to use their functionalities
~~~
import re
import os
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
~~~

yummly.json file is read using pandas.read_json file and is called RAW_DATA. <br/>

- 5 functions are written in this program

### **get_yummly_data**
The RAW_DATA is made a copy and is loaded into dataframe. And the ingredients column of the data undergoes **pre_process_ingredients** function processing, which is described below. And TfidVectorization is done on ingredients data and Label encoding is done on cuisine data which is used as label. These two vectorizations are fitted and transformed. The vectorized ingredients and cuisine data is returned. The code is as below

~~~
def get_yummly_data():
    df = RAW_DATA.copy()
    df.ingredients = df.ingredients.map(pre_process_ingredients)
    
    x = TFIDF.fit_transform(df.ingredients)
    y = LE.fit_transform(df.cuisine)
    return x,y
~~~

Referred: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html, https://www.youtube.com/watch?v=0_LPy-LtYqI

### **pre_process_ingredients**
Ingredients data is taken as input. All the data is turned into lower case. The words are tokenized and are lemmatized to get base form and the words representing same ingredient is joined by underscore. I got this idea by seeing cuisine data, where one value have underscore present between the words instead of space. Later all ingredients are joined with a comman in between. Ingredients cleaning of data is done and it is returned.
~~~
def pre_process_ingredients(ingredients):
    lemma = WordNetLemmatizer()
    ingredients = [ingredient.lower() for ingredient in ingredients]
    ingredients = [word_tokenize(i) for i in ingredients]
    ingredients = [" ".join([lemma.lemmatize(j) for j in i]) for i in ingredients]
    ingredients = [re.sub(r" +", "_", i.strip()) for i in ingredients]
    ingredients = ",".join(ingredients)

    return ingredients
~~~
Referred: https://www.youtube.com/watch?v=0_LPy-LtYqI

### **fit_prediction_models**
get_yummly_data function is executed. The processed ingredients and label data i.e cuisine data is given into CalibratedClassifierCV model where the base estimator is LinearSVC() and the x,y data are fitted into that model. Tested many machine learning models like Naive Bayes, Random forest, etc, where out of those I found that LinearSVC is giving better prediction score. To find the nearest dishes close to the ingredients given I used NearestNeighbors, it is an unsupervised learner for implementing neighbor searches. So giving just the ingredients data is sufficient, no need of giving labeled data. To calculate the distance between the ingredients I used the "cosine" distance as it was mentioned in class as it is best to use. 

~~~
def fit_prediction_models():
    x, y = get_yummly_data()

    svc = CalibratedClassifierCV(LinearSVC()).fit(x, y)
    nn = NearestNeighbors(n_neighbors=5, metric='cosine').fit(x)
~~~

Referred: https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html, https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors.kneighbors, https://stats.stackexchange.com/questions/337050/how-to-find-nearest-neighbors-using-cosine-similarity-for-all-items-from-a-large

### **find_cuisine**
pre_process_ingredients functionality is carried out on ingredient data. Then the vectorization will be done on it. To predict the probability score predict_proba is used. scores tuple is created where the cuisine name is taken from label encoding class and score of probability is taken. The scores are sorted in ascending order and to get the highest probability acquired cuisine the last tuple member is accessed and the label of the tuple is accessed. The probability score is rounded to 2 digits after decimal.
~~~
def find_cuisine(svc, ingredient):
    processed_ingreds = pre_process_ingredients(ingredient)
    inp_X = TFIDF.transform([processed_ingreds])
    probs = svc.predict_proba(inp_X)

    scores = [(cuisine,  score) for cuisine, score in zip(LE.classes_, probs[0])]
    scores.sort(key=lambda x: x[1])

    return scores[-1][0], round(scores[-1][1], 2)
~~~

### **find_closest**
pre_process_ingredients functionality is carried out on ingredient data. Then the vectorization will be done on it. From the nearestneighbors model the distances and the indices are taken. The indices value is accessed using the position value given by the model and the similarity between the ingredients is 1 - the distance. Distance taken here is cosine similarity distance. The id is to be given as string and the score is the similarity score which is given after rounding it to 2 digits after decimal.
~~~
def find_closest(nn, ingredient, N):
    processed_ingreds = pre_process_ingredients(ingredient)
    inp_X = TFIDF.transform([processed_ingreds])
    distances, indices = nn.kneighbors(inp_X, n_neighbors=N)

    closest = [(RAW_DATA.iloc[ind, 0], 1 - dist) for ind, dist in zip(indices[0], distances[0])]
    closest = [{"id": str(id), "score": round(score, 2)} for id, score in closest]
    return closest
~~~

### **Path to store models**
Path to store models is created as below.
~~~
models = {
    "svc": os.path.join(os.getcwd(), "models", "svc.model"),
    "nn": os.path.join(os.getcwd(), "models", "nn.model")
}
~~~

### **get_models**
For the persistence of the models that I used I wrote below code. This helps in models not be retrained always. Firstly I checked if the models are existing in the path, then I dummped the models. I used python built-in persistence model, namely pickle. Pickle dump and load functionality are used to do this. If models are not present then "dump" is used to create models. If models are already present then they are loaded.
~~~
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
~~~


### **project2.py**
Packages imported in this file are 
- argparse
- json
- predictor python file is imported to use it's functions.
<br/>
By using the argparser package we create a object "parser" and input arguments are added by add_argument method
Below are the input arguments that are added by the add_argument method.
- -- N
- -- ingredient
#### --N
This is an integer given, it specifies the top N closest foods that user wants. The code for it is.

~~~
parser.add_argument('--N',required = True, type = int,help='Top N closest meals that are to be revealed')
~~~

### **-- ingredient**
This is a string type, and user gives the ingredients as input.
~~~
parser.add_argument('--ingredient', required = True, type = str, action = "append", help='ingredients are to be inputted')
~~~

These arguments are passed to main method.
### **main**
A try catch block is used to catch errors. An empty dictionary is created and is named as output. In the try block the functions from predictor program is called.
<br/>
- This calls the function predictor.find_cuisine which finds the cuisine from the ingredients given. Arguments passed are svc, ingredient. Cusine and the prediction score is collected into the output dictionary that is created before.
<br/>
- predictor.find_closest function is called in order to find the top N dishes by considering the ingredients given. Arguments passed are nn, ingredient and N. The results are taken into the dictionary with the key as "closest".
<br/>
- The output where data is collected is changed into json format and is printed onto the screen. This is the output that is expected.
~~~
 print(json.dumps(output, indent=2))
~~~
If something went wrong then the except block error is written.


## **Tests**
In the tests I tested all the 5 functions functionalities of predictor.py file.

### **test_pre_process_ingredients**
The pre_process_ingredients function is tested here. Ingredients are sent and the output is checked with the expected. In this function the ingredients data is cleaned and set properly as we want it to be kept in the models for the next stages.
~~~
def test_pre_process_ingredients():
    actual_ingredients = pre_process_ingredients(["Rices flour", "scrambled eggs"])
    assert actual_ingredients == "rice_flour,scrambled_egg"
~~~

### **test_get_yummly_data**
get_yummly_data function is called in this test method. The return variables are taken. And the labels i.e cuisine and ingredient corpus vectorized shape is taken and is checked with the expected ones.
~~~
def test_get_yummly_data():
    actual_x, actual_y = get_yummly_data()
    assert actual_x.shape == (39774, 6880)
    assert actual_y.shape == (39774,)
~~~
### **models()**
In this test fit_prediction_models function is called. svc and nn are returned. I used pytest fixture with the scope of module in order to save time, because this function of models returning will be used in other test functions. 
~~~
@pytest.fixture(scope="module")
def models():
    svc, nn = fit_prediction_models()

    return svc, nn
~~~
### **test_fit_prediction_models**
The svc and nn models are checked if they are not NoneType are not. So see if the models are working and are giving some output.
~~~
def test_fit_prediction_models(models):
    assert models[0] is not NoneType
    assert models[1] is not NoneType
~~~
### **test_find_cuisine**
The models are given as argument. The svc model is taken as argument and it's functionality is checked. And the output is checked with the expected output i.e cuisine and the score.
~~~
def test_find_cuisine(models):
    actual_cuisine, actual_score = find_cuisine(models[0], ["Rices flour", "scrambled eggs"])
    
    assert actual_cuisine == "indian"
    assert actual_score == 0.36
~~~
### **test_find_closest**
Now here the closest dishes are checked. The nn model is given to get the neighbors i.e the closest dishes. I am giving that I need 5 neighbors and I checked the output length to see if I got 5 dishes.
~~~
def test_find_closest(models):
    actual_closest = find_closest(models[1], ["Rices flour", "scrambled eggs"], 5)
    
    assert len(actual_closest) == 5
~~~

## Assumptions/Bugs
- The results are based on models that are used, they might not give exact results.
- The dishes which matches the ingredients are given from all the cuisines, not from the cuisine that is predicted first based on ingredients.

## **Steps to Run project1**

- **Step1**  
Clone the project directory using below command

~~~json
git clone https://github.com/VarshithaCVasireddy/Cuisine_Predictor
~~~
After cloning go to that folder and run below commands
<br/>

-**Step2**
Run below command to install pipenv
~~~
pip install pipenv
~~~

- **Step3**  
Navigate to directory that we cloned from git and run the below command to install dependencies

~~~
pipenv install nltk, scikit-learn, pandas , pytest
~~~

- **Step4**  
Then run the below command by providing URL
~~~
pipenv run python project2.py --N 5 --ingredient paprika
                                    --ingredient banana 
                                    --ingredient "rice krispies" 
~~~
- **Step5** 

Then run the below command to test the testcases. 

~~~
 pipenv run python -m pytest -v
~~~
