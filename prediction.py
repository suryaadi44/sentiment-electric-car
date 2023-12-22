import pickle
import re
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from fastapi import FastAPI
from pydantic import BaseModel


def regexOperation(tweet):
    # Remove Non-ascii
    res = re.sub(r"[^\x00-\x7F]+", " ", tweet)

    # Remove url
    res = re.sub(r"http[s]?\:\/\/.[a-zA-Z0-9\.\/\_?=%&#\-\+!]+", " ", res)
    res = re.sub(r"pic.twitter.com?.[a-zA-Z0-9\.\/\_?=%&#\-\+!]+", " ", res)
    # Remove mention
    res = re.sub(r"\@([\w]+)", " ", res)

    # Remove hashtag
    # res = re.sub(r'\#([\w]+)',' ', res)
    # Proccessing hashtag (split camel case)
    res = re.sub(r"((?<=[a-z])[A-Z]|[A-Z](?=[a-z]))", " \\1", res)
    # res = re.sub(r'([A-Z])(?<=[a-z]\1|[A-Za-z]\1(?=[a-z]))',' \\1', res)

    # Remove special character
    res = re.sub(r'[!$%^&*@#()_+|~=`{}\[\]%\-:";\'<>?,.\/]', " ", res)
    # Remove number
    res = re.sub(r"[0-9]+", "", res)
    # Remove consecutive alphabetic characters
    res = re.sub(r"([a-zA-Z])\1\1", "\\1", res)
    # Remove consecutive whitespace
    res = re.sub(" +", " ", res)
    # Remove trailing and leading whitespace
    res = re.sub(r"^[ ]|[ ]$", "", res)

    # Convert to lower case
    res = res.lower()

    return res


def tokenize(tweet):
    return word_tokenize(tweet)


stopwordFactory = StopWordRemoverFactory()
stopwords = stopwordFactory.get_stop_words()

def remove_stopwords(tweet):
    return [word for word in tweet if word not in stopwords]


stemmerFactory = StemmerFactory()
stemmer = stemmerFactory.create_stemmer()

def stemming(tweet):
    return [stemmer.stem(word) for word in tweet]


def preprocess(tweet):
    tweet = regexOperation(tweet)
    tweet = tokenize(tweet)
    tweet = remove_stopwords(tweet)
    tweet = stemming(tweet)
    tweet = " ".join(tweet)
    return tweet

filename = "svm_model.sav"
loaded_model = pickle.load(open(filename, "rb"))

app = FastAPI()

class PredictionRequest(BaseModel):
    text: str


class PredictionsRequest(BaseModel):
    texts: list[str]


class PredictionResponse(BaseModel):
    prediction: int
    predictionStr: str
    score: list[float]


class PredictionsResponse(BaseModel):
    predictions: list[PredictionResponse]


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    tweet = request.text
    tweet = preprocess(tweet)
    prediction = loaded_model.predict([tweet])
    score = loaded_model.predict_proba([tweet])

    prediction_str = "Positif" if prediction[0] == 1 else "Negatif"

    return PredictionResponse(
        prediction=prediction, predictionStr=prediction_str, score=score[0]
    )


@app.post("/predict_batch", response_model=PredictionsResponse)
def predict_batch(request: PredictionsRequest):
    tweets = request.texts
    tweets = [preprocess(tweet) for tweet in tweets]
    predictions = loaded_model.predict(tweets)
    scores = loaded_model.predict_proba(tweets)

    predictions_str = ["Positif" if prediction == 1 else "Negatif" for prediction in predictions]

    return PredictionsResponse(
        predictions=[
            PredictionResponse(
                prediction=prediction, predictionStr=prediction_str, score=score
            )
            for prediction, prediction_str, score in zip(predictions, predictions_str, scores)
        ]
    )

