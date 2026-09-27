import re
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

dataset = load_dataset("stanfordnlp/imdb")
train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])

def no_preprocessing(text): return text
def lowercase(text): return text.lower()
def remove_punctuation(text): return re.sub(r"[^\w\s]", "", text.lower())
def remove_stopwords(text):
    words = re.sub(r"[^\w\s]", "", text.lower()).split()
    return " ".join(w for w in words if w not in ENGLISH_STOP_WORDS)

def run_experiment(preprocessing_function, experiment_name):
    X_train_text=[preprocessing_function(t) for t in train_df["text"]]
    X_test_text=[preprocessing_function(t) for t in test_df["text"]]
    vectorizer=TfidfVectorizer(max_features=50000, ngram_range=(1,2))
    X_train=vectorizer.fit_transform(X_train_text); X_test=vectorizer.transform(X_test_text)
    model=LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, train_df["label"])
    pred=model.predict(X_test)
    y=test_df["label"]
    return {"Experiment":experiment_name,"Accuracy":accuracy_score(y,pred),"Precision":precision_score(y,pred),"Recall":recall_score(y,pred),"F1":f1_score(y,pred),"ConfusionMatrix":confusion_matrix(y,pred).tolist()}

experiments=[(no_preprocessing,"E1 - Raw Text"),(lowercase,"E2 - Lowercase"),(remove_punctuation,"E3 - Lowercase + Punctuation Removal"),(remove_stopwords,"E4 - Lowercase + Punctuation + Stopword Removal")]
results=[run_experiment(fn,name) for fn,name in experiments]
print(pd.DataFrame([{k:v for k,v in r.items() if k!="ConfusionMatrix"} for r in results]).round(4))
for r in results: print(r["Experiment"], r["ConfusionMatrix"])
