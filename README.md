# nlp-preprocessing-sentiment

Reproducibility repository for the poster **The Impact of Text Preprocessing on Sentiment Classification**.

## Question
How does text preprocessing affect sentiment-classification performance?

## Dataset
IMDb Movie Reviews (`stanfordnlp/imdb`): 25,000 train and 25,000 test reviews.

## Conditions
E1 raw; E2 lowercase; E3 lowercase + punctuation removal; E4 lowercase + punctuation removal + stopword removal.

## Fixed setup
TF-IDF (`max_features=50000`, `ngram_range=(1,2)`) + Logistic Regression (`max_iter=1000`, `random_state=42`).

## Reproduce
Install `requirements.txt` and run `python experiment.py`.

## Caveat
Findings are specific to this dataset/model/representation setup. No statistical significance test is reported.
