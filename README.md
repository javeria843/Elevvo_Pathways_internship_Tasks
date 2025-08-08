**Sentiment Analysis on Amazon Product Reviews**
This project analyzes Amazon product reviews to classify them as positive or negative using Natural Language Processing (NLP) techniques and Machine Learning models. The performance of Naive Bayes is compared with Logistic Regression.

🎯 Objective
To build a binary sentiment classifier that:

Preprocesses and cleans raw text data.

Transforms text into numerical features.

Trains and evaluates classification models.

Compares performance between Naive Bayes and Logistic Regression.

📌 Workflow
Data Loading

Amazon Product Reviews dataset (Kaggle)

IMDb Reviews dataset can also be used as an alternative.

Text Preprocessing (NLTK / spaCy)

Lowercasing all text.

Removing stopwords.

Tokenizing words.

Optional: Lemmatization or stemming.

Feature Engineering

Convert cleaned text to numerical features using:

TF-IDF Vectorizer

or CountVectorizer

Model Training

Naive Bayes (MultinomialNB)

Logistic Regression (max_iter=1000)

Evaluation

Model accuracy comparison.

Detailed classification report (Precision, Recall, F1-score).

🛠 Tools & Libraries
Pandas → Data loading & handling

NLTK / spaCy → Text preprocessing

Scikit-learn → Feature extraction, model training, evaluation

📊 Expected Outcome
Well-preprocessed dataset ready for ML.

Trained models for binary sentiment classification.

Accuracy comparison between Naive Bayes and Logistic Regression.
