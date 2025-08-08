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

**TASK 2: 📰 News Category Classification**
Classify news articles into categories like Sports, Business, Politics, and Technology using NLP and machine learning.

**📌 Project Overview**
This project tackles multiclass classification using the AG News dataset. The goal is to build a robust pipeline that preprocesses text, extracts meaningful features, and trains a classifier to predict the correct news category.

**🧠 Key Concepts Covered**
Multiclass classification

Text preprocessing (tokenization, stopword removal, lemmatization)

Feature engineering using TF-IDF or word embeddings

Model training with Logistic Regression, Random Forest, or SVM

**🛠️ Tools & Libraries**
pandas for data handling

scikit-learn for preprocessing and modeling

Optional: XGBoost, LightGBM, or Keras for advanced models

**🔍 Workflow Summary**
Data Loading Load AG News dataset and inspect class distribution.

Text Preprocessing

Tokenize text

Remove stopwords

Apply lemmatization

Feature Extraction

Use TF-IDF or pre-trained word embeddings

Optionally, try dimensionality reduction (e.g., PCA)

Model Training

Train a multiclass classifier

Evaluate using accuracy, precision, recall, and F1-score

Bonus Visualizations

Bar plots or word clouds for frequent words per category

Neural network implementation using Keras (optional)

**📊 Results**
Model performance metrics and visualizations will be added here once training is complete.
