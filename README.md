# 📰 Fake News Analyzer & Credibility Checker

An AI-powered web application that detects whether a news article is **Fake** or **Real** using Natural Language Processing (NLP) and Machine Learning.

## 🚀 Live Demo

(Add Streamlit deployment link here)

## 📌 Project Overview

Fake News Analyzer is a machine learning project built to identify misleading or fabricated news articles. The application processes textual news content, converts it into numerical features using TF-IDF Vectorization, and predicts authenticity using a Logistic Regression model.

The project demonstrates an end-to-end NLP workflow including data preprocessing, feature engineering, model training, evaluation, and deployment.

## 🛠️ Technologies Used

* Python
* Streamlit
* Scikit-learn
* Pandas
* NumPy
* NLTK
* Joblib

## 📊 Dataset Information

* Total Articles: 44,058
* Fake News Articles
* Real News Articles
* Text-based classification dataset

## ⚙️ Machine Learning Pipeline

### Data Preprocessing

* Text cleaning
* Lowercasing
* Punctuation removal
* URL removal
* Stopword removal

### Feature Engineering

* TF-IDF Vectorization
* 5,000 Features

### Model

* Logistic Regression

## 📈 Model Performance

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 98.91% |
| Precision | 98%+   |
| Recall    | 98%+   |
| F1 Score  | 98%+   |

## 🌟 Key Features

* Real-time Fake News Detection
* User-friendly Streamlit Interface
* Confidence Score Prediction
* NLP-based Text Processing
* End-to-End Deployment

## 📂 Project Structure

```text
fake-news-analyzer/
│
├── app.py
├── models/
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   └── fake_news_analysis.ipynb
│
├── data/
├── README.md
└── requirements.txt


## ▶️ Run Locally

bash
pip install -r requirements.txt
streamlit run app.py


## 👩‍💻 Author

Khushbu Kumari

B.Tech CSE (AI & ML)

Machine Learning & AI Enthusiast

GitHub:
https://github.com/khushbukumari-dev

