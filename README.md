# 🧠 Emotion Detection from Text using NLP

An NLP-based Emotion Detection project that classifies human emotions from text using traditional Machine Learning techniques. The project performs text preprocessing, converts text into numerical vectors using **Bag of Words** and **TF-IDF**, and compares different machine learning models to identify the best-performing classifier.

---

## 📌 Project Overview

Understanding emotions expressed in text is an important task in Natural Language Processing (NLP). This project builds an emotion classification system using a labeled dataset containing text and their corresponding emotions.

The workflow includes:

- Data Cleaning
- Text Preprocessing
- Feature Extraction
- Model Training
- Model Evaluation
- Accuracy Comparison

---

## 🚀 Features

- Converts emotion labels into numerical values
- Cleans text by removing:
  - Punctuation
  - Numbers
  - Emojis
  - Stopwords
- Converts text to lowercase
- Uses **Bag of Words (CountVectorizer)**
- Uses **TF-IDF Vectorizer**
- Trains multiple machine learning models
- Compares model performance

---

## 📂 Dataset

Dataset Format:

| Text | Emotion |
|------|---------|
| i am happy today | joy |
| i feel lonely | sadness |
| i am very angry | anger |

The dataset is loaded using:

```python
pd.read_csv("train.txt", sep=';', header=None,
            names=['text','emotion'])
```

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK
- Scikit-Learn

---

## 📖 Text Preprocessing

The following preprocessing steps are applied:

### 1. Convert text to lowercase

Example:

```
I LOVE NLP
```

↓

```
i love nlp
```

---

### 2. Remove punctuation

Example:

```
I love NLP!!!
```

↓

```
I love NLP
```

---

### 3. Remove numbers

Example:

```
I have 2 books
```

↓

```
I have books
```

---

### 4. Remove emojis

Example:

```
I am happy 😊
```

↓

```
I am happy
```

---

### 5. Remove stopwords

Example:

```
I am very happy today
```

↓

```
happy today
```

---

## 🔤 Feature Extraction

### 1. Bag of Words

Text is converted into a sparse matrix using

```python
CountVectorizer()
```

Example:

```
I love NLP
I love AI
```

Vocabulary:

```
love
nlp
ai
```

Vectors:

```
[1 1 0]
[1 0 1]
```

---

### 2. TF-IDF

Text is also converted using

```python
TfidfVectorizer()
```

TF-IDF gives more importance to informative words while reducing the weight of common words.

---

## 🤖 Machine Learning Models

### Multinomial Naive Bayes

Used with:

- Bag of Words
- TF-IDF

```python
MultinomialNB()
```

---

### Logistic Regression

```python
LogisticRegression(max_iter=1000)
```

This model achieved the best performance among all tested models.


## 📈 Project Workflow

```
Dataset
   │
   ▼
Load Data
   │
   ▼
Text Cleaning
   │
   ▼
Lowercase Conversion
   │
   ▼
Remove Punctuation
   │
   ▼
Remove Numbers
   │
   ▼
Remove Emojis
   │
   ▼
Remove Stopwords
   │
   ▼
Feature Extraction
(Bag of Words / TF-IDF)
   │
   ▼
Train-Test Split
   │
   ▼
Model Training
   │
   ▼
Prediction
   │
   ▼
Accuracy Evaluation
```

---

## ▶️ How to Run

Clone the repository

```bash
git clone https://github.com/your-username/emotion-detection-nlp.git
```

Install dependencies

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn
```

Run the notebook or Python script.

---


## 🎯 Conclusion

This project demonstrates a complete NLP pipeline for text emotion classification. After experimenting with different feature extraction techniques and machine learning models, **Logistic Regression with TF-IDF** provided the best classification performance, showing that linear models often perform strongly on sparse text features.

---

## 👨‍💻 Author

**Vishal Sorout**
