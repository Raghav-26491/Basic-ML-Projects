# 📰 Fake News Detection

This project aims to detect **fake or misleading news articles** using various Machine Learning models.  
It analyzes text-based features extracted from a news dataset and classifies whether a given news piece is *real* or *fake*.

---

## 🚀 Project Overview

With the massive spread of misinformation online, automated fake news detection has become crucial.  
This project implements and compares multiple supervised learning algorithms to classify news articles as **True** or **Fake**.

---

## 🧠 Models Implemented

The following machine learning models were trained and evaluated:

- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Gradient Boosting Classifier**

Each model was tested and compared based on accuracy, precision, recall, and F1-score.

---

## 📊 Dataset

- **Source:** [Kaggle Fake News Dataset](https://www.kaggle.com/c/fake-news)
- **Structure:**
  - `id` – Unique identifier for each article  
  - `title` – Title of the news article  
  - `text` – Full content of the news article  
  - `label` – Target variable (1 = Fake, 0 = Real)

---

## 🧩 Workflow

1. **Data Loading & Cleaning**  
   - Removal of null values and duplicates  
   - Text preprocessing (stopword removal, stemming, tokenization)

2. **Feature Extraction**  
   - Text vectorization using **TF-IDF**

3. **Model Training & Evaluation**  
   - Training multiple classifiers  
   - Evaluation using metrics such as Accuracy, Precision, Recall, and F1-Score  
   - Comparison of model performance

4. **Prediction**  
   - Input a custom news text and classify it as *Real* or *Fake*

---

## 🛠️ Technologies Used

| Category | Tools / Libraries |
|-----------|-------------------|
| Programming Language | Python |
| Data Handling | pandas, numpy |
| Text Processing | nltk, re |
| Feature Extraction | scikit-learn (TF-IDF Vectorizer) |
| Modeling | scikit-learn (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting) |
| Visualization | matplotlib, seaborn |

---

## 📈 Results

- **Best Model:** *Gradient Boosting Classifier* (highest accuracy and F1-score)  
- Demonstrates the effectiveness of ensemble methods in handling text classification tasks.

