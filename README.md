# English Hate Speech Detection

English Hate Speech Detection using Machine Learning and TF-IDF with Error Analysis.

## 1. Project Overview

This project develops a supervised machine learning approach to classify English social media comments into two categories: **Hate** and **Non-Hate**.

The dataset consists of 1,427 English comments collected from Threads across five domains:

- American Election
- Islamophobia
- Racism (Black/White)
- Cyberbullying
- LGBTQ+ Issues

The project compares three machine learning classifiers and a majority-class baseline to evaluate their performance in hate speech detection.

## 2. Dataset Information

The dataset contains 1,427 comments with the following distribution:

| Class | Number of Comments |
|---|---:|
| Hate | 394 |
| Non-Hate | 1,033 |
| **Total** | **1,427** |

The dataset contains five attributes:

- ID
- Domain
- Original Comment
- Cleaned Comment
- Label

The cleaned comments are used as input features, while the Label column serves as the target variable.

## 3. Machine Learning Models

Four classification models are implemented:

1. Majority-Class Classifier (baseline)
2. Logistic Regression
3. Complement Naive Bayes
4. Linear Support Vector Machine (SVM)

TF-IDF is used for feature extraction with a maximum of 5,000 features and a word n-gram range of (1, 2).

## 4. Experimental Setup

The dataset is divided into training and testing sets using an 80:20 stratified split with a random state of 42.

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Linear SVM is selected for further error analysis based on its overall performance.

## 5. Model Performance

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Majority-Class Classifier | 0.7238 | 0.0000 | 0.0000 | 0.0000 |
| Logistic Regression | 0.8427 | 0.9722 | 0.4430 | 0.6087 |
| Complement Naive Bayes | 0.8427 | 0.8542 | 0.5190 | 0.6457 |
| Linear SVM | **0.8741** | 0.8644 | **0.6456** | **0.7391** |

Linear SVM achieves the highest accuracy, recall and F1-score among the evaluated models.

## 6. Error Analysis

Error analysis is performed on Linear SVM predictions to identify:

- False Positives (FP)
- False Negatives (FN)

A total of 36 misclassified comments are exported into `Linear SVM Error Analysis.csv` for further inspection.

## 7. Project Files

| File | Description |
|---|---|
| `threadcomment.py` | Main Python implementation |
| `Linear SVM Error Analysis.csv` | Misclassified comments |
| `Figure_1.png` | Confusion matrix visualization |
| `linear_svm_model.pkl` | Saved Linear SVM model |
| `tfidf_vectorizer.pkl` | Saved TF-IDF vectorizer |
| `Threads English Hate Speech Dataset.csv` | Threads dataset |

## 8. Requirements

The project uses Python and the following libraries:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

Install the required libraries using:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## 9. How to Run

1. Clone or download this repository.
2. Ensure the dataset CSV file is available at the file path specified in the Python script.
3. Install the required Python libraries.
4. Run the following command:

```bash
python threadcomment.py
```

The script performs dataset inspection, data preparation, TF-IDF feature extraction, model training, evaluation and error analysis.

## 10. Limitations

This implementation evaluates the models using an in-domain train-test split from a single Threads dataset. It does not currently perform cross-dataset evaluation.

## 11. Development Tools

- Python
- Visual Studio Code
- Scikit-learn
- GitHub
