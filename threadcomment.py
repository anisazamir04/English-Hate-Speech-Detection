import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

# ==========================================
# ENGLISH HATE SPEECH DETECTION
# Assignment 3b: NLP Model Implementation
# ==========================================

# 1. Import libraries
import pandas as pd
import numpy as np

# 2. Load dataset
file_path = "Threads English Hate Speech Dataset.csv"

df = pd.read_csv(file_path)

# 3. Display dataset information
print("===== DATASET INFORMATION =====")

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
df.info()

# 4. Check missing values
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# 5. Check duplicate records
print("\n===== DUPLICATE RECORDS =====")
print("Duplicate rows:", df.duplicated().sum())

# 6. Check label distribution
print("\n===== LABEL DISTRIBUTION =====")
print(df["Label"].value_counts())

print("\nLabel Percentages:")
print(df["Label"].value_counts(normalize=True) * 100)

# 7. Check domain distribution
print("\n===== DOMAIN DISTRIBUTION =====")
print(df["Domain"].value_counts())

print("\n===== DATA LOADING COMPLETED =====")



# ==========================================
# STEP 2: DATA PREPARATION
# ==========================================

print("\n===== DATA PREPARATION =====")

# Check empty cleaned comments
empty_comments = df["Cleaned Comment"].fillna("").str.strip().eq("").sum()

print("Empty cleaned comments:", empty_comments)

# Check valid labels
print("\nUnique Labels:")
print(df["Label"].unique())

expected_labels = {"Hate", "Non-Hate"}

if set(df["Label"].unique()) != expected_labels:
    raise ValueError("Unexpected labels found.")

if empty_comments > 0:
    raise ValueError("Empty cleaned comments detected.")

# Define features and target
X = df["Cleaned Comment"]
y = df["Label"]

# Split dataset (80:20)
train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["Label"]
)

# Separate training and testing data
X_train = train_df["Cleaned Comment"]
X_test = test_df["Cleaned Comment"]

y_train = train_df["Label"]
y_test = test_df["Label"]

# Display results
print("\n===== TRAIN-TEST SPLIT =====")

print("Total samples:", len(df))
print("Training samples:", len(train_df))
print("Testing samples:", len(test_df))

print("\nTraining Label Distribution:")
print(y_train.value_counts())

print("\nTesting Label Distribution:")
print(y_test.value_counts())

print("\n===== DATA PREPARATION COMPLETED =====")



# ==========================================
# STEP 3: TF-IDF FEATURE EXTRACTION
# ==========================================

print("\n===== TF-IDF FEATURE EXTRACTION =====")

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

# Fit on training data and transform training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transform testing data using the same vectorizer
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Display TF-IDF results
print("Training TF-IDF Shape:", X_train_tfidf.shape)
print("Testing TF-IDF Shape:", X_test_tfidf.shape)

print("Number of TF-IDF Features:",
      len(tfidf_vectorizer.get_feature_names_out()))

print("\n===== TF-IDF COMPLETED =====")



# ==========================================
# STEP 4: MODEL TRAINING
# ==========================================

print("\n===== MODEL TRAINING =====")

# Initialize models
models = {
    "Majority-Class Classifier": DummyClassifier(
        strategy="most_frequent"
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Complement Naive Bayes": ComplementNB(),

    "Linear SVM": LinearSVC(
        random_state=42
    )
}

# Train each model
for model_name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    print(f"{model_name}: Training completed")

print("\n===== ALL MODELS TRAINED SUCCESSFULLY =====")



# ==========================================
# STEP 5: MODEL EVALUATION
# ==========================================

print("\n===== MODEL EVALUATION =====")

results = []

for model_name, model in models.items():

    # Predict testing data
    y_pred = model.predict(X_test_tfidf)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred,
        pos_label="Hate",
        zero_division=0
    )
    recall = recall_score(
        y_test, y_pred,
        pos_label="Hate",
        zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred,
        pos_label="Hate",
        zero_division=0
    )

    # Store results
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

    print(f"\n===== {model_name} =====")
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-Score:", round(f1, 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(
        y_test,
        y_pred,
        labels=["Hate", "Non-Hate"]
    ))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=["Hate", "Non-Hate"],
        zero_division=0
    ))

# Create comparison table
results_df = pd.DataFrame(results)

print("\n===== MODEL COMPARISON =====")
print(results_df.to_string(index=False))

print("\n===== MODEL EVALUATION COMPLETED =====")



# ==========================================
# STEP 6: CONFUSION MATRIX VISUALIZATION
# ==========================================

print("\n===== CONFUSION MATRIX VISUALIZATION =====")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes = axes.flatten()

for i, (model_name, model) in enumerate(models.items()):

    # Predict testing data
    y_pred = model.predict(X_test_tfidf)

    # Generate confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=["Hate", "Non-Hate"]
    )

    # Plot confusion matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Hate", "Non-Hate"],
        yticklabels=["Hate", "Non-Hate"],
        ax=axes[i]
    )

    axes[i].set_title(model_name)
    axes[i].set_xlabel("Predicted Label")
    axes[i].set_ylabel("Actual Label")

plt.tight_layout()
plt.show()

print("\n===== VISUALIZATION COMPLETED =====")



# ==========================================
# STEP 7: ERROR ANALYSIS
# ==========================================

print("\n===== ERROR ANALYSIS (LINEAR SVM) =====")

# Get Linear SVM model
svm_model = models["Linear SVM"]

# Predict testing data
y_pred_svm = svm_model.predict(X_test_tfidf)

# Create error analysis dataframe
error_analysis_df = test_df.copy()

error_analysis_df["Actual Label"] = y_test.values
error_analysis_df["Predicted Label"] = y_pred_svm

# Identify prediction errors
error_analysis_df["Error Type"] = np.where(
    (error_analysis_df["Actual Label"] == "Hate") &
    (error_analysis_df["Predicted Label"] == "Non-Hate"),
    "False Negative",
    np.where(
        (error_analysis_df["Actual Label"] == "Non-Hate") &
        (error_analysis_df["Predicted Label"] == "Hate"),
        "False Positive",
        "Correct"
    )
)

# Extract incorrect predictions only
errors_df = error_analysis_df[
    error_analysis_df["Error Type"] != "Correct"
].copy()

# Display error counts
print("\nError Distribution:")
print(errors_df["Error Type"].value_counts())

# Display all misclassified comments
print("\n===== MISCLASSIFIED COMMENTS =====")

print(
    errors_df[
        [
            "ID",
            "Domain",
            "Original Comment",
            "Cleaned Comment",
            "Actual Label",
            "Predicted Label",
            "Error Type"
        ]
    ].to_string(index=False)
)

print("\nTotal Misclassified Comments:", len(errors_df))

print("\n===== ERROR ANALYSIS COMPLETED =====")



# ==========================================
# STEP 8: EXPORT ERROR ANALYSIS RESULTS
# ==========================================

print("\n===== EXPORTING ERROR ANALYSIS =====")

# Export misclassified comments to CSV
errors_df.to_csv(
    "Linear SVM Error Analysis.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Error analysis successfully exported.")
print("File: Linear SVM Error Analysis.csv")
print("Total records exported:", len(errors_df))

print("\n===== EXPORT COMPLETED =====")



# ==========================================
# STEP 9: SAVE TRAINED MODEL
# ==========================================

print("\n===== SAVING TRAINED MODEL =====")

import joblib

# Save Linear SVM model
joblib.dump(
    models["Linear SVM"],
    "linear_svm_model.pkl"
)

# Save TF-IDF Vectorizer
joblib.dump(
    tfidf_vectorizer,
    "tfidf_vectorizer.pkl"
)

print("Linear SVM model saved successfully.")
print("TF-IDF Vectorizer saved successfully.")

print("\n===== MODEL SAVING COMPLETED =====")



# ==========================================
# STEP 10: TEST NEW COMMENTS
# ==========================================

print("\n===== TESTING NEW COMMENTS =====")

import joblib
import re

# Load the saved model and TF-IDF vectorizer
loaded_model = joblib.load("linear_svm_model.pkl")
loaded_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Function to clean new comments
def clean_new_comment(comment):
    comment = comment.lower()
    comment = re.sub(r"http\S+|www\S+", "", comment)
    comment = re.sub(r"[^\w\s]", " ", comment)
    comment = re.sub(r"\s+", " ", comment)
    return comment.strip()

# Enter new comments for testing
new_comments = [
    "People from this community deserve respect.",
    "I hate these people and they should not be allowed here.",
    "Everyone deserves equal rights and treatment."
]

# Apply text cleaning
cleaned_comments = [
    clean_new_comment(comment) for comment in new_comments
]

# Convert comments into TF-IDF features
new_comments_tfidf = loaded_vectorizer.transform(cleaned_comments)

# Predict labels
predictions = loaded_model.predict(new_comments_tfidf)

# Display results
print("\n===== PREDICTION RESULTS =====")

for comment, prediction in zip(new_comments, predictions):
    print("\nOriginal Comment:", comment)
    print("Predicted Label:", prediction)

print("\n===== TESTING COMPLETED =====")