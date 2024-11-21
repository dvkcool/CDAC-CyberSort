import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import pickle

# Load the dataset (assumes data has 'complaint', 'category', 'subcategory' columns)
data = pd.read_csv('cleaned_dataset.csv')

# Data Preprocessing
data['complaint'] = data['complaint'].str.lower().str.replace(r'[^\w\s]', ' ').str.strip()

# Feature Extraction with TF-IDF
tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
X = tfidf.fit_transform(data['complaint'])

# Train-Test Split for Category
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X, data['true_category'], test_size=0.2, random_state=42)

# Train-Test Split for Subcategory
X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(X, data['true_subcategory'], test_size=0.2, random_state=42)

# Train the Category Model
category_model = LogisticRegression(max_iter=1000)
category_model.fit(X_train_cat, y_train_cat)

# Train the Subcategory Model
subcategory_model = LogisticRegression(max_iter=1000)
subcategory_model.fit(X_train_sub, y_train_sub)

# Evaluate the Category Model
y_pred_cat = category_model.predict(X_test_cat)
cat_accuracy = accuracy_score(y_test_cat, y_pred_cat)
cat_report = classification_report(y_test_cat, y_pred_cat, output_dict=True)
print("Category Model Accuracy:", cat_accuracy)
print("Category Classification Report:\n", classification_report(y_test_cat, y_pred_cat))

# Evaluate the Subcategory Model
y_pred_sub = subcategory_model.predict(X_test_sub)
sub_accuracy = accuracy_score(y_test_sub, y_pred_sub)
sub_report = classification_report(y_test_sub, y_pred_sub, output_dict=True)
print("Subcategory Model Accuracy:", sub_accuracy)
print("Subcategory Classification Report:\n", classification_report(y_test_sub, y_pred_sub))

# Save the Models and TF-IDF Vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
with open('category_model.pkl', 'wb') as f:
    pickle.dump(category_model, f)
with open('subcategory_model.pkl', 'wb') as f:
    pickle.dump(subcategory_model, f)

print("Models and Vectorizer saved successfully.")

