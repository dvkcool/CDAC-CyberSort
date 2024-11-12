import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load your dataset from a CSV file
# Make sure the CSV has columns named 'complaint', 'category', and 'subcategory'
df = pd.read_csv('cleaned_dataset.csv')

# Step 1: Preprocess the text (vectorize it)
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['complaint'])

# Combine category and subcategory into one label for simplicity
df['combined_label'] = df['true_category'] + " - " + df['true_subcategory']

# Encode the labels
le = LabelEncoder()
y = le.fit_transform(df['combined_label'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Define and train the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Step 3: Make predictions
y_pred = clf.predict(X_test)

# Step 4: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
