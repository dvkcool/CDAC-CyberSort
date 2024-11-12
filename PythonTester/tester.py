import pandas as pd
import json
from sklearn.metrics import accuracy_score  # Use sklearn here, even though the package is scikit-learn
from ollama import Client
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize the Ollama client
client = Client(host='http://192.168.60.21:11434')

# Load your test data (assuming a DataFrame format)
test_data = pd.read_csv("test_data_cl.csv")

# Function to call the Llama model and get predictions
def get_prediction(row):
    complaint, tc, tsubc = row['complaint'], row['true_category'], row['true_subcategory']
    complaint_cleaned = re.sub(r'[^\w\s]', '', complaint)
    complaint_cleaned = complaint_cleaned.replace('\n', ' ').replace('\r', ' ')
    try:
        response = client.chat(model='gemma2:2b', messages=[
            {
                'role': 'user',
                'content': f'Categories for classifying cyber complaints include: category: Women/Child Related Crime (subcategories:  Child Pornography/CSAM, Rape/Gang Rape-Sexually Abusive Content, Sexually Explicit Act, Sexually Obscene Material),  category: Financial Fraud Crimes (subcategories: Debit/Credit Card Fraud, SIM Swap Fraud, Internet Banking-Related Fraud, Business Email Compromise, E-Wallet Frauds, Fraud Call/Vishing, Demat Fraud, UPI-Related Frauds, Aadhaar Enabled Payment System Fraud, Email Phishing, Cheating by Impersonation, Cryptocurrency Crime, Online Job Fraud, Online Matrimonial Fraud), and category:  Other Cyber Crime (subcategories: Fake/Impersonating Profile, Profile Hacking, Cyber Bullying, Email Hacking, Damage to Computer Systems, Unauthorized Access, Ransomware, Cyber Terrorism, Denial of Service Attacks, Data Breaches, Identity Theft, Malware Attacks). \n\n Based on above can you return the category and subcategory of following complaint: "{complaint}"\nReturn *only* the result in json format containing "subcategory": "X", "category": "Y" without any other explanation or text',        
            },
        ])
        response_text = response['message']['content']
        predicted_category = ' & '.join(response_text.split('"category":')[1].split('"')[1].split('"'))
        predicted_subcategory = ' & '.join(response_text.split('"subcategory":')[1].split('"')[1].split('"'))
    except Exception:
        predicted_category, predicted_subcategory = "X", "Y"

    # Print the prediction and true values as each task completes
    print(f"{predicted_category} , {predicted_subcategory} , {tc} , {tsubc}")
    
    return tc, tsubc, predicted_category, predicted_subcategory

# Lists to hold true and predicted values for accuracy calculation
true_categories = []
predicted_categories = []
true_subcategories = []
predicted_subcategories = []

# Run the get_prediction function in parallel
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(get_prediction, row) for _, row in test_data.iterrows()]
    
    for future in as_completed(futures):
        tc, tsubc, predicted_category, predicted_subcategory = future.result()
        true_categories.append(tc)
        true_subcategories.append(tsubc)
        predicted_categories.append(predicted_category)
        predicted_subcategories.append(predicted_subcategory)

# Calculate accuracy for category and subcategory
category_matches = [
    1 if predicted in true else 0
    for true, predicted in zip(true_categories, predicted_categories)
]
subcategory_matches = [
    1 if predicted in true else 0
    for true, predicted in zip(true_subcategories, predicted_subcategories)
]

# Calculate the accuracy as the mean of matches
category_accuracy = sum(category_matches) / len(category_matches)
subcategory_accuracy = sum(subcategory_matches) / len(subcategory_matches)

print(f"Category Accuracy (contains match): {category_accuracy * 100:.2f}%")
print(f"Subcategory Accuracy (contains match): {subcategory_accuracy * 100:.2f}%")
