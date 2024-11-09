import pandas as pd
import json
from sklearn.metrics import accuracy_score  # Use sklearn here, even though the package is scikit-learn
from ollama import Client
import re

# Initialize the Ollama client
client = Client(host='http://localhost:11434')

# Load your test data (assuming a DataFrame format)
# test_data.csv contains columns: "complaint", "true_category", "true_subcategory"
test_data = pd.read_csv("test_data_up.csv")

# Function to call the Llama model and get predictions
def get_prediction(complaint, tc, tsubc):
    # gemma2:2b
    # response = client.chat(model='llama3.1', messages=[
    complaint_cleaned = re.sub(r'[^\w\s]', '', complaint)
    complaint_cleaned = complaint_cleaned.replace('\n', ' ').replace('\r', ' ')
    response = client.chat(model='gemma2:2b', messages=[
        {
            'role': 'user',
            'content': f'Consider that there are following categories: Women/Child Related Crime,Financial Fraud Crimes, Other Cyber Crime and following sub categories: Child Pornography/Child Sexual Abuse    Material (CSAM), Rape/Gang Rape-Sexually Abusive Content, Sale, Publishing and Transmitting Obscene Material/Sexually Explicit Material, Debit/Credit Card Fraud, SIM Swap Fraud, I   nternet Banking-Related Fraud, Business Email Compromise/Email Takeover, E-Wallet Related Frauds, Fraud Call/Vishing, Demat/Depository Fraud, UPI-Related Frauds, Aadhaar Enabled Pa   yment System (AEPS) Fraud, Email Phishing, Cheating by Impersonation, Fake/Impersonating Profile, Profile Hacking/Identity Theft, Provocative Speech of Unlawful Acts, Impersonating    Email, Intimidating Email, Online Job Fraud, Online Matrimonial Fraud, Cyber Bullying/Stalking/Sexting, Email Hacking, Damage to Computer Systems, Tampering with Computer Source D   ocuments, Defacement/Hacking, Unauthorized Access/Data Breach, Online Cyber Trafficking, Online Gambling/Betting Fraud, Ransomware, Cryptocurrency Crime, Cyber Terrorism, Any Other    Cyber Crime, Targeted scanning/probing of critical networks/systems., Compromise of critical systems/information., Unauthorised access to IT systems/data., Defacement of websites    or unauthorized changes, such as inserting malicious code or  external links., Malicious code attacks (e.g., virus, worm, Trojan, Bots, Spyware, Ransomware, Crypto  miners)., Attac   ks on servers (Database, Mail, DNS) and network devices (Routers)., Identity theft, spoofing, and phishing attacks., Denial of Service (DoS) and Distributed Denial of Service (DDoS   ) attacks., Attacks on critical infrastructure, SCADA, operational technology systems, and wireless  networks., Attacks on applications (e.g., E-Governance, E-Commerce)., Data brea   ches., Data leaks., Attacks on Internet of Things (IoT) devices and associated systems, networks, and  servers., Attacks or incidents affecting digital payment systems., Attacks vi   a malicious mobile apps., Fake mobile apps., Unauthorised access to social media accounts., Attacks or suspicious activities affecting cloud computing systems, servers, software,     and applications., Attacks or malicious/suspicious activities affecting systems related to Big Data,  Blockchain, virtual assets, and robotics., Attacks on systems related to Artif   icial Intelligence (AI) and Machine Learning (ML)., Backdoor attacks., Disinformation or misinformation campaigns., Supply chain attacks., Cyber espionage., Zero-day exploits., Pas   sword attacks., Web application vulnerabilities., Hacking, Malware attacks. \n\n Based on above can you return the category and subcategory of following complaint: "{complaint}"\nReturn *only* the result in json format containing "subcategory": "X", "category": "Y" without any other explanation or text',
        },
    ])
    # print(response)
    # Parse the response text for category and subcategory information
    response_text = response['message']['content']
    #print(response_text)
    print(response_text.split('"category":')[1].split('"')[1], ",", response_text.split('"subcategory":')[1].split('"')[1], ",",  tc, ",", tsubc)
    # prediction = json.loads(response_text.strip())
    return response_text.split('"category":')[1].split('"')[1], response_text.split('"subcategory":')[1].split('"')[1]

# Lists to hold true and predicted values for accuracy calculation
true_categories = []
predicted_categories = []
true_subcategories = []
predicted_subcategories = []

# Loop over each complaint and get predictions
for index, row in test_data.iterrows():
    true_categories.append(row['true_category'])
    true_subcategories.append(row['true_subcategory'])
    
    predicted_category, predicted_subcategory = get_prediction(row['complaint'], row['true_category'], row['true_subcategory'])
    predicted_categories.append(predicted_category)
    predicted_subcategories.append(predicted_subcategory)

# Calculate accuracy for category and subcategory
# Calculate custom accuracy based on substring matching
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
