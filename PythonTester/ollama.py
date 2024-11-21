import pandas as pd
from transformers import pipeline
from sklearn.metrics import precision_recall_fscore_support

# Define predefined categories and subcategories

CATEGORIES = {
    "Women/Child Related Crime": [
        "Child Pornography/Child Sexual Abuse Material (CSAM)",
        "Rape/Gang Rape-Sexually Abusive Content",
        "Sale, Publishing and Transmitting Obscene Material/Sexually Explicit Material"
    ],
    "Financial Fraud Crimes": [
        "Debit/Credit Card Fraud",
        "SIM Swap Fraud",
        "Internet Banking-Related Fraud",
        "Business Email Compromise/Email Takeover",
        "E-Wallet Related Frauds",
        "Fraud Call/Vishing",
        "Demat/Depository Fraud",
        "UPI-Related Frauds",
        "Aadhaar Enabled Payment System (AEPS) Fraud"
    ],
    "Other Cyber Crime": [
        "Email Phishing",
        "Cheating by Impersonation",
        "Fake/Impersonating Profile",
        "Profile Hacking/Identity Theft",
        "Provocative Speech of Unlawful Acts",
        "Impersonating Email",
        "Intimidating Email",
        "Online Job Fraud",
        "Online Matrimonial Fraud",
        "Cyber Bullying/Stalking/Sexting",
        "Email Hacking",
        "Damage to Computer Systems",
        "Tampering with Computer Source Documents",
        "Defacement/Hacking",
        "Unauthorized Access/Data Breach",
        "Online Cyber Trafficking",
        "Online Gambling/Betting Fraud",
        "Ransomware",
        "Cryptocurrency Crime",
        "Cyber Terrorism"
    ]
}


def categorize_complaints(complaints: pd.Series) -> pd.DataFrame:
    # Initialize a lightweight model for text classification
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    results = []
    for complaint in complaints:
        # Predict category
        categories = list(CATEGORIES.keys())
        category_result = classifier(complaint, categories)
        category = category_result["labels"][0]  # Top predicted category
        
        # Predict subcategory
        subcategories = CATEGORIES[category]
        subcategory_result = classifier(complaint, subcategories)
        subcategory = subcategory_result["labels"][0]  # Top predicted subcategory
        
        results.append({
            "complaint": complaint,
            "predicted_category": category,
            "predicted_subcategory": subcategory,
        })

    return pd.DataFrame(results)

def calculate_metrics(df: pd.DataFrame):
    """
    Calculate precision, recall, and F1-score for categories and subcategories.
    
    Args:
        df (pd.DataFrame): DataFrame with true and predicted labels.
        
    Returns:
        None: Prints metrics to console.
    """
    # Calculate metrics for categories
    category_metrics = precision_recall_fscore_support(
        df["true_category"], df["predicted_category"], average="weighted"
    )
    print("\nCategory Metrics:")
    print(f"Precision: {category_metrics[0]:.2f}")
    print(f"Recall:    {category_metrics[1]:.2f}")
    print(f"F1 Score:  {category_metrics[2]:.2f}")

    # Calculate metrics for subcategories
    subcategory_metrics = precision_recall_fscore_support(
        df["true_subcategory"], df["predicted_subcategory"], average="weighted"
    )
    print("\nSubcategory Metrics:")
    print(f"Precision: {subcategory_metrics[0]:.2f}")
    print(f"Recall:    {subcategory_metrics[1]:.2f}")
    print(f"F1 Score:  {subcategory_metrics[2]:.2f}")

# Main Execution
if __name__ == "__main__":
    # Load the input CSV file
    # CSV should have columns: 'complaint', 'true_category', 'true_subcategory'
    input_csv = "cleaned_dataset.csv"  # Change this to your file path
    data = pd.read_csv(input_csv)

    data_subset = data.head(1000)
    
    # Categorize complaints
    predictions = categorize_complaints(data_subset["complaint"])
    
    # Merge predictions with true labels
    results = pd.concat([data, predictions], axis=1)
    
    # Calculate and print precision, recall, and F1 scores
    calculate_metrics(results)
    
    # Save results to a CSV file
    results.to_csv("categorized_complaints.csv", index=False)
    print("\nResults saved to 'categorized_complaints.csv'")
