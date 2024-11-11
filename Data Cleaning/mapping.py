import pandas as pd

# Load your data from the CSV file
df = pd.read_csv("merged_file.csv")

# Define the mapping of subcategories to the guidelines (Subcategory Names)
subcategory_to_guideline = {
    'Cyber Bullying  Stalking  Sexting': 'Cyber Bullying/Stalking/Sexting',
    'Fraud CallVishing': 'Fraud Call/Vishing',
    'Online Gambling  Betting': 'Online Gambling/Betting Fraud',
    'Online Job Fraud': 'Online Job Fraud',
    'UPI Related Frauds': 'UPI-Related Frauds',
    'Internet Banking Related Fraud': 'Internet Banking-Related Fraud',
    'Other': 'Any Other Cyber Crime',
    'Profile Hacking Identity Theft': 'Profile Hacking/Identity Theft',
    'DebitCredit Card FraudSim Swap Fraud': 'Debit/Credit Card Fraud, SIM Swap Fraud',
    'EWallet Related Fraud': 'E-Wallet Related Frauds',
    'Data Breach/Theft': 'Unauthorized Access/Data Breach',
    'Cheating by Impersonation': 'Cheating by Impersonation',
    'Denial of Service (DoS)/Distributed Denial of Service (DDOS) attacks': 'Denial of Service (DoS) and Distributed Denial of Service (DDoS) attacks',
    'FakeImpersonating Profile': 'Fake/Impersonating Profile',
    'Cryptocurrency Fraud': 'Cryptocurrency Crime',
    'Malware Attack': 'Malware attacks',
    'Business Email CompromiseEmail Takeover': 'Business Email Compromise/Email Takeover',
    'Email Hacking': 'Email Hacking',
    'Hacking/Defacement': 'Defacement/Hacking',
    'Unauthorised AccessData Breach': 'Unauthorized Access/Data Breach',
    'SQL Injection': 'Unauthorized Access/Data Breach',
    'Provocative Speech for unlawful acts': 'Provocative Speech of Unlawful Acts',
    'Ransomware Attack': 'Ransomware',
    'Cyber Terrorism': 'Cyber Terrorism',
    'Tampering with computer source documents': 'Tampering with Computer Source Documents',
    'DematDepository Fraud': 'Demat/Depository Fraud',
    'Online Trafficking': 'Online Cyber Trafficking',
    'Online Matrimonial Fraud': 'Online Matrimonial Fraud',
    'Website DefacementHacking': 'Defacement/Hacking',
    'Damage to computer computer systems etc': 'Damage to Computer Systems',
    'Impersonating Email': 'Impersonating Email',
    'EMail Phishing': 'Email Phishing',
    'Ransomware': 'Ransomware',
    'Intimidating Email': 'Intimidating Email',
    'Against Interest of sovereignty or integrity of India': 'Any Other Cyber Crime',
    'Computer Generated CSAM/CSEM': 'Child Pornography/Child Sexual Abuse Material (CSAM)',
    'Cyber Blackmailing & Threatening': 'Any Other Cyber Crime',
    'Sexual Harassment': 'Rape/Gang Rape-Sexually Abusive Content'
}

# Define the mapping for categories when sub_category is empty
category_to_subcategory_when_empty = {
    'Online and Social Media Related Crime': 'Cyber Bullying/Stalking/Sexting',
    'Online Financial Fraud': 'Fraud Call/Vishing',
    'Online Gambling  Betting': 'Online Gambling/Betting Fraud',
    'RapeGang Rape RGRSexually Abusive Content': 'Rape/Gang Rape-Sexually Abusive Content',
    'Any Other Cyber Crime': 'Any Other Cyber Crime',
    'Cyber Attack/ Dependent Crimes': 'Denial of Service (DoS) and Distributed Denial of Service (DDoS) attacks',
    'Cryptocurrency Crime': 'Cryptocurrency Crime',
    'Sexually Explicit Act': 'Sexually Explicit Act',
    'Sexually Obscene material': 'Sexually Obscene material',
    'Hacking  Damage to computercomputer system etc': 'Damage to Computer Systems',
    'Cyber Terrorism': 'Cyber Terrorism',
    'Child Pornography CPChild Sexual Abuse Material CSAM': 'Child Pornography/Child Sexual Abuse Material (CSAM)',
    'Online Cyber Trafficking': 'Online Cyber Trafficking',
    'Ransomware': 'Ransomware',
    'Report Unlawful Content': 'Any Other Cyber Crime',
    'Crime Against Women & Children': 'Child Pornography/Child Sexual Abuse Material (CSAM)'
}

# Define the mapping of subcategories to the main categories
subcategory_to_main_category = {
    'Women/Child Related Crime': [
        'Child Pornography/Child Sexual Abuse Material (CSAM)',
        'Rape/Gang Rape-Sexually Abusive Content',
        'Sexually Explicit Act',
        'Sexually Obscene material'
    ],
    'Financial Fraud Crimes': [
        'Debit/Credit Card Fraud',
        'SIM Swap Fraud',
        'Debit/Credit Card Fraud, SIM Swap Fraud',
        'Internet Banking-Related Fraud',
        'Business Email Compromise/Email Takeover',
        'E-Wallet Related Frauds',
        'Fraud Call/Vishing',
        'Demat/Depository Fraud',
        'UPI-Related Frauds',
        'Aadhaar Enabled Payment System (AEPS) Fraud',
        'Email Phishing',
        'Cheating by Impersonation',
        'Cryptocurrency Crime',
        'Online Job Fraud',
        'Online Matrimonial Fraud'
    ],
    'Other Cyber Crime': [
        'Fake/Impersonating Profile',
        'Profile Hacking/Identity Theft',
        'Provocative Speech of Unlawful Acts',
        'Impersonating Email',
        'Intimidating Email',
        'Cyber Bullying/Stalking/Sexting',
        'Email Hacking',
        'Damage to Computer Systems',
        'Tampering with Computer Source Documents',
        'Defacement/Hacking',
        'Unauthorized Access/Data Breach',
        'Online Cyber Trafficking',
        'Online Gambling/Betting Fraud',
        'Ransomware',
        'Cyber Terrorism',
        'Targeted scanning/probing of critical networks/systems',
        'Compromise of critical systems/information',
        'Unauthorised access to IT systems/data',
        'Defacement of websites or unauthorized changes',
        'Malicious code attacks',
        'Attacks on servers and network devices',
        'Identity theft, spoofing, and phishing attacks',
        'Denial of Service (DoS) and Distributed Denial of Service (DDoS) attacks',
        'Attacks on critical infrastructure',
        'Attacks on applications',
        'Data breaches',
        'Data leaks',
        'Attacks on Internet of Things (IoT) devices',
        'Attacks or incidents affecting digital payment systems',
        'Attacks via malicious mobile apps',
        'Fake mobile apps',
        'Unauthorized access to social media accounts',
        'Attacks or suspicious activities affecting cloud systems',
        'Attacks or activities affecting systems related to Big Data, Blockchain, virtual assets, robotics',
        'Attacks on AI and ML systems',
        'Backdoor attacks',
        'Disinformation or misinformation campaigns',
        'Supply chain attacks',
        'Cyber espionage',
        'Zero-day exploits',
        'Password attacks',
        'Web application vulnerabilities',
        'Hacking',
        'Malware attacks'
    ]
}

# Function to map subcategory to guideline subcategory
def map_subcategory(row):
    subcategory = row['sub_category']
    
    # Check for empty sub_category (NaN or empty string)
    if pd.isna(subcategory) or subcategory.strip() == "":
        # If sub_category is empty, map it based on category
        category = row['category']
        return category_to_subcategory_when_empty.get(category, 'Unspecified')  # Default if no match
        
    # If sub_category is not empty, map it based on the predefined mapping
    return subcategory_to_guideline.get(subcategory, 'Unspecified')

# Function to map subcategory to the main category
def map_to_main_category(subcategory):
    # Find the main category based on the subcategory
    for main_category, subcategories in subcategory_to_main_category.items():
        if subcategory in subcategories:
            return main_category
    return "Other Cyber Crime"  # Default category if no match is found

df['true_subcategory'] = df.apply(map_subcategory, axis=1)
df['true_category'] = df['true_subcategory'].apply(map_to_main_category)

# Keep only the required columns: 'true_category', 'true_subcategory', and 'complaint'
df = df[['true_category', 'true_subcategory', 'crimeaditionalinfo']]

# Rename the 'crimeaditionalinfo' column to 'complaint'
df.rename(columns={'crimeaditionalinfo': 'complaint'}, inplace=True)

# Save the updated dataframe to a new CSV file
df.to_csv("mapped_data.csv", index=False)

print("Subcategories, Categories, and Complaints have been mapped successfully!")
