import pandas as pd

# Load your dataset
df = pd.read_csv('mapped_data.csv')

# First, trim spaces from both ends and then replace continuous spaces with a single space
df['complaint'] = df['complaint'].str.strip().str.replace(r'\s+', ' ', regex=True).str.strip()

# Remove rows where the text has less than 30 characters
df = df[df['complaint'].str.len() >= 30]

# Save the cleaned dataset
df.to_csv('cleaned_dataset.csv', index=False)
