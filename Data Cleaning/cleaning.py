import pandas as pd

# Load your dataset
df = pd.read_csv('mapped_data.csv')

# Remove rows where the text in 'your_column' has less than 30 characters
df = df[df['complaint'].str.len() >= 30]

# Save the cleaned dataset
df.to_csv('cleaned_dataset.csv', index=False)

