import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset
file_path = 'cleaned_dataset.csv'
df = pd.read_csv(file_path)

# First, split the dataset into 60% training and 40% remaining
train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)

# Then, split the remaining 40% into 20% testing and 20% validation
test_df, val_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Optional: Check the size of the resulting datasets
print(f"Training Set Size: {len(train_df)}")
print(f"Testing Set Size: {len(test_df)}")
print(f"Validation Set Size: {len(val_df)}")

# Save the splits if needed (optional)
train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)
val_df.to_csv('validation_data.csv', index=False)

# Print the first few rows of the training set to verify
print(train_df.head())

