import pandas as pd
from sklearn.utils import resample

# Load the cleaned dataset
data = pd.read_csv("./../PythonTester/cleaned_dataset.csv")

# Ensure the dataset has the required columns
required_columns = {"complaint", "true_category", "true_subcategory"}
if not required_columns.issubset(data.columns):
    raise ValueError(f"The dataset must contain the following columns: {required_columns}")

# Create a function to generate balanced samples
def generate_balanced_sample(data, text_col, label_col, n_samples):
    # Group by the label column
    grouped = data.groupby(label_col)
    samples = []
    
    # Resample each group to balance the distribution
    for _, group in grouped:
        resampled_group = resample(group, replace=True, n_samples=min(len(group), n_samples ), random_state=42)
        samples.append(resampled_group)
    
    # Concatenate the balanced groups and shuffle
    balanced_data = pd.concat(samples).sample(n=n_samples, random_state=42)
    return balanced_data

# Generate balanced category data
category_data = generate_balanced_sample(data, "complaint", "true_category", 1000)
category_data = category_data[["complaint", "true_category"]].rename(columns={"complaint": "text", "true_category": "label"})
category_data.to_csv("category.csv", index=False)

# Generate balanced subcategory data
subcategory_data = generate_balanced_sample(data, "complaint", "true_subcategory", 1000)
subcategory_data = subcategory_data[["complaint", "true_subcategory"]].rename(columns={"complaint": "text", "true_subcategory": "label"})
subcategory_data.to_csv("subcategory.csv", index=False)

print("Category and Subcategory files created successfully!")
