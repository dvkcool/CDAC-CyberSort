import csv
from collections import Counter

# Read the merged CSV file and collect both categories and sub-categories
with open('merged_file.csv', 'r') as file:
    reader = csv.DictReader(file)
    categories = [row['category'] if row['category'] else 'Empty Category' for row in reader]

# Re-read the file to get sub-categories, labeling empty ones as "Empty Sub-Category"
with open('merged_file.csv', 'r') as file:
    reader = csv.DictReader(file)
    sub_categories = [
        row['sub_category'] if row['sub_category'] else 'Empty Sub-Category' for row in reader
    ]

# Use Counter to count occurrences of each category and sub-category
category_counts = Counter(categories)
sub_category_counts = Counter(sub_categories)

# Calculate the total number of rows
total_category_count = sum(category_counts.values())
total_sub_category_count = sum(sub_category_counts.values())

# Write the results to a text file
with open('category_and_sub_category_counts.txt', 'w') as output_file:
    output_file.write("Category Counts:\n")
    for category, count in category_counts.items():
        output_file.write(f"{category}: {count}\n")
    output_file.write(f"\nTotal number of rows with a category: {total_category_count}\n\n")

    output_file.write("Sub-Category Counts:\n")
    for sub_category, count in sub_category_counts.items():
        output_file.write(f"{sub_category}: {count}\n")
    output_file.write(f"\nTotal number of rows with a sub-category: {total_sub_category_count}\n")

print("Category and sub-category counts written to 'category_and_sub_category_counts.txt'")

