import csv

# Open the first CSV file and read its contents
with open('train.csv', 'r') as file1:
    reader1 = csv.reader(file1)
    header1 = next(reader1)  # Read the header
    data1 = list(reader1)

# Open the second CSV file and read its contents
with open('test.csv', 'r') as file2:
    reader2 = csv.reader(file2)
    header2 = next(reader2)  # Read the header
    data2 = list(reader2)

# Combine the data horizontally (side by side)
if header1 != header2:
    print("Headers do not match! Make sure the files have the same headers.")
else:
    # Combine the data
    merged_data = [header1] + data1 + data2

# Write the merged data to a new CSV file
with open('merged_file.csv', 'w', newline='') as output_file:
    writer = csv.writer(output_file)
    writer.writerows(merged_data)

print("Files merged successfully into 'merged_file.csv'")
