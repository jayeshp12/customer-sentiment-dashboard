import json
import pandas as pd
import re

# File paths
input_file = '/Users/dinabandhupanigrahi/customer-sentiment-dashboard/data/raw/Musical_Instruments_5.json'
output_file = '/Users/dinabandhupanigrahi/customer-sentiment-dashboard/data/processed/Musical_Instruments_cleaned.csv'

# Read line-delimited JSON
data = []
with open(input_file, 'r') as file:
    for line in file:
        data.append(json.loads(line))

# Convert to DataFrame
df = pd.DataFrame(data)
print(f"Loaded dataset with {len(df)} records.")

# Drop rows with missing reviewText or overall ratings
df = df.dropna(subset=['reviewText', 'overall'])
print(f"Dataset after dropping missing values: {len(df)} records.")

# Clean the reviewText field
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters
    return text

df['reviewText'] = df['reviewText'].apply(clean_text)
print("Text normalization complete.")

# Save the processed data
df.to_csv(output_file, index=False)
print(f"Cleaned data saved to {output_file}")
