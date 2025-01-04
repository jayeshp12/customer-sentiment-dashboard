#!/usr/bin/env python
# coding: utf-8

# In[5]:


import json

# File path
file_path = '/Users/dinabandhupanigrahi/customer-sentiment-dashboard/data/raw/Musical_Instruments_5.json'

# Read line-delimited JSON
data = []
with open(file_path, 'r') as file:
    for line in file:
        data.append(json.loads(line))

# Display basic information
print(f"Total records: {len(data)}")
print("Sample record:")
print(json.dumps(data[0], indent=2))


# In[6]:


import pandas as pd

# Path to the cleaned data
cleaned_data_path = '/Users/dinabandhupanigrahi/customer-sentiment-dashboard/data/processed/Musical_Instruments_cleaned.csv'

# Load and inspect
df = pd.read_csv(cleaned_data_path)
print(df.head())
print(df.info())


# In[ ]:




