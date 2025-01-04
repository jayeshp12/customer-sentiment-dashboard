#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# Load the cleaned dataset
cleaned_data_path = '../data/processed/Musical_Instruments_cleaned.csv'
df = pd.read_csv(cleaned_data_path)

# Analyze distribution of ratings
rating_counts = df['overall'].value_counts()
rating_percentages = df['overall'].value_counts(normalize=True) * 100

# Display results
print("Ratings Distribution (Counts):")
print(rating_counts)
print("\nRatings Distribution (Percentages):")
print(rating_percentages)

# Calculate mode
most_common_rating = rating_counts.idxmax()
print(f"\nMost common rating: {most_common_rating} ({rating_counts[most_common_rating]} reviews)")

# Positive reviews (4 or 5 stars)
positive_reviews = rating_percentages[4] + rating_percentages[5]
print(f"\nPercentage of positive reviews (4 or 5 stars): {positive_reviews:.2f}%")

# Negative reviews (1 or 2 stars)
negative_reviews = rating_percentages[1] + rating_percentages[2]
print(f"Percentage of negative reviews (1 or 2 stars): {negative_reviews:.2f}%")



# In[ ]:




