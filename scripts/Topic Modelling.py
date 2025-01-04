#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import matplotlib.pyplot as plt

# Verify scikit-learn functionality
print("scikit-learn is functional!")

# Initialize sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Load the processed dataset
file_path = '/Users/dinabandhupanigrahi/customer-sentiment-dashboard/data/processed/Musical_Instruments_cleaned.csv'
df = pd.read_csv(file_path)

# Precompute additional variables
df['review_length'] = df['reviewText'].fillna('').apply(len)  # Review length
df['reviewText'] = df['reviewText'].fillna('').astype(str)    # Ensure text is string

# Sentiment Scores and Classification
df['sentiment_score'] = df['reviewText'].apply(lambda x: sid.polarity_scores(x)['compound'])
df['sentiment'] = df['sentiment_score'].apply(
    lambda x: 'positive' if x > 0.05 else 'negative' if x < -0.05 else 'neutral'
)

# Filter negative reviews
negative_reviews = df[df['sentiment'] == 'negative'].copy()

# Vectorize the text data for topic modeling
count_vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
count_matrix = count_vectorizer.fit_transform(negative_reviews['reviewText'])

# Perform LDA
num_topics = 3  # Adjust based on dataset
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(count_matrix)

# Display top words for each topic
terms = count_vectorizer.get_feature_names_out()
print("\nTop Words for Each Topic:")
for idx, topic in enumerate(lda.components_):
    top_terms = [terms[i] for i in topic.argsort()[-10:][::-1]]
    print(f"Topic {idx + 1}: {', '.join(top_terms)}")

# Assign topics to reviews
negative_reviews['topic'] = lda.transform(count_matrix).argmax(axis=1)

# Analyze topic distribution
print("\nTopic Distribution Across Reviews:")
print(negative_reviews['topic'].value_counts(normalize=True))

# Visualize topics with pyLDAvis
pyLDAvis.enable_notebook()

# Calculate term frequencies
term_frequency = count_matrix.sum(axis=0).A1
vocab = count_vectorizer.get_feature_names_out()

# Prepare visualization
vis = pyLDAvis.prepare(
    topic_term_dists=lda.components_,
    doc_topic_dists=lda.transform(count_matrix),
    doc_lengths=count_matrix.sum(axis=1).A1,
    vocab=vocab,
    term_frequency=term_frequency,
)
pyLDAvis.save_html(vis, 'topic_modeling_vis.html')
print("Interactive topic modeling visualization saved as 'topic_modeling_vis.html'.")


# In[ ]:




