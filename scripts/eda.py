#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np

# Load the cleaned dataset
cleaned_data_path = '/Users/dinabandhupanigrahi/customer-sentiment-dashboard/data/processed/Musical_Instruments_cleaned.csv'
df = pd.read_csv(cleaned_data_path)

# Display basic information
print("Dataset Info:")
print(df.info())
print("\nFirst few records:")
print(df.head())

# ---- Ratings Distribution ----
rating_counts = df['overall'].value_counts()
rating_percentages = df['overall'].value_counts(normalize=True) * 100
print("\nRatings Distribution:")
print(rating_counts)
print("\nRatings Percentages:")
print(rating_percentages)

# ---- Review Text Length Analysis ----
df['review_length'] = df['reviewText'].str.len()
print("\nReview Length Statistics:")
print(df['review_length'].describe())
correlation = df['review_length'].corr(df['overall'])
print(f"\nCorrelation between review length and rating: {correlation}")

# ---- Frequent Reviewers ----
reviewer_counts = df['reviewerID'].value_counts()
print("\nTop 10 Reviewers by Number of Reviews:")
print(reviewer_counts.head(10))
top_reviewer = reviewer_counts.index[0]
top_reviewer_data = df[df['reviewerID'] == top_reviewer]
print(f"\nRatings by Top Reviewer ({top_reviewer}):")
print(top_reviewer_data['overall'].value_counts())

# ---- Trends Over Time ----
df['reviewTime'] = pd.to_datetime(df['reviewTime'], format='%m %d, %Y')
df['year'] = df['reviewTime'].dt.year
yearly_reviews = df.groupby('year').size()
print("\nYearly Review Counts:")
print(yearly_reviews)

# ---- Product Performance ----
product_stats = df.groupby('asin').agg(
    total_reviews=('overall', 'count'),
    avg_rating=('overall', 'mean')
)
most_reviewed = product_stats.sort_values(by='total_reviews', ascending=False).head(10)
highest_rated = product_stats.sort_values(by='avg_rating', ascending=False).head(10)

print("\nTop 10 Most Reviewed Products:")
print(most_reviewed)
print("\nTop 10 Highest Rated Products:")
print(highest_rated)

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


# Add a new column for review length
df['review_length'] = df['reviewText'].str.len()

# Summary statistics for review length
print("Review Length Statistics:")
print(df['review_length'].describe())

# Correlation between review length and overall rating
correlation = df['review_length'].corr(df['overall'])
print(f"\nCorrelation between review length and rating: {correlation:.2f}")

# Shortest reviews
short_reviews = df[df['review_length'] < 20]
print("\nExamples of very short reviews:")
print(short_reviews[['reviewText', 'overall']].head())

# Longest reviews
long_reviews = df[df['review_length'] > 500]
print("\nExamples of very long reviews:")
print(long_reviews[['reviewText', 'overall']].head())


#Deep dive into outliers

# Define a threshold for "short" reviews
short_threshold = 20  # Characters

# Filter reviews below the threshold
short_reviews = df[df['review_length'] < short_threshold]

# Count and inspect short reviews
print(f"Number of short reviews (< {short_threshold} characters): {len(short_reviews)}")
print("\nExamples of very short reviews:")
print(short_reviews[['reviewText', 'overall']].head(10))

# Define a threshold for "long" reviews
long_threshold = 500  # Characters

# Filter reviews above the threshold
long_reviews = df[df['review_length'] > long_threshold]

# Count and inspect long reviews
print(f"\nNumber of long reviews (> {long_threshold} characters): {len(long_reviews)}")
print("\nExamples of very long reviews:")
print(long_reviews[['reviewText', 'overall']].head(10))

# Average ratings for short reviews
short_reviews_avg_rating = short_reviews['overall'].mean()
print(f"\nAverage rating for short reviews: {short_reviews_avg_rating:.2f}")

# Average ratings for long reviews
long_reviews_avg_rating = long_reviews['overall'].mean()
print(f"Average rating for long reviews: {long_reviews_avg_rating:.2f}")

from collections import Counter

# Common words in short reviews
short_words = Counter(" ".join(short_reviews['reviewText']).split())
print("\nMost common words in short reviews:")
print(short_words.most_common(10))

# Common words in long reviews
long_words = Counter(" ".join(long_reviews['reviewText']).split())
print("\nMost common words in long reviews:")
print(long_words.most_common(10))

#segment analysis to study the weak correlation
for rating in sorted(df['overall'].unique()):
    subset = df[df['overall'] == rating]
    correlation = subset['review_length'].corr(subset['overall'])
    print(f"Correlation for {rating}-star reviews: {correlation:.2f}")
    
avg_length_by_rating = df.groupby('overall')['review_length'].mean()
print(avg_length_by_rating)


from collections import Counter

# Group by ratings
for rating in sorted(df['overall'].unique()):
    print(f"\n--- Analyzing {rating}-Star Reviews ---")
    
    # Create a copy of the subset
    subset = df[df['overall'] == rating].copy()
    
    # Ensure reviewText contains only strings
    subset['reviewText'] = subset['reviewText'].fillna('').astype(str)
    
    # Summary statistics
    avg_length = subset['review_length'].mean()
    print(f"Average review length: {avg_length:.2f}")
    
    # Common themes
    most_common_words = Counter(" ".join(subset['reviewText']).split()).most_common(10)
    print(f"Most common words: {most_common_words}")




# Calculate product statistics
product_stats = df.groupby('asin').agg(
    total_reviews=('overall', 'count'),
    avg_rating=('overall', 'mean')
)

# Filter for highly rated and poorly rated products
high_rated_products = product_stats[product_stats['avg_rating'] >= 4.5].sort_values('total_reviews', ascending=False).head(5)
low_rated_products = product_stats[product_stats['avg_rating'] <= 2.5].sort_values('total_reviews', ascending=False).head(5)

print("\nTop 5 Highly Rated Products:")
print(high_rated_products)
print("\nTop 5 Poorly Rated Products:")
print(low_rated_products)


# Convert reviewTime to datetime if not already done
df['reviewTime'] = pd.to_datetime(df['reviewTime'], format='%m %d, %Y')
df['year_month'] = df['reviewTime'].dt.to_period('M')

# Monthly trend of average ratings
monthly_trends = df.groupby('year_month').agg(
    avg_rating=('overall', 'mean'),
    review_count=('overall', 'count')
)

print("\nMonthly Trends:")
print(monthly_trends.tail())

# Frequent reviewers
frequent_reviewers = df['reviewerID'].value_counts().head(10).index

for reviewer in frequent_reviewers:
    reviewer_data = df[df['reviewerID'] == reviewer]
    avg_rating = reviewer_data['overall'].mean()
    avg_length = reviewer_data['review_length'].mean()
    print(f"\nReviewer: {reviewer}")
    print(f"Average Rating: {avg_rating:.2f}")
    print(f"Average Review Length: {avg_length:.2f}")
    print(f"Number of Reviews: {len(reviewer_data)}")

    
# Identify top 10 reviewers by number of reviews
top_reviewers = df['reviewerID'].value_counts().head(10)
print("\nTop 10 Reviewers by Review Count:")
print(top_reviewers)

# Analyze behavior of each top reviewer
for reviewer in top_reviewers.index:
    reviewer_data = df[df['reviewerID'] == reviewer]
    avg_rating = reviewer_data['overall'].mean()
    avg_length = reviewer_data['review_length'].mean()
    num_reviews = len(reviewer_data)

    print(f"\nReviewer: {reviewer}")
    print(f"  Number of Reviews: {num_reviews}")
    print(f"  Average Rating: {avg_rating:.2f}")
    print(f"  Average Review Length: {avg_length:.2f}")
    
# Convert reviewTime to datetime format if not already done
df['reviewTime'] = pd.to_datetime(df['reviewTime'], format='%m %d, %Y')

# Group by year and month
df['year_month'] = df['reviewTime'].dt.to_period('M')
monthly_trends = df.groupby('year_month').agg(
    avg_rating=('overall', 'mean'),
    review_count=('overall', 'count')
)

print("\nMonthly Trends (Last 12 Months):")
print(monthly_trends.tail(12))

# Check overall trends
overall_avg_rating = df.groupby(df['reviewTime'].dt.year)['overall'].mean()
print("\nYearly Average Ratings:")
print(overall_avg_rating)


# Convert reviewTime to datetime format if not already done
df['reviewTime'] = pd.to_datetime(df['reviewTime'], format='%m %d, %Y')

# Group by year and month
df['year_month'] = df['reviewTime'].dt.to_period('M')
monthly_trends = df.groupby('year_month').agg(
    avg_rating=('overall', 'mean'),
    review_count=('overall', 'count')
)

print("\nMonthly Trends (Last 12 Months):")
print(monthly_trends.tail(12))

# Check overall trends
overall_avg_rating = df.groupby(df['reviewTime'].dt.year)['overall'].mean()
print("\nYearly Average Ratings:")
print(overall_avg_rating)



# Calculate rating distributions per product
product_rating_dist = df.groupby('asin')['overall'].value_counts().unstack(fill_value=0)

# Calculate percentage of 1-star and 5-star reviews
product_rating_dist['low_ratings'] = product_rating_dist[1.0] + product_rating_dist[2.0]
product_rating_dist['high_ratings'] = product_rating_dist[4.0] + product_rating_dist[5.0]
product_rating_dist['polarization'] = product_rating_dist['low_ratings'] + product_rating_dist['high_ratings']

# Sort by most polarized products
polarized_products = product_rating_dist.sort_values('polarization', ascending=False).head(5)

print("\nTop 5 Most Polarized Products:")
print(polarized_products[['low_ratings', 'high_ratings', 'polarization']])


# Identify reviewers with only 1-star or 5-star ratings
extreme_reviewers = df.groupby('reviewerID')['overall'].apply(lambda x: set(x)).reset_index()
extreme_reviewers = extreme_reviewers[extreme_reviewers['overall'].apply(lambda x: x.issubset({1.0, 5.0}))]

# Filter reviews by extreme reviewers
extreme_reviews = df[df['reviewerID'].isin(extreme_reviewers['reviewerID'])]

# Analyze behavior
extreme_stats = extreme_reviews.groupby('reviewerID').agg(
    num_reviews=('overall', 'count'),
    avg_rating=('overall', 'mean'),
    avg_length=('review_length', 'mean')
)
print("\nExtreme Reviewers Stats:")
print(extreme_stats)


# Identify top 5 most reviewed products
top_products = df['asin'].value_counts().head(5).index

# Group reviews of top products by year and calculate average rating
for product in top_products:
    product_data = df[df['asin'] == product]
    yearly_trends = product_data.groupby(product_data['reviewTime'].dt.year)['overall'].mean()
    print(f"\nYearly Trends for Product {product}:")
    print(yearly_trends)

    
# Identify top 5 reviewers by number of reviews
top_reviewers = df['reviewerID'].value_counts().head(5).index

# Group reviews by year for each top reviewer
for reviewer in top_reviewers:
    reviewer_data = df[df['reviewerID'] == reviewer]
    yearly_trends = reviewer_data.groupby(reviewer_data['reviewTime'].dt.year)['overall'].mean()
    print(f"\nYearly Trends for Reviewer {reviewer}:")
    print(yearly_trends)


# In[ ]:




