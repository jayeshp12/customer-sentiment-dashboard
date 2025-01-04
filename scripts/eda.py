import pandas as pd

# Load the cleaned dataset
cleaned_data_path = '/Users/dinabandhupanigrahi/customer-sentiment-dashboard/data/processed/Musical_Instruments_cleaned.csv'
df = pd.read_csv(cleaned_data_path)

# Display basic information
print(df.head())
print(df.info())


# Analyze distribution of ratings
rating_counts = df['overall'].value_counts()
rating_percentages = df['overall'].value_counts(normalize=True) * 100

# Print results
print("Ratings Distribution:")
print(rating_counts)
print("\nRatings Percentages:")
print(rating_percentages)


# Add a column for review length
df['review_length'] = df['reviewText'].str.len()

# Analyze review length distribution
print("Review Length Statistics:")
print(df['review_length'].describe())

# Correlation between review length and rating
correlation = df['review_length'].corr(df['overall'])
print(f"\nCorrelation between review length and rating: {correlation}")


# Count reviews per reviewer
reviewer_counts = df['reviewerID'].value_counts()

# Top 10 reviewers
print("Top 10 Reviewers by Number of Reviews:")
print(reviewer_counts.head(10))

# Analyze consistency of ratings by top reviewers
top_reviewer = reviewer_counts.index[0]  # First top reviewer
top_reviewer_data = df[df['reviewerID'] == top_reviewer]
print(f"\nRatings by Top Reviewer ({top_reviewer}):")
print(top_reviewer_data['overall'].value_counts())


# Convert reviewTime to datetime format if not done already
df['reviewTime'] = pd.to_datetime(df['reviewTime'], format='%m %d, %Y')

# Group by year and analyze
df['year'] = df['reviewTime'].dt.year
yearly_reviews = df.groupby('year').size()

# Print yearly review counts
print("Yearly Review Counts:")
print(yearly_reviews)


# Count reviews per product and calculate average rating
product_stats = df.groupby('asin').agg(
    total_reviews=('overall', 'count'),
    avg_rating=('overall', 'mean')
)

# Sort products by total reviews and average rating
most_reviewed = product_stats.sort_values(by='total_reviews', ascending=False).head(10)
highest_rated = product_stats.sort_values(by='avg_rating', ascending=False).head(10)

# Print results
print("Top 10 Most Reviewed Products:")
print(most_reviewed)
print("\nTop 10 Highest Rated Products:")
print(highest_rated)


