# Customer Sentiment Analysis Dashboard
This project provides a comprehensive analysis of customer sentiment using the Amazon Musical Instruments Reviews Dataset. It includes end-to-end data processing, exploratory data analysis (EDA), advanced sentiment analysis, topic modeling, time-series analysis, and plans for visualization using Tableau.

## Project Files
- [Customer Sentiment Analysis Presentation](docs/Customer_Sentiment_Analysis_Presentation.pptx)

## Project Overview
The goal of this project is to extract meaningful insights from customer reviews, focusing on:

  Sentiment Analysis: Understand customer sentiment towards musical instruments.
  Topic Modeling: Identify recurring themes in customer reviews.
  Time-Series Analysis: Analyze trends over time in review sentiment and volume.
  Visualization: Present results interactively using Tableau (upcoming).

## Technologies Used
The project leverages the following tools and technologies:

### Python Libraries
- Data Handling: pandas, numpy
- NLP: NLTK, TextBlob
- Machine Learning: scikit-learn, KMeans
- Topic Modeling: LatentDirichletAllocation, pyLDAvis
- Visualization: matplotlib

### Visualization
- Tableau

## Folder Structure 

├── data/                  # Contains raw and processed datasets
│   ├── raw/               # Original datasets
│   ├── processed/         # Cleaned and prepared datasets
├── notebooks/             # Jupyter notebooks for analysis
│   ├── preprocessing.ipynb
│   ├── eda.ipynb
│   ├── sentiment_analysis.ipynb
│   ├── advanced_analysis.ipynb
│   ├── time_series_analysis.ipynb
├── scripts/               # Python scripts for reproducibility
├── tableau/               # Tableau files (to be added)
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation


## Getting Started
1. Clone the repository.
   git clone git@github.com:jayeshp12/customer-sentiment-dashboard.git
   cd customer-sentiment-dashboard

2. Install dependencies using `pip install -r requirements.txt`.
   pip install -r requirements.txt

3. Run Jupyter Notebooks by navigating to the notebooks/ folder.

## Results Overview

Key Insights
- Majority of reviews are overwhelmingly positive, with over 85% classified as positive sentiment.
- Topic modeling revealed key themes such as product quality, sound, and durability.
- Time-series analysis indicated a steady rise in review volume, peaking in recent years.

## Contact 
Email: jayeshpanigrahi12@gmail.com
LinkedIn: https://www.linkedin.com/in/jayesh-panigrahi-08b964179/

## License
This project uses publicly available data and is for educational purposes.
