🎬 Movie Recommendation System

A content-based movie recommendation system built using the TMDB 5000 dataset.
This project preprocesses movie metadata, extracts key features, and recommends similar movies based on cosine similarity of combined text features.

🔍 What You’ll Find Here

📂 Data loading with robust local path handling

🧹 Careful data cleaning & feature engineering

🧠 A content-based similarity model using:

CountVectorizer (Bag-of-Words)

Cosine Similarity

🔄 A reusable recommend(title, n) function

📊 Visualizations:

Top movie genres

Popularity distribution

Similarity score distribution

🛠️ Tech Stack

Python

Pandas, NumPy for data handling

Scikit-learn for vectorization & similarity

Matplotlib for visualization

📦 Setup
1️⃣ Install Dependencies
pip install pandas numpy scikit-learn matplotlib

2️⃣ Download Dataset

Get the TMDB 5000 dataset from Kaggle
.
Place these files in your project folder:

tmdb_5000_movies.csv
tmdb_5000_credits.csv

📂 Project Structure
Movie-Recommendation-System/
│
├── content_based_movie_recommender.py   # Main Python script
├── tmdb_5000_movies.csv                # Movies dataset
├── tmdb_5000_credits.csv               # Credits dataset
└── README.md                           # Project documentation

🚀 How It Works
🔹 1. Data Loading
DATA_PATHS = {
    "movies": "tmdb_5000_movies.csv",
    "credits": "tmdb_5000_credits.csv",
}


Validates paths and loads both CSV files into a single merged DataFrame.

🔹 2. Data Cleaning & Feature Engineering

Parses JSON-like columns (genres, keywords, cast, crew)

Extracts top cast and director

Combines:

Overview

Genres

Keywords

Top 5 cast members

Director

This is stored in a single tag column for vectorization.

🔹 3. Vectorization
vectorizer = CountVectorizer(max_features=10000, stop_words='english')
matrix = vectorizer.fit_transform(df['tags'])
similarity = cosine_similarity(matrix)

🔹 4. Recommendation Function
recommend('Avatar', n=5)


Returns the top N similar movies with details:

Title

Genres

Overview

Vote average

Popularity

📊 Sample Visualizations

📌 Top Genres


📌 Popularity Distribution


📌 Similarity Scores


🧩 Example Output
Input: Avatar  
Top 5 Recommendations:
1. Titan A.E.  
2. Lifeforce  
3. Ender's Game  
4. Aliens vs Predator: Requiem  
5. Independence Day  

🎯 Next Steps

✅ Add collaborative filtering (user-based or item-based)

✅ Try matrix factorization (SVD)

✅ Build a web app (Streamlit/Flask)

✅ Deploy to Hugging Face Spaces or Heroku
