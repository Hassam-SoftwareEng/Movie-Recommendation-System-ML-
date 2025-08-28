import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# ✅ Set the path to your dataset folder
dataset_path = r"D:\Remote Internship\Movie Recomanded Task\ml-100k"

# ✅ Load MovieLens dataset
ratings = pd.read_csv(
    os.path.join(dataset_path, "u.data"),
    sep="\t",
    names=["user_id", "item_id", "rating", "timestamp"]
)

movies = pd.read_csv(
    os.path.join(dataset_path, "u.item"),
    sep="|",
    encoding="latin-1",
    names=["movie_id", "title"],
    usecols=[0, 1]
)

# ✅ Merge ratings with movie titles
data = pd.merge(ratings, movies, left_on="item_id", right_on="movie_id")

# ✅ Create User-Item Matrix
user_item_matrix = data.pivot_table(index='user_id', columns='title', values='rating')

# ✅ Fill missing values with 0 for similarity calculation
user_item_matrix_filled = user_item_matrix.fillna(0)

# ✅ Compute user similarity
user_similarity = cosine_similarity(user_item_matrix_filled)
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)

# ✅ Movie Recommendation Function
def recommend_movies(user_id, top_n=5):
    # Get similarity scores for this user
    sim_scores = user_similarity_df[user_id].drop(user_id)
    similar_users = sim_scores.sort_values(ascending=False)

    # Movies already rated by this user
    user_movies = user_item_matrix.loc[user_id].dropna().index

    # Weighted scores for each movie
    movie_scores = {}
    for other_user, sim in similar_users.items():
        other_ratings = user_item_matrix.loc[other_user].dropna()
        for movie, rating in other_ratings.items():
            if movie not in user_movies:
                movie_scores[movie] = movie_scores.get(movie, 0) + sim * rating

    # Sort and recommend top N movies
    recommended = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [movie for movie, score in recommended]

# ✅ Test the system
print("🎬 Recommended Movies for User 1:")
print(recommend_movies(1))
# ==========================================
# Content-Based Movie Recommendation System
# Dataset: TMDB 5000 Movies + Credits
# ==========================================

# 1) Setup
# Install required libraries first:
# pip install pandas numpy scikit-learn matplotlib

import os
import re
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

# Display settings
pd.set_option('display.max_colwidth', 120)
pd.set_option('display.float_format', lambda x: f'{x:.2f}')
plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['axes.grid'] = True


# 2) Data Paths
# 🔹 Change this path to where you downloaded the TMDB dataset
DATA_PATHS = {
    "movies": r"D:\Remote Internship\Movie Recomanded Task\tmdb_5000_movies.csv",
    "credits": r"D:\Remote Internship\Movie Recomanded Task\tmdb_5000_credits.csv",
}

for k, p in DATA_PATHS.items():
    if not os.path.exists(p):
        print(f"[Warning] Could not find '{p}'. Update DATA_PATHS for '{k}'.")


# 3) Load Data
def load_tmdb(paths: dict) -> pd.DataFrame:
    """Load TMDB 5000 dataset and merge movies + credits."""
    movies = pd.read_csv(paths["movies"])
    credits = pd.read_csv(paths["credits"])

    assert {"id", "title", "genres", "keywords", "overview"}.issubset(movies.columns), "Movies CSV missing columns."
    assert {"movie_id", "cast", "crew"}.issubset(credits.columns), "Credits CSV missing columns."

    df = movies.merge(credits, left_on="id", right_on="movie_id", how="inner", suffixes=("", "_cred"))
    return df


try:
    df_raw = load_tmdb(DATA_PATHS)
    print(f"[Info] Loaded {len(df_raw):,} rows. Columns: {list(df_raw.columns)[:10]}...")
except FileNotFoundError as e:
    raise FileNotFoundError("Place TMDB CSVs in your folder or fix DATA_PATHS.") from e


# 4) Parse JSON-like Columns
def parse_name_list(s):
    """Parse a JSON-like list string and extract names."""
    if pd.isna(s):
        return []
    try:
        items = ast.literal_eval(s)
        if isinstance(items, list):
            return [d.get("name", "") for d in items if isinstance(d, dict) and d.get("name")]
    except Exception:
        pass
    return []


def parse_top_n(lst, n=5):
    return lst[:n] if isinstance(lst, list) else []


def extract_director(s):
    if pd.isna(s):
        return ""
    try:
        crew = ast.literal_eval(s)
        for d in crew:
            if isinstance(d, dict) and d.get("job") == "Director":
                return d.get("name", "")
    except Exception:
        pass
    return ""


df = df_raw.copy()
df["genres_list"] = df["genres"].apply(parse_name_list)
df["keywords_list"] = df["keywords"].apply(parse_name_list)
df["cast_list"] = df["cast"].apply(parse_name_list).apply(lambda x: parse_top_n(x, 5))
df["director"] = df["crew"].apply(extract_director)
df["title_clean"] = df["title"].str.strip().str.lower()


# 5) Quick Visualizations
# Top genres
all_genres = [g for lst in df["genres_list"] for g in lst]
top_genres = pd.Series(all_genres).value_counts().head(15)

plt.figure()
top_genres.sort_values().plot(kind="barh", color="orange")
plt.title("Top Genres by Frequency")
plt.xlabel("Count")
plt.tight_layout()
plt.show()

# Popularity distribution
plt.figure()
df["popularity"].dropna().plot(kind="hist", bins=40, color="purple", edgecolor="white")
plt.title("Distribution of Popularity")
plt.xlabel("Popularity")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Vote average distribution
plt.figure()
df["vote_average"].dropna().plot(kind="hist", bins=20, color="green", edgecolor="white")
plt.title("Distribution of Vote Average")
plt.xlabel("Vote Average")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# 6) Build Tags
def clean_token(s):
    s = re.sub(r"[^a-zA-Z0-9\s]", " ", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()


df["overview_clean"] = df["overview"].fillna("").apply(clean_token)
df["genres_clean"] = df["genres_list"].apply(lambda lst: " ".join([clean_token(x).replace(" ", "") for x in lst]))
df["keywords_clean"] = df["keywords_list"].apply(lambda lst: " ".join([clean_token(x).replace(" ", "") for x in lst[:10]]))
df["cast_clean"] = df["cast_list"].apply(lambda lst: " ".join([clean_token(x).replace(" ", "") for x in lst]))
df["director_clean"] = df["director"].apply(lambda x: clean_token(x).replace(" ", ""))

df["tags"] = (df["overview_clean"] + " " +
              df["genres_clean"] + " " +
              df["keywords_clean"] + " " +
              df["cast_clean"] + " " +
              df["director_clean"]).str.strip()


# 7) Vectorization and Similarity
vectorizer = CountVectorizer(max_features=10000, stop_words="english")
matrix = vectorizer.fit_transform(df["tags"])
similarity = cosine_similarity(matrix)

print(f"[Info] Matrix shape: {matrix.shape}, Similarity shape: {similarity.shape}")


# 8) Recommendation API
title_to_index = {t: i for i, t in enumerate(df["title_clean"])}

def recommend(title, n=10, return_scores=False):
    """Return top-n similar movies."""
    if not isinstance(title, str) or not title.strip():
        raise ValueError("Provide a valid movie title.")

    key = title.strip().lower()
    idx = title_to_index.get(key)

    # Fuzzy match
    if idx is None:
        candidates = get_close_matches(key, list(title_to_index.keys()), n=3, cutoff=0.6)
        if candidates:
            idx = title_to_index[candidates[0]]
            print(f"[Info] Closest match: '{df.loc[idx, 'title']}' for '{title}'.")
        else:
            raise ValueError(f"Title '{title}' not found.")

    sims = list(enumerate(similarity[idx]))
    sims = sorted(sims, key=lambda x: x[1], reverse=True)
    top = sims[1:n+1]
    rec_idx = [i for i, s in top]
    rec_scores = [s for i, s in top]

    out = df.loc[rec_idx, ["title", "genres_list", "overview", "vote_average", "popularity"]].copy().reset_index(drop=True)
    if return_scores:
        out.insert(1, "score", rec_scores)
    return out


# 9) Demo
try:
    print("\n🎬 Recommendations for 'Avatar':")
    print(recommend("Avatar", 5, return_scores=True))
except Exception as e:
    print("[Error] Try another title from df['title'].head().tolist()")
