import pandas as pd
import difflib
import streamlit as slt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movie data
movies_data = pd.read_csv(r"C:\Users\91739\Desktop\SMV\268069\Elevate Lab intern\mini pro\MRS\movies.csv")

# Ensure index exists
if 'index' not in movies_data.columns:
    movies_data.reset_index(inplace=True)

# Handle missing values
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# Combine features into one string
combined_features = movies_data[selected_features].agg(' '.join, axis=1)

# Vectorization and similarity computation
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # better matching with bigrams
feature_vectors = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_vectors)

# Recommendation logic
def movie_recommend(movie_name):
    movie_name = movie_name.lower()
    list_of_all_titles = movies_data['title'].str.lower().tolist()
    
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles, n=1)
    if not find_close_match:
        return ["No matching movie found."]
    
    close_match = find_close_match[0]
    actual_title = movies_data[movies_data['title'].str.lower() == close_match]['title'].values[0]
    index_of_movie = movies_data[movies_data['title'] == actual_title]['index'].values[0]
    
    similarity_score = list(enumerate(similarity[index_of_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    recommended = []
    for movie in sorted_similar_movies[1:21]:  # show top 20
        index = movie[0]
        title_from_index = movies_data[movies_data.index == index]['title'].values[0]
        if title_from_index not in recommended:
            recommended.append(title_from_index)
    
    return recommended, actual_title

# Streamlit app
def main():
    slt.title('Movie Recommendation System')
    movie_name = slt.text_input('Enter your favourite movie')

    if slt.button('Recommend'):
        if movie_name:
            suggestions, matched_title = movie_recommend(movie_name)
            
            slt.subheader(f"Top Recommendations Based on: {matched_title}")
            for i, movie in enumerate(suggestions, 1):
                slt.write(f"{i}. {movie}")
            
            # Also show matching titles directly
            matching_titles = movies_data[movies_data['title'].str.lower().str.contains(movie_name.lower())]['title'].tolist()
            if matching_titles:
                slt.subheader("Other movies that contain your keyword:")
                for t in matching_titles:
                    slt.write(f"- {t}")
        else:
            slt.warning("Please enter a movie name.")

if __name__ == '__main__':
    main()
