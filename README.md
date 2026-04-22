# Movie Recommendation System

A content-based movie recommendation system built with Python, Streamlit, and machine learning.

## Features

- **Content-based filtering**: Recommends movies based on plot, genres, keywords, cast, and crew similarity
- **Interactive web interface**: Built with Streamlit for easy movie selection and recommendations
- **TMDB API integration**: Fetches movie posters (when network allows)
- **Efficient similarity matching**: Uses cosine similarity on text features

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/GowthamAI-ENGR/Movie_recommendation_system.git
   cd Movie_recommendation_system
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the data:**
   ```bash
   python prepare_data.py
   ```
   This will generate `movie_dict.pkl` and `similarity.pkl` files.

5. **Set up TMDB API key:**
   - Get your API key from [TMDB](https://www.themoviedb.org/settings/api)
   - Create a `.env` file in the project root:
   ```
   TMDB_API_KEY=your_api_key_here
   ```

6. **Run the application:**
   ```bash
   streamlit run app.py
   ```

7. **Open your browser** and go to `http://localhost:8501`

## Project Structure

```
Movie_recommendation_system/
├── app.py                    # Main Streamlit application
├── prepare_data.py          # Data preprocessing script
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
├── README.md               # This file
├── tmdb_5000_movies.csv    # Movie dataset
├── tmdb_5000_credits.csv   # Credits dataset
├── movie_dict.pkl          # Processed movie data (generated)
└── similarity.pkl          # Similarity matrix (generated)
```

## How It Works

1. **Data Processing**: The system processes movie data from TMDB 5000 dataset, extracting features like genres, keywords, cast, crew, and plot overview.

2. **Feature Engineering**: Text features are combined and vectorized using CountVectorizer.

3. **Similarity Calculation**: Cosine similarity is computed between all movie vectors.

4. **Recommendation**: When a user selects a movie, the system finds the most similar movies based on the similarity matrix.

## Dependencies

- pandas
- numpy
- scikit-learn
- streamlit
- requests
- python-dotenv

## API Key Security

The TMDB API key is stored in a `.env` file which is excluded from version control. Never commit API keys to public repositories.

## Troubleshooting

- **Posters not showing**: This may happen due to network restrictions. The app will show movie titles instead.
- **Large files**: The similarity matrix is large and may take time to generate.
- **Memory issues**: If you encounter memory issues, reduce the `max_features` parameter in `prepare_data.py`.

## License

This project is for educational purposes.