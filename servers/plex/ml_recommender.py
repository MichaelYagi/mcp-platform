"""
Plex ML Recommendation Ranker
Learns from your viewing history to recommend content you'll like
"""

import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
# scikit-learn = For traditional ML (XGBoost, Random Forest, etc.), not LLMs
# The ML model
from sklearn.ensemble import RandomForestClassifier
# Convert genres to numbers
from sklearn.preprocessing import LabelEncoder
# Split data
from sklearn.model_selection import train_test_split


class PlexRecommender:
    """ML-powered Plex recommendation system"""

    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / "plex_recommender.pkl"
        self.encoders_path = self.model_dir / "plex_encoders.pkl"
        self.history_path = self.model_dir / "plex_viewing_history.csv"

        self.model = None
        self.encoders = {}

        # Create models directory if it doesn't exist
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Load existing model if available
        self._load_model()

    def _load_model(self):
        """Load existing model and encoders"""
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                self.encoders = joblib.load(self.encoders_path)
                print("✅ Loaded existing recommendation model")
            except Exception as e:
                print(f"⚠️  Could not load model: {e}")
                self.model = None

    def record_view(self, title, genre, year, rating, runtime, finished=True):
        """
        Record a viewing event

        Args:
            title: Movie/show title
            genre: Genre (Action, Comedy, Drama, etc.)
            year: Release year
            rating: IMDb/audience rating (0-10)
            runtime: Runtime in minutes
            finished: Whether you finished watching (True) or abandoned (False)
        """
        # Create viewing record
        record = {
            'title': title,
            'genre': genre,
            'year': year,
            'rating': rating,
            'runtime': runtime,
            'finished': 1 if finished else 0,
            'timestamp': datetime.now().isoformat()
        }

        # Append to history
        df = pd.DataFrame([record])

        if self.history_path.exists():
            existing = pd.read_csv(self.history_path)
            df = pd.concat([existing, df], ignore_index=True)

        df.to_csv(self.history_path, index=False)

        return {
            "status": "recorded",
            "total_views": len(df),
            "can_train": len(df) >= 20
        }

    def _prepare_features(self, df, fit_encoders=False):
        """Convert data to ML features"""

        # Encode categorical features
        if fit_encoders or 'genre' not in self.encoders:
            self.encoders['genre'] = LabelEncoder()
            df['genre_encoded'] = self.encoders['genre'].fit_transform(df['genre'])
        else:
            # Handle unseen genres
            df['genre_encoded'] = df['genre'].apply(
                lambda x: self.encoders['genre'].transform([x])[0]
                if x in self.encoders['genre'].classes_
                else -1
            )

        # Create features
        features = df[['genre_encoded', 'year', 'rating', 'runtime']].copy()

        # Add derived features
        features['is_recent'] = (df['year'] >= 2020).astype(int)
        features['is_short'] = (df['runtime'] < 100).astype(int)
        features['is_highly_rated'] = (df['rating'] >= 7.5).astype(int)

        return features

    def train(self, min_samples=20):
        """
        Train the recommendation model on viewing history

        Args:
            min_samples: Minimum number of viewing records needed to train

        Returns:
            dict with training results
        """
        # Check if we have history
        if not self.history_path.exists():
            return {
                "status": "no_data",
                "message": "No viewing history found. Record some views first!",
                "views_needed": min_samples
            }

        # Load history
        df = pd.read_csv(self.history_path)

        # Check if we have enough data
        if len(df) < min_samples:
            return {
                "status": "insufficient_data",
                "message": f"Need at least {min_samples} views to train",
                "current_views": len(df),
                "views_needed": min_samples - len(df)
            }

        # Prepare features
        X = self._prepare_features(df, fit_encoders=True)
        y = df['finished']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X_train, y_train)

        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)

        # Save model and encoders
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.encoders, self.encoders_path)

        return {
            "status": "success",
            "message": "Model trained successfully!",
            "training_samples": len(df),
            "train_accuracy": f"{train_score:.1%}",
            "test_accuracy": f"{test_score:.1%}",
            "model_path": str(self.model_path)
        }

    def predict_enjoyment(self, items):
        """
        Predict which items you'll enjoy

        Args:
            items: List of dicts with keys: title, genre, year, rating, runtime

        Returns:
            List of items sorted by predicted enjoyment (best first)
        """
        if self.model is None:
            return {
                "status": "no_model",
                "message": "No trained model. Train first with viewing history!"
            }

        # Convert to DataFrame
        df = pd.DataFrame(items)

        # Prepare features
        X = self._prepare_features(df, fit_encoders=False)

        # Predict probability of finishing (enjoyment proxy)
        # If model was trained on only one class, predict_proba returns 1 column
        proba = self.model.predict_proba(X)
        if proba.shape[1] == 1:
            # Only one class seen during training — check which one
            only_class = self.model.classes_[0]
            # If that class is 1 (all finished), assign high scores; else low
            probabilities = np.full(len(X), 0.9 if only_class == 1 else 0.1)
        else:
            probabilities = proba[:, 1]

        # Add scores to items
        for item, prob in zip(items, probabilities):
            item['ml_score'] = float(prob)
            item['ml_rank'] = 0  # Will be set after sorting

        # Sort by score
        ranked_items = sorted(items, key=lambda x: x['ml_score'], reverse=True)

        # Add rank
        for rank, item in enumerate(ranked_items, 1):
            item['ml_rank'] = rank

        return {
            "status": "success",
            "items": ranked_items
        }

    def get_stats(self):
        """Get recommendation system statistics"""

        stats = {
            "model_trained": self.model is not None,
            "total_views": 0,
            "genres_seen": [],
            "avg_rating": 0,
            "finish_rate": 0
        }

        if self.history_path.exists():
            df = pd.read_csv(self.history_path)
            stats['total_views'] = len(df)
            stats['genres_seen'] = df['genre'].unique().tolist()
            stats['avg_rating'] = float(df['rating'].mean())
            stats['finish_rate'] = f"{df['finished'].mean():.1%}"

        return stats

    def reset(self):
        """Clear all data and retrain from scratch"""

        # Remove files
        for path in [self.model_path, self.encoders_path, self.history_path]:
            if path.exists():
                path.unlink()

        self.model = None
        self.encoders = {}

        return {"status": "reset", "message": "All data cleared"}


# Singleton instance
_recommender = None


def get_recommender():
    """Get the global recommender instance"""
    global _recommender
    if _recommender is None:
        _recommender = PlexRecommender()
    return _recommender