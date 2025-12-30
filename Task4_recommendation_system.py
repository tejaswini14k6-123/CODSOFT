"""Task 4: Recommendation System
Implements a recommendation engine using collaborative filtering and content-based filtering.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict

class RecommendationSystem:
    def __init__(self):
        self.users = {}
        self.items = {}
        self.ratings = {}
        self.similarity_matrix = None
    
    def add_user(self, user_id: str):
        """Add a new user to the system"""
        if user_id not in self.users:
            self.users[user_id] = {}
    
    def add_item(self, item_id: str, title: str, category: str, description: str):
        """Add a new item with metadata"""
        self.items[item_id] = {
            'title': title,
            'category': category,
            'description': description
        }
    
    def rate_item(self, user_id: str, item_id: str, rating: float):
        """Record a user's rating for an item"""
        if user_id not in self.users:
            self.add_user(user_id)
        if item_id not in self.items:
            return False
        
        if user_id not in self.ratings:
            self.ratings[user_id] = {}
        self.ratings[user_id][item_id] = rating
        return True
    
    def collaborative_filtering(self, user_id: str, n_recommendations: int = 5) -> List[tuple]:
        """Recommend items using collaborative filtering"""
        if user_id not in self.ratings:
            return []
        
        # Find similar users based on rating patterns
        user_ratings = self.ratings[user_id]
        similar_users = {}
        
        for other_user, other_ratings in self.ratings.items():
            if other_user == user_id:
                continue
            
            # Find common items
            common_items = set(user_ratings.keys()) & set(other_ratings.keys())
            if not common_items:
                continue
            
            # Calculate similarity
            user_vec = np.array([user_ratings[item] for item in common_items])
            other_vec = np.array([other_ratings[item] for item in common_items])
            
            similarity = cosine_similarity([user_vec], [other_vec])[0][0]
            if similarity > 0:
                similar_users[other_user] = similarity
        
        # Get recommendations from similar users
        recommendations = {}
        for similar_user, similarity_score in similar_users.items():
            for item_id, rating in self.ratings[similar_user].items():
                if item_id not in user_ratings:  # Haven't rated this item
                    if item_id not in recommendations:
                        recommendations[item_id] = []
                    recommendations[item_id].append(rating * similarity_score)
        
        # Calculate weighted scores
        scores = {item: np.mean(ratings) for item, ratings in recommendations.items()}
        
        # Sort and return top recommendations
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n_recommendations]
    
    def content_based_filtering(self, item_id: str, n_recommendations: int = 5) -> List[tuple]:
        """Recommend items based on content similarity"""
        if item_id not in self.items:
            return []
        
        # Create TF-IDF vectors for item descriptions
        descriptions = [self.items[iid]['description'] for iid in self.items.keys()]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(descriptions)
        
        # Calculate similarity
        item_index = list(self.items.keys()).index(item_id)
        similarities = cosine_similarity(tfidf_matrix[item_index], tfidf_matrix)[0]
        
        # Get top similar items (excluding the item itself)
        item_list = list(self.items.keys())
        similar_items = [(item_list[i], similarities[i]) for i in range(len(item_list)) 
                        if i != item_index and similarities[i] > 0]
        similar_items.sort(key=lambda x: x[1], reverse=True)
        
        return similar_items[:n_recommendations]
    
    def hybrid_recommendations(self, user_id: str, item_id: str = None, 
                              n_recommendations: int = 5) -> List[tuple]:
        """Combine collaborative and content-based filtering"""
        collab_recs = self.collaborative_filtering(user_id, n_recommendations)
        
        if item_id:
            content_recs = self.content_based_filtering(item_id, n_recommendations)
        else:
            # Use most recently rated item
            if user_id in self.ratings:
                recent_item = list(self.ratings[user_id].keys())[-1]
                content_recs = self.content_based_filtering(recent_item, n_recommendations)
            else:
                content_recs = []
        
        # Merge recommendations
        all_recs = {}
        for item, score in collab_recs:
            all_recs[item] = all_recs.get(item, 0) + score * 0.6
        for item, score in content_recs:
            all_recs[item] = all_recs.get(item, 0) + score * 0.4
        
        sorted_recs = sorted(all_recs.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:n_recommendations]

if __name__ == "__main__":
    # Example usage
    rec_system = RecommendationSystem()
    
    # Add items
    rec_system.add_item('movie1', 'Inception', 'Sci-Fi', 
                       'A mind-bending sci-fi thriller about dreams within dreams')
    rec_system.add_item('movie2', 'The Matrix', 'Sci-Fi', 
                       'A groundbreaking sci-fi action film with advanced special effects')
    rec_system.add_item('movie3', 'Interstellar', 'Sci-Fi', 
                       'An epic space exploration sci-fi drama')
    rec_system.add_item('book1', 'Python Cookbook', 'Tech', 
                       'A technical book about Python programming recipes')
    
    # Add users and ratings
    rec_system.add_user('user1')
    rec_system.rate_item('user1', 'movie1', 5)
    rec_system.rate_item('user1', 'movie2', 4.5)
    rec_system.rate_item('user1', 'book1', 3)
    
    rec_system.add_user('user2')
    rec_system.rate_item('user2', 'movie1', 4.5)
    rec_system.rate_item('user2', 'movie3', 5)
    
    print("Recommendation System Demo")
    print("=" * 50)
    print("\nCollaborative Filtering Recommendations for user1:")
    collab = rec_system.collaborative_filtering('user1')
    print(collab)
    
    print("\nContent-Based Filtering Recommendations for movie1:")
    content = rec_system.content_based_filtering('movie1')
    print(content)
    
    print("\nHybrid Recommendations for user1:")
    hybrid = rec_system.hybrid_recommendations('user1')
    print(hybrid)
