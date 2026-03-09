#!/usr/bin/env python
"""MovieLens data loader for generating test samples for LLM re-ranking tasks"""

import os
import random
from typing import List, Dict, Tuple
from collections import defaultdict


class MovieLensDataLoader:
    """Load the MovieLens dataset and generate re-ranking test samples"""
    
    def __init__(self, data_dir: str = "ml-100k"):
        self.data_dir = data_dir
        self.movies = {}  # movie_id -> {title, genres}
        self.ratings = defaultdict(list)  # user_id -> [(movie_id, rating, timestamp)]
        self._load_movies()
        self._load_ratings()
    
    def _load_movies(self):
        """Read u.item file and parse movie information"""
        item_path = os.path.join(self.data_dir, "u.item")
        genre_names = [
            "unknown", "Action", "Adventure", "Animation", "Children's", 
            "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
            "Film-Noir", "Horror", "Musical", "Mystery", "Romance", 
            "Sci-Fi", "Thriller", "War", "Western"
        ]
        
        with open(item_path, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) < 24:
                    continue
                
                movie_id = int(parts[0])
                title = parts[1]
                genre_flags = [int(x) for x in parts[5:24]]
                genres = [genre_names[i] for i, flag in enumerate(genre_flags) if flag == 1]
                
                self.movies[movie_id] = {
                    'title': title,
                    'genres': genres
                }
    
    def _load_ratings(self):
        """Read the u.data file and parse user ratings"""
        data_path = os.path.join(self.data_dir, "u.data")
        
        with open(data_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 4:
                    continue
                
                user_id = int(parts[0])
                movie_id = int(parts[1])
                rating = int(parts[2])
                timestamp = int(parts[3])
                
                self.ratings[user_id].append((movie_id, rating, timestamp))
    
    def generate_test_samples(
        self, 
        n_users: int = 10, 
        n_candidates: int = 50,
        min_history: int = 10,
        seed: int = 42
    ) -> List[Dict]:
        """
        Generate test samples
        
        Args:
            n_users: How many user test samples are generated?
            n_candidates: how many candidate movies each user generates
            min_history: The minimum number of rating records a user needs to have
            seed: random seed
        
        Returns:
            List of test samples, each sample contains:
            - user_id: user ID
            - context: description of movies the user has liked in history
            - candidates: list of candidate movies
            - ground_truth: list of real and related movie IDs (highly rated movies)
        """
        random.seed(seed)
        
        # Filter users with sufficient rating records
        valid_users = [
            uid for uid, ratings in self.ratings.items() 
            if len(ratings) >= min_history + n_candidates
        ]
        
        if len(valid_users) < n_users:
            print(f"Warning: Only {len(valid_users)} users with enough ratings, using all")
            n_users = len(valid_users)
        
        selected_users = random.sample(valid_users, n_users)
        samples = []
        
        for user_id in selected_users:
            user_ratings = self.ratings[user_id]
            # Sort by timestamp
            user_ratings.sort(key=lambda x: x[2])
            
            # Split: the former is used as history and the latter is used as candidate
            history_ratings = user_ratings[:min_history]
            candidate_pool = user_ratings[min_history:]
            
            # Select randomly from candidate pool
            if len(candidate_pool) > n_candidates:
                selected_candidates = random.sample(candidate_pool, n_candidates)
            else:
                selected_candidates = candidate_pool
            
            # Build context: Movies the user likes (rating >= 4)
            liked_movies = [
                self.movies[mid]['title'] 
                for mid, rating, _ in history_ratings 
                if rating >= 4 and mid in self.movies
            ]
            
            context = "User liked movies: " + ", ".join(liked_movies[:5])  # Only take the first 5
            
            # Build candidate list
            candidates = []
            ground_truth = []
            
            for movie_id, rating, _ in selected_candidates:
                if movie_id not in self.movies:
                    continue
                
                movie_info = self.movies[movie_id]
                candidate_text = f"{movie_info['title']} ({', '.join(movie_info['genres'])})"
                candidates.append({
                    'movie_id': movie_id,
                    'text': candidate_text,
                    'rating': rating
                })
                
                if rating >= 4:
                    ground_truth.append(movie_id)
            
            samples.append({
                'user_id': user_id,
                'context': context,
                'candidates': candidates,
                'ground_truth': ground_truth
            })
        
        return samples
    
    def get_stats(self) -> Dict:
        """Get dataset statistics"""
        return {
            'n_movies': len(self.movies),
            'n_users': len(self.ratings),
            'n_ratings': sum(len(ratings) for ratings in self.ratings.values()),
            'avg_ratings_per_user': sum(len(ratings) for ratings in self.ratings.values()) / len(self.ratings)
        }


if __name__ == "__main__":
    # Test data loader
    loader = MovieLensDataLoader()
    
    print("=== MovieLens数据集统计 ===")
    stats = loader.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n=== 生成测试样本 ===")
    samples = loader.generate_test_samples(n_users=3, n_candidates=10)
    
    for i, sample in enumerate(samples):
        print(f"\n样本 {i+1}:")
        print(f"  用户ID: {sample['user_id']}")
        print(f"  上下文: {sample['context'][:100]}...")
        print(f"  候选数量: {len(sample['candidates'])}")
        print(f"  真实相关数量: {len(sample['ground_truth'])}")
        print(f"  前3个候选: {[c['text'] for c in sample['candidates'][:3]]}")

