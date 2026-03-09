#!/usr/bin/env python
"""Test script for NDCG metrics"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.metrics import (
    ndcg_at_k, 
    ndcg_at_k_continuous,
    mrr_at_k,
    recall_at_k,
    evaluate_ranking_quality,
    evaluate_batch
)

def test_metrics():
    """Test NDCG and other ranking metrics"""
    print("=== Testing Ranking Metrics ===\n")
    
    # Test case 1: Perfect ranking
    print("1. Perfect Ranking (scores match ratings):")
    candidates_perfect = [
        {'rating': 5, 'text': 'Movie A'},
        {'rating': 5, 'text': 'Movie B'},
        {'rating': 4, 'text': 'Movie C'},
        {'rating': 4, 'text': 'Movie D'},
        {'rating': 3, 'text': 'Movie E'},
        {'rating': 2, 'text': 'Movie F'},
        {'rating': 1, 'text': 'Movie G'},
    ]
    perfect_scores = [5.0, 5.0, 4.0, 4.0, 3.0, 2.0, 1.0]
    
    metrics = evaluate_ranking_quality(candidates_perfect, perfect_scores, k_list=[5, 10])
    for metric, value in sorted(metrics.items()):
        print(f"   {metric}: {value:.4f}")
    
    # Test case 2: Random ranking
    print("\n2. Random Ranking:")
    candidates_random = [
        {'rating': 5, 'text': 'Movie A'},
        {'rating': 1, 'text': 'Movie B'},
        {'rating': 4, 'text': 'Movie C'},
        {'rating': 2, 'text': 'Movie D'},
        {'rating': 5, 'text': 'Movie E'},
        {'rating': 3, 'text': 'Movie F'},
    ]
    random_scores = [2.1, 3.5, 1.8, 4.2, 2.9, 3.1]
    
    metrics = evaluate_ranking_quality(candidates_random, random_scores, k_list=[5, 10])
    for metric, value in sorted(metrics.items()):
        print(f"   {metric}: {value:.4f}")
    
    # Test case 3: Worst case (reverse ranking)
    print("\n3. Worst Case (reverse ranking):")
    worst_scores = [1.0, 2.0, 3.0, 4.0, 5.0, 5.0]  # Reverse order
    
    metrics = evaluate_ranking_quality(candidates_random, worst_scores, k_list=[5, 10])
    for metric, value in sorted(metrics.items()):
        print(f"   {metric}: {value:.4f}")
    
    # Test case 4: Batch evaluation
    print("\n4. Batch Evaluation:")
    samples = [
        {
            'candidates': candidates_perfect,
            'context': 'Test context 1'
        },
        {
            'candidates': candidates_random,
            'context': 'Test context 2'
        }
    ]
    all_scores = [perfect_scores, random_scores]
    
    batch_metrics = evaluate_batch(samples, all_scores, k_list=[5, 10])
    print("   Average metrics across batch:")
    for metric, value in sorted(batch_metrics.items()):
        print(f"   {metric}: {value:.4f}")
    
    print("\n✓ All tests completed!")


if __name__ == "__main__":
    test_metrics()

