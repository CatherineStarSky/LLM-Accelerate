#!/usr/bin/env python
"""Evaluation metrics for ranking quality: NDCG, MRR, Recall"""

import numpy as np
from typing import List, Dict, Tuple, Optional


def dcg(relevance_scores: List[float], k: Optional[int] = None) -> float:
    """
    Calculate Discounted Cumulative Gain (DCG)
    
    Args:
        relevance_scores: List of relevance scores (higher is better)
        k: Cutoff position (if None, use all positions)
    
    Returns:
        DCG score
    """
    if k is not None:
        relevance_scores = relevance_scores[:k]
    
    if len(relevance_scores) == 0:
        return 0.0
    
    # DCG = sum(relevance[i] / log2(i + 2)) for i in range(len)
    dcg_score = 0.0
    for i, rel in enumerate(relevance_scores):
        dcg_score += rel / np.log2(i + 2)  # i+2 because position starts from 1
    
    return dcg_score


def ndcg(
    y_true: List[float],
    y_pred: List[float],
    k: Optional[int] = None,
    relevance_type: str = 'binary'
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG)
    
    Args:
        y_true: True relevance scores (can be binary 0/1 or continuous ratings)
        y_pred: Predicted scores for ranking
        k: Cutoff position for NDCG@k (if None, use all positions)
        relevance_type: 'binary' for 0/1 relevance, 'continuous' for ratings
    
    Returns:
        NDCG score (0.0 to 1.0)
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred must have same length: {len(y_true)} vs {len(y_pred)}")
    
    if len(y_true) == 0:
        return 0.0
    
    # Sort by predicted scores (descending)
    sorted_indices = np.argsort(y_pred)[::-1]
    sorted_relevance = [y_true[i] for i in sorted_indices]
    
    # Calculate DCG for predicted ranking
    dcg_pred = dcg(sorted_relevance, k=k)
    
    # Calculate IDCG (Ideal DCG) - perfect ranking
    ideal_relevance = sorted(y_true, reverse=True)
    idcg = dcg(ideal_relevance, k=k)
    
    # NDCG = DCG / IDCG
    if idcg == 0.0:
        return 0.0
    
    return dcg_pred / idcg


def ndcg_at_k(
    candidates: List[Dict],
    scores: List[float],
    k: int = 10,
    rating_threshold: float = 4.0
) -> float:
    """
    Calculate NDCG@k for a single query
    
    Args:
        candidates: List of candidate dictionaries, each must have 'rating' field
        scores: Predicted relevance scores (same order as candidates)
        k: Cutoff position for NDCG@k
        rating_threshold: Rating threshold for binary relevance (>= threshold is relevant)
    
    Returns:
        NDCG@k score
    """
    if len(candidates) != len(scores):
        raise ValueError(f"Candidates and scores length mismatch: {len(candidates)} vs {len(scores)}")
    
    if len(candidates) == 0:
        return 0.0
    
    # Extract true relevance (binary: 1 if rating >= threshold, else 0)
    y_true = [1.0 if cand.get('rating', 0) >= rating_threshold else 0.0 for cand in candidates]
    
    # Calculate NDCG@k
    return ndcg(y_true, scores, k=k, relevance_type='binary')


def ndcg_at_k_continuous(
    candidates: List[Dict],
    scores: List[float],
    k: int = 10
) -> float:
    """
    Calculate NDCG@k using continuous ratings (not binary)
    
    Args:
        candidates: List of candidate dictionaries, each must have 'rating' field
        scores: Predicted relevance scores (same order as candidates)
        k: Cutoff position for NDCG@k
    
    Returns:
        NDCG@k score using continuous ratings
    """
    if len(candidates) != len(scores):
        raise ValueError(f"Candidates and scores length mismatch: {len(candidates)} vs {len(scores)}")
    
    if len(candidates) == 0:
        return 0.0
    
    # Extract true relevance (use rating as continuous relevance score)
    y_true = [float(cand.get('rating', 0)) for cand in candidates]
    
    # Normalize ratings to [0, 1] range (assuming ratings are 1-5)
    y_true = [r / 5.0 for r in y_true]
    
    # Calculate NDCG@k
    return ndcg(y_true, scores, k=k, relevance_type='continuous')


def mrr(y_true: List[int], y_pred: List[float], k: Optional[int] = None) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR)
    
    Args:
        y_true: Binary relevance labels (1 for relevant, 0 for irrelevant)
        y_pred: Predicted scores for ranking
        k: Cutoff position for MRR@k (if None, use all positions)
    
    Returns:
        MRR score (0.0 to 1.0)
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred must have same length")
    
    if len(y_true) == 0:
        return 0.0
    
    # Sort by predicted scores (descending)
    sorted_indices = np.argsort(y_pred)[::-1]
    
    # Find the rank of the first relevant item
    if k is not None:
        sorted_indices = sorted_indices[:k]
    
    for rank, idx in enumerate(sorted_indices, start=1):
        if y_true[idx] == 1:
            return 1.0 / rank
    
    return 0.0


def mrr_at_k(
    candidates: List[Dict],
    scores: List[float],
    k: int = 10,
    rating_threshold: float = 4.0
) -> float:
    """
    Calculate MRR@k for a single query
    
    Args:
        candidates: List of candidate dictionaries, each must have 'rating' field
        scores: Predicted relevance scores (same order as candidates)
        k: Cutoff position for MRR@k
        rating_threshold: Rating threshold for binary relevance
    
    Returns:
        MRR@k score
    """
    if len(candidates) != len(scores):
        raise ValueError(f"Candidates and scores length mismatch")
    
    if len(candidates) == 0:
        return 0.0
    
    # Extract binary relevance
    y_true = [1 if cand.get('rating', 0) >= rating_threshold else 0 for cand in candidates]
    
    return mrr(y_true, scores, k=k)


def recall_at_k(
    candidates: List[Dict],
    scores: List[float],
    k: int = 50,
    rating_threshold: float = 4.0
) -> float:
    """
    Calculate Recall@k for a single query
    
    Args:
        candidates: List of candidate dictionaries, each must have 'rating' field
        scores: Predicted relevance scores (same order as candidates)
        k: Cutoff position for Recall@k
        rating_threshold: Rating threshold for binary relevance
    
    Returns:
        Recall@k score (0.0 to 1.0)
    """
    if len(candidates) != len(scores):
        raise ValueError(f"Candidates and scores length mismatch")
    
    if len(candidates) == 0:
        return 0.0
    
    # Extract binary relevance
    y_true = [1 if cand.get('rating', 0) >= rating_threshold else 0 for cand in candidates]
    
    # Count total relevant items
    total_relevant = sum(y_true)
    if total_relevant == 0:
        return 0.0
    
    # Sort by predicted scores and get top-k
    sorted_indices = np.argsort(scores)[::-1][:k]
    
    # Count relevant items in top-k
    relevant_in_topk = sum(y_true[i] for i in sorted_indices)
    
    return relevant_in_topk / total_relevant


def evaluate_ranking_quality(
    candidates: List[Dict],
    scores: List[float],
    k_list: List[int] = [10, 50],
    rating_threshold: float = 4.0
) -> Dict[str, float]:
    """
    Evaluate ranking quality with multiple metrics
    
    Args:
        candidates: List of candidate dictionaries with 'rating' field
        scores: Predicted relevance scores
        k_list: List of k values for metrics (e.g., [10, 50])
        rating_threshold: Rating threshold for binary relevance
    
    Returns:
        Dictionary of metric scores
    """
    results = {}
    
    for k in k_list:
        # NDCG@k (binary)
        results[f'ndcg@{k}'] = ndcg_at_k(candidates, scores, k=k, rating_threshold=rating_threshold)
        
        # NDCG@k (continuous)
        results[f'ndcg@{k}_continuous'] = ndcg_at_k_continuous(candidates, scores, k=k)
        
        # MRR@k
        results[f'mrr@{k}'] = mrr_at_k(candidates, scores, k=k, rating_threshold=rating_threshold)
        
        # Recall@k
        results[f'recall@{k}'] = recall_at_k(candidates, scores, k=k, rating_threshold=rating_threshold)
    
    return results


def evaluate_batch(
    samples: List[Dict],
    all_scores: List[List[float]],
    k_list: List[int] = [10, 50],
    rating_threshold: float = 4.0
) -> Dict[str, float]:
    """
    Evaluate ranking quality for a batch of queries
    
    Args:
        samples: List of test samples, each containing 'candidates' field
        all_scores: List of score lists, one per sample
        k_list: List of k values for metrics
        rating_threshold: Rating threshold for binary relevance
    
    Returns:
        Dictionary of average metric scores across all queries
    """
    if len(samples) != len(all_scores):
        raise ValueError(f"Samples and scores length mismatch: {len(samples)} vs {len(all_scores)}")
    
    if len(samples) == 0:
        return {}
    
    # Collect all metric values
    metric_values = {}
    for k in k_list:
        metric_values[f'ndcg@{k}'] = []
        metric_values[f'ndcg@{k}_continuous'] = []
        metric_values[f'mrr@{k}'] = []
        metric_values[f'recall@{k}'] = []
    
    # Evaluate each sample
    for sample, scores in zip(samples, all_scores):
        sample_metrics = evaluate_ranking_quality(
            sample['candidates'],
            scores,
            k_list=k_list,
            rating_threshold=rating_threshold
        )
        
        for metric_name, value in sample_metrics.items():
            metric_values[metric_name].append(value)
    
    # Calculate averages
    results = {}
    for metric_name, values in metric_values.items():
        results[f'avg_{metric_name}'] = np.mean(values)
        results[f'std_{metric_name}'] = np.std(values)
    
    return results


if __name__ == "__main__":
    # Test metrics
    print("=== Testing Ranking Metrics ===\n")
    
    # Test data
    candidates = [
        {'rating': 5, 'text': 'Movie A'},
        {'rating': 1, 'text': 'Movie B'},
        {'rating': 4, 'text': 'Movie C'},
        {'rating': 2, 'text': 'Movie D'},
        {'rating': 5, 'text': 'Movie E'},
        {'rating': 3, 'text': 'Movie F'},
    ]
    
    # Perfect ranking (scores match ratings)
    perfect_scores = [5.0, 1.0, 4.0, 2.0, 5.0, 3.0]
    
    # Random ranking
    random_scores = [2.1, 3.5, 1.8, 4.2, 2.9, 3.1]
    
    print("1. Perfect Ranking:")
    perfect_metrics = evaluate_ranking_quality(candidates, perfect_scores, k_list=[5, 10])
    for metric, value in perfect_metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    print("\n2. Random Ranking:")
    random_metrics = evaluate_ranking_quality(candidates, random_scores, k_list=[5, 10])
    for metric, value in random_metrics.items():
        print(f"   {metric}: {value:.4f}")

