# src/eval/metrics.py
from typing import List, Set, Dict
import numpy as np


def ndcg_at_k(recommended: List[int], ground_truth: Set[int], k: int = 10) -> float:
    if not ground_truth:
        return 0.0
    recommended = recommended[:k]
    dcg = 0.0
    for i, item in enumerate(recommended):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 2)  # positions starting at 1
    ideal_hits = min(len(ground_truth), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return float(dcg / idcg) if idcg > 0 else 0.0


def recall_at_k(recommended: List[int], ground_truth: Set[int], k: int = 20) -> float:
    if not ground_truth:
        return 0.0
    recommended = set(recommended[:k])
    hits = len(recommended & ground_truth)
    return float(hits / len(ground_truth))


def coverage(all_recommendations: Dict[int, List[int]], all_items: List[int]) -> float:
    """
    all_recommendations: dict user_id -> list[item_id]
    """
    recommended_items = set()
    for recs in all_recommendations.values():
        recommended_items.update(recs)
    return float(len(recommended_items) / len(set(all_items))) if all_items else 0.0
