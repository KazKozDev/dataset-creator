"""
Diversity Metrics for Dataset Evaluation

Implements Distinct-n and Self-BLEU metrics for measuring dataset diversity.
"""

from typing import List, Dict, Tuple, Optional
from collections import Counter
from dataclasses import dataclass
import random
import math


@dataclass
class DiversityMetrics:
    """Container for diversity metrics"""
    distinct_1: float  # Ratio of unique unigrams
    distinct_2: float  # Ratio of unique bigrams
    distinct_3: float  # Ratio of unique trigrams
    self_bleu_1: float  # Self-BLEU with unigrams
    self_bleu_2: float  # Self-BLEU with bigrams
    self_bleu_3: float  # Self-BLEU with trigrams
    self_bleu_avg: float  # Average Self-BLEU
    vocabulary_size: int
    total_tokens: int
    diversity_score: float  # Overall diversity score (0-1, higher = more diverse)


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization with lowercasing"""
    text = text.lower()
    # Remove punctuation and split
    words = text.split()
    return [w.strip('.,!?;:()[]{}"\'-') for w in words if w.strip('.,!?;:()[]{}"\'-')]


def get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Extract n-grams from token list"""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def calculate_distinct_n(texts: List[str], n: int = 1) -> float:
    """
    Calculate Distinct-n metric.
    
    Distinct-n = (number of unique n-grams) / (total number of n-grams)
    
    Higher values indicate more diverse text.
    
    Args:
        texts: List of text strings
        n: n-gram size (1, 2, or 3)
    
    Returns:
        Distinct-n score between 0 and 1
    """
    all_ngrams = []
    for text in texts:
        tokens = tokenize(text)
        ngrams = get_ngrams(tokens, n)
        all_ngrams.extend(ngrams)
    
    if not all_ngrams:
        return 0.0
    
    unique_ngrams = len(set(all_ngrams))
    total_ngrams = len(all_ngrams)
    
    return unique_ngrams / total_ngrams


def calculate_bleu_score(
    reference: List[str],
    hypothesis: List[str],
    max_n: int = 4
) -> float:
    """
    Calculate BLEU score between reference and hypothesis.
    
    Simplified implementation without smoothing.
    """
    if not hypothesis or not reference:
        return 0.0
    
    # Calculate n-gram precisions
    precisions = []
    for n in range(1, min(max_n + 1, len(hypothesis) + 1)):
        ref_ngrams = Counter(get_ngrams(reference, n))
        hyp_ngrams = Counter(get_ngrams(hypothesis, n))
        
        # Count matches
        matches = 0
        for ngram, count in hyp_ngrams.items():
            matches += min(count, ref_ngrams.get(ngram, 0))
        
        total = sum(hyp_ngrams.values())
        if total > 0:
            precisions.append(matches / total)
        else:
            precisions.append(0.0)
    
    if not precisions or all(p == 0 for p in precisions):
        return 0.0
    
    # Geometric mean of precisions
    log_precisions = [math.log(p) if p > 0 else -float('inf') for p in precisions]
    avg_log = sum(log_precisions) / len(log_precisions)
    
    if avg_log == -float('inf'):
        return 0.0
    
    # Brevity penalty
    ref_len = len(reference)
    hyp_len = len(hypothesis)
    if hyp_len >= ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0
    
    return bp * math.exp(avg_log)


def calculate_self_bleu(
    texts: List[str],
    n: int = 4,
    sample_size: Optional[int] = 500
) -> float:
    """
    Calculate Self-BLEU score for a set of texts.
    
    Self-BLEU measures how similar generated texts are to each other.
    Lower Self-BLEU indicates more diverse outputs.
    
    For each text, we calculate BLEU score using all other texts as references,
    then average across all texts.
    
    Args:
        texts: List of text strings
        n: Maximum n-gram order for BLEU
        sample_size: If set, randomly sample this many texts for efficiency
    
    Returns:
        Average Self-BLEU score (0-1, lower = more diverse)
    """
    if len(texts) < 2:
        return 0.0
    
    # Sample if dataset is large
    if sample_size and len(texts) > sample_size:
        texts = random.sample(texts, sample_size)
    
    # Tokenize all texts
    tokenized = [tokenize(text) for text in texts]
    
    bleu_scores = []
    for i, hypothesis in enumerate(tokenized):
        if not hypothesis:
            continue
        
        # Use all other texts as references
        references = [t for j, t in enumerate(tokenized) if j != i and t]
        
        if not references:
            continue
        
        # Calculate BLEU against each reference and take max
        max_bleu = 0.0
        for reference in references:
            bleu = calculate_bleu_score(reference, hypothesis, max_n=n)
            max_bleu = max(max_bleu, bleu)
        
        bleu_scores.append(max_bleu)
    
    if not bleu_scores:
        return 0.0
    
    return sum(bleu_scores) / len(bleu_scores)


def analyze_diversity(texts: List[str], sample_size: int = 500) -> DiversityMetrics:
    """
    Comprehensive diversity analysis of texts.
    
    Args:
        texts: List of text strings
        sample_size: Sample size for Self-BLEU calculation
    
    Returns:
        DiversityMetrics with all diversity scores
    """
    if not texts:
        return DiversityMetrics(
            distinct_1=0.0,
            distinct_2=0.0,
            distinct_3=0.0,
            self_bleu_1=0.0,
            self_bleu_2=0.0,
            self_bleu_3=0.0,
            self_bleu_avg=0.0,
            vocabulary_size=0,
            total_tokens=0,
            diversity_score=0.0
        )
    
    # Calculate Distinct-n
    distinct_1 = calculate_distinct_n(texts, n=1)
    distinct_2 = calculate_distinct_n(texts, n=2)
    distinct_3 = calculate_distinct_n(texts, n=3)
    
    # Calculate Self-BLEU (sample for efficiency)
    self_bleu_1 = calculate_self_bleu(texts, n=1, sample_size=sample_size)
    self_bleu_2 = calculate_self_bleu(texts, n=2, sample_size=sample_size)
    self_bleu_3 = calculate_self_bleu(texts, n=3, sample_size=sample_size)
    self_bleu_avg = (self_bleu_1 + self_bleu_2 + self_bleu_3) / 3
    
    # Calculate vocabulary stats
    all_tokens = []
    for text in texts:
        all_tokens.extend(tokenize(text))
    
    vocabulary_size = len(set(all_tokens))
    total_tokens = len(all_tokens)
    
    # Calculate overall diversity score
    # Higher Distinct-n is better, lower Self-BLEU is better
    distinct_avg = (distinct_1 + distinct_2 + distinct_3) / 3
    diversity_score = (distinct_avg + (1 - self_bleu_avg)) / 2
    
    return DiversityMetrics(
        distinct_1=round(distinct_1, 4),
        distinct_2=round(distinct_2, 4),
        distinct_3=round(distinct_3, 4),
        self_bleu_1=round(self_bleu_1, 4),
        self_bleu_2=round(self_bleu_2, 4),
        self_bleu_3=round(self_bleu_3, 4),
        self_bleu_avg=round(self_bleu_avg, 4),
        vocabulary_size=vocabulary_size,
        total_tokens=total_tokens,
        diversity_score=round(diversity_score, 4)
    )
