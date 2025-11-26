"""
Text Statistics for Dataset Evaluation

Provides statistical analysis of text content in datasets.
"""

from typing import List, Dict, Optional, Tuple
from collections import Counter
from dataclasses import dataclass, field
import statistics
import re


@dataclass
class LengthDistribution:
    """Distribution statistics for text lengths"""
    min_length: int
    max_length: int
    mean_length: float
    median_length: float
    std_dev: float
    percentile_25: float
    percentile_75: float
    percentile_90: float
    histogram: Dict[str, int]  # Bucketed counts


@dataclass
class TextStatistics:
    """Comprehensive text statistics"""
    total_examples: int
    total_tokens: int
    total_characters: int
    
    # Question/User message stats
    question_length: LengthDistribution
    
    # Answer/Assistant message stats
    answer_length: LengthDistribution
    
    # Q/A ratio
    avg_qa_ratio: float  # Average answer_length / question_length
    
    # Vocabulary
    vocabulary_size: int
    top_words: List[Tuple[str, int]]  # Top 20 most common words
    
    # Language detection (if available)
    languages: Dict[str, int]
    
    # Special patterns
    has_code_blocks: int  # Examples with code
    has_lists: int  # Examples with bullet/numbered lists
    has_urls: int  # Examples with URLs
    has_numbers: int  # Examples with significant numbers
    
    # Quality indicators
    empty_responses: int
    very_short_responses: int  # < 10 words
    very_long_responses: int  # > 500 words


def calculate_length_distribution(lengths: List[int]) -> LengthDistribution:
    """Calculate distribution statistics for a list of lengths"""
    if not lengths:
        return LengthDistribution(
            min_length=0,
            max_length=0,
            mean_length=0.0,
            median_length=0.0,
            std_dev=0.0,
            percentile_25=0.0,
            percentile_75=0.0,
            percentile_90=0.0,
            histogram={}
        )
    
    sorted_lengths = sorted(lengths)
    n = len(sorted_lengths)
    
    # Calculate percentiles
    def percentile(p: float) -> float:
        k = (n - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < n else f
        return sorted_lengths[f] + (k - f) * (sorted_lengths[c] - sorted_lengths[f])
    
    # Create histogram buckets
    histogram = {}
    buckets = [
        (0, 10, "0-10"),
        (11, 25, "11-25"),
        (26, 50, "26-50"),
        (51, 100, "51-100"),
        (101, 200, "101-200"),
        (201, 500, "201-500"),
        (501, 1000, "501-1000"),
        (1001, float('inf'), "1000+")
    ]
    
    for min_val, max_val, label in buckets:
        count = sum(1 for l in lengths if min_val <= l <= max_val)
        if count > 0:
            histogram[label] = count
    
    return LengthDistribution(
        min_length=min(lengths),
        max_length=max(lengths),
        mean_length=round(statistics.mean(lengths), 2),
        median_length=round(statistics.median(lengths), 2),
        std_dev=round(statistics.stdev(lengths), 2) if len(lengths) > 1 else 0.0,
        percentile_25=round(percentile(25), 2),
        percentile_75=round(percentile(75), 2),
        percentile_90=round(percentile(90), 2),
        histogram=histogram
    )


def extract_messages(example: Dict) -> Tuple[List[str], List[str]]:
    """
    Extract user and assistant messages from an example.
    
    Returns:
        Tuple of (user_messages, assistant_messages)
    """
    user_messages = []
    assistant_messages = []
    
    if "messages" in example:
        for msg in example["messages"]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("user", "human"):
                user_messages.append(content)
            elif role in ("assistant", "bot", "gpt"):
                assistant_messages.append(content)
    elif "conversation" in example:
        for msg in example["conversation"]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("user", "human"):
                user_messages.append(content)
            elif role in ("assistant", "bot", "gpt"):
                assistant_messages.append(content)
    elif "prompt" in example and "completion" in example:
        user_messages.append(example["prompt"])
        assistant_messages.append(example["completion"])
    elif "instruction" in example and "output" in example:
        user_messages.append(example["instruction"])
        assistant_messages.append(example["output"])
    elif "question" in example and "answer" in example:
        user_messages.append(example["question"])
        assistant_messages.append(example["answer"])
    
    return user_messages, assistant_messages


def count_words(text: str) -> int:
    """Count words in text"""
    return len(text.split())


def has_code_block(text: str) -> bool:
    """Check if text contains code blocks"""
    return "```" in text or bool(re.search(r'`[^`]+`', text))


def has_list(text: str) -> bool:
    """Check if text contains bullet or numbered lists"""
    patterns = [
        r'^\s*[-*•]\s+',  # Bullet points
        r'^\s*\d+\.\s+',  # Numbered lists
        r'\n\s*[-*•]\s+',
        r'\n\s*\d+\.\s+'
    ]
    return any(re.search(p, text, re.MULTILINE) for p in patterns)


def has_url(text: str) -> bool:
    """Check if text contains URLs"""
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return bool(re.search(url_pattern, text))


def has_significant_numbers(text: str) -> bool:
    """Check if text contains significant numbers (not just single digits)"""
    # Look for numbers with more than 2 digits or decimal numbers
    return bool(re.search(r'\d{3,}|\d+\.\d+', text))


def detect_language_simple(text: str) -> str:
    """
    Simple language detection based on character patterns.
    Returns 'en', 'ru', 'zh', 'other', or 'mixed'
    """
    # Count character types
    cyrillic = len(re.findall(r'[а-яА-ЯёЁ]', text))
    latin = len(re.findall(r'[a-zA-Z]', text))
    chinese = len(re.findall(r'[\u4e00-\u9fff]', text))
    
    total = cyrillic + latin + chinese
    if total == 0:
        return "other"
    
    cyrillic_ratio = cyrillic / total
    latin_ratio = latin / total
    chinese_ratio = chinese / total
    
    if chinese_ratio > 0.3:
        return "zh"
    elif cyrillic_ratio > 0.5:
        return "ru"
    elif latin_ratio > 0.5:
        return "en"
    elif cyrillic_ratio > 0.2 and latin_ratio > 0.2:
        return "mixed"
    else:
        return "other"


def calculate_text_stats(examples: List[Dict]) -> TextStatistics:
    """
    Calculate comprehensive text statistics for a dataset.
    
    Args:
        examples: List of examples in chat format
    
    Returns:
        TextStatistics with all metrics
    """
    if not examples:
        return TextStatistics(
            total_examples=0,
            total_tokens=0,
            total_characters=0,
            question_length=calculate_length_distribution([]),
            answer_length=calculate_length_distribution([]),
            avg_qa_ratio=0.0,
            vocabulary_size=0,
            top_words=[],
            languages={},
            has_code_blocks=0,
            has_lists=0,
            has_urls=0,
            has_numbers=0,
            empty_responses=0,
            very_short_responses=0,
            very_long_responses=0
        )
    
    question_lengths = []
    answer_lengths = []
    all_words = []
    languages = Counter()
    
    has_code = 0
    has_list_count = 0
    has_url_count = 0
    has_nums = 0
    empty_count = 0
    very_short = 0
    very_long = 0
    
    total_chars = 0
    qa_ratios = []
    
    for example in examples:
        user_msgs, assistant_msgs = extract_messages(example)
        
        # Process user messages
        for msg in user_msgs:
            words = count_words(msg)
            question_lengths.append(words)
            all_words.extend(msg.lower().split())
            total_chars += len(msg)
        
        # Process assistant messages
        for msg in assistant_msgs:
            words = count_words(msg)
            answer_lengths.append(words)
            all_words.extend(msg.lower().split())
            total_chars += len(msg)
            
            # Check patterns
            if has_code_block(msg):
                has_code += 1
            if has_list(msg):
                has_list_count += 1
            if has_url(msg):
                has_url_count += 1
            if has_significant_numbers(msg):
                has_nums += 1
            
            # Quality checks
            if words == 0:
                empty_count += 1
            elif words < 10:
                very_short += 1
            elif words > 500:
                very_long += 1
            
            # Language detection
            lang = detect_language_simple(msg)
            languages[lang] += 1
        
        # Calculate Q/A ratio
        if user_msgs and assistant_msgs:
            q_len = sum(count_words(m) for m in user_msgs)
            a_len = sum(count_words(m) for m in assistant_msgs)
            if q_len > 0:
                qa_ratios.append(a_len / q_len)
    
    # Calculate vocabulary
    word_counts = Counter(all_words)
    vocabulary_size = len(word_counts)
    top_words = word_counts.most_common(20)
    
    return TextStatistics(
        total_examples=len(examples),
        total_tokens=len(all_words),
        total_characters=total_chars,
        question_length=calculate_length_distribution(question_lengths),
        answer_length=calculate_length_distribution(answer_lengths),
        avg_qa_ratio=round(statistics.mean(qa_ratios), 2) if qa_ratios else 0.0,
        vocabulary_size=vocabulary_size,
        top_words=top_words,
        languages=dict(languages),
        has_code_blocks=has_code,
        has_lists=has_list_count,
        has_urls=has_url_count,
        has_numbers=has_nums,
        empty_responses=empty_count,
        very_short_responses=very_short,
        very_long_responses=very_long
    )
