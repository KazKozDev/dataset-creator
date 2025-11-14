"""
Deduplication and Data Cleaning Module
Handles duplicate removal, data validation, and quality filtering
"""

import hashlib
import json
import re
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import Counter
from difflib import SequenceMatcher
import unicodedata


class DataDeduplicator:
    """Remove duplicates from datasets using various strategies"""

    def __init__(self, similarity_threshold: float = 0.9):
        self.similarity_threshold = similarity_threshold
        self.seen_hashes: Set[str] = set()
        self.seen_fuzzy: List[str] = []

    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove punctuation (optional)
        text = re.sub(r'[^\w\s]', '', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)

        return text

    def hash_content(self, content: str, method: str = 'md5') -> str:
        """Generate hash of content"""
        content_bytes = content.encode('utf-8')

        if method == 'md5':
            return hashlib.md5(content_bytes).hexdigest()
        elif method == 'sha256':
            return hashlib.sha256(content_bytes).hexdigest()
        else:
            raise ValueError(f"Unknown hash method: {method}")

    def extract_text_from_example(self, example: Dict[str, Any], format_type: str) -> str:
        """Extract comparable text from example"""
        if format_type == 'chat':
            messages = example.get('messages', [])
            # Concatenate all message contents
            texts = [msg.get('content', '') for msg in messages]
            return ' '.join(texts)

        elif format_type == 'instruction':
            instruction = example.get('instruction', '')
            output = example.get('output', '')
            return f"{instruction} {output}"

        else:
            # Try to convert to string
            return json.dumps(example, sort_keys=True)

    def is_exact_duplicate(
        self,
        example: Dict[str, Any],
        format_type: str,
        use_hash: bool = True
    ) -> bool:
        """Check if example is an exact duplicate"""
        text = self.extract_text_from_example(example, format_type)
        normalized = self.normalize_text(text)

        if use_hash:
            content_hash = self.hash_content(normalized)

            if content_hash in self.seen_hashes:
                return True

            self.seen_hashes.add(content_hash)
            return False
        else:
            # Direct string comparison
            if normalized in self.seen_fuzzy:
                return True

            self.seen_fuzzy.append(normalized)
            return False

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        return SequenceMatcher(None, text1, text2).ratio()

    def is_fuzzy_duplicate(
        self,
        example: Dict[str, Any],
        format_type: str
    ) -> bool:
        """Check if example is a fuzzy duplicate"""
        text = self.extract_text_from_example(example, format_type)
        normalized = self.normalize_text(text)

        # Compare with previously seen examples
        for seen_text in self.seen_fuzzy:
            similarity = self.calculate_similarity(normalized, seen_text)

            if similarity >= self.similarity_threshold:
                return True

        # Not a duplicate, add to seen
        self.seen_fuzzy.append(normalized)
        return False

    def deduplicate_dataset(
        self,
        examples: List[Dict[str, Any]],
        format_type: str,
        method: str = 'exact',
        keep_first: bool = True
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Remove duplicates from dataset

        Args:
            examples: List of examples to deduplicate
            format_type: Format type ('chat' or 'instruction')
            method: Deduplication method ('exact', 'fuzzy', 'hash')
            keep_first: Keep first occurrence of duplicate (vs last)

        Returns:
            Tuple of (deduplicated examples, statistics)
        """
        # Reset state
        self.seen_hashes.clear()
        self.seen_fuzzy.clear()

        unique_examples = []
        duplicates_count = 0

        if not keep_first:
            # Reverse to keep last occurrence
            examples = list(reversed(examples))

        for example in examples:
            is_duplicate = False

            if method == 'exact':
                is_duplicate = self.is_exact_duplicate(example, format_type, use_hash=False)
            elif method == 'hash':
                is_duplicate = self.is_exact_duplicate(example, format_type, use_hash=True)
            elif method == 'fuzzy':
                is_duplicate = self.is_fuzzy_duplicate(example, format_type)
            else:
                raise ValueError(f"Unknown deduplication method: {method}")

            if not is_duplicate:
                unique_examples.append(example)
            else:
                duplicates_count += 1

        if not keep_first:
            # Reverse back to original order
            unique_examples = list(reversed(unique_examples))

        stats = {
            'original_count': len(examples),
            'unique_count': len(unique_examples),
            'duplicates_removed': duplicates_count,
            'deduplication_rate': duplicates_count / len(examples) if examples else 0
        }

        return unique_examples, stats


class DataCleaner:
    """Clean and validate dataset examples"""

    def __init__(self):
        pass

    def is_valid_chat_example(self, example: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate chat format example"""
        if 'messages' not in example:
            return False, "Missing 'messages' field"

        messages = example['messages']

        if not isinstance(messages, list):
            return False, "'messages' is not a list"

        if len(messages) < 2:
            return False, "Too few messages (need at least 2)"

        # Check message structure
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                return False, f"Message {i} is not an object"

            if 'role' not in msg:
                return False, f"Message {i} missing 'role'"

            if 'content' not in msg:
                return False, f"Message {i} missing 'content'"

            if msg['role'] not in ['user', 'assistant', 'system']:
                return False, f"Message {i} has invalid role: {msg['role']}"

            # Check content is non-empty string
            if not isinstance(msg['content'], str) or not msg['content'].strip():
                return False, f"Message {i} has empty or invalid content"

        return True, None

    def is_valid_instruction_example(self, example: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate instruction format example"""
        if 'instruction' not in example:
            return False, "Missing 'instruction' field"

        if 'output' not in example:
            return False, "Missing 'output' field"

        if not isinstance(example['instruction'], str) or not example['instruction'].strip():
            return False, "Invalid or empty 'instruction'"

        if not isinstance(example['output'], str) or not example['output'].strip():
            return False, "Invalid or empty 'output'"

        return True, None

    def check_minimum_length(
        self,
        example: Dict[str, Any],
        format_type: str,
        min_length: int = 10
    ) -> bool:
        """Check if example meets minimum length requirement"""
        if format_type == 'chat':
            messages = example.get('messages', [])
            for msg in messages:
                content = msg.get('content', '')
                if len(content.strip()) < min_length:
                    return False
            return True

        elif format_type == 'instruction':
            instruction = example.get('instruction', '')
            output = example.get('output', '')
            return len(instruction.strip()) >= min_length and len(output.strip()) >= min_length

        return True

    def check_maximum_length(
        self,
        example: Dict[str, Any],
        format_type: str,
        max_length: int = 10000
    ) -> bool:
        """Check if example doesn't exceed maximum length"""
        if format_type == 'chat':
            messages = example.get('messages', [])
            for msg in messages:
                content = msg.get('content', '')
                if len(content) > max_length:
                    return False
            return True

        elif format_type == 'instruction':
            instruction = example.get('instruction', '')
            output = example.get('output', '')
            return len(instruction) <= max_length and len(output) <= max_length

        return True

    def contains_toxic_content(self, text: str, toxic_patterns: Optional[List[str]] = None) -> bool:
        """Check if text contains toxic/inappropriate content"""
        if toxic_patterns is None:
            # Basic toxic patterns (expand as needed)
            toxic_patterns = [
                r'\b(offensive|inappropriate|nsfw)\b',
                # Add more patterns as needed
            ]

        for pattern in toxic_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def clean_example(
        self,
        example: Dict[str, Any],
        format_type: str,
        remove_extra_whitespace: bool = True,
        fix_encoding: bool = True
    ) -> Dict[str, Any]:
        """Clean and normalize an example"""
        cleaned = example.copy()

        def clean_text(text: str) -> str:
            if remove_extra_whitespace:
                # Remove extra whitespace
                text = re.sub(r'\s+', ' ', text)
                text = text.strip()

            if fix_encoding:
                # Fix common encoding issues
                text = text.encode('utf-8', errors='ignore').decode('utf-8')

            return text

        if format_type == 'chat':
            messages = cleaned.get('messages', [])
            for msg in messages:
                if 'content' in msg and isinstance(msg['content'], str):
                    msg['content'] = clean_text(msg['content'])

        elif format_type == 'instruction':
            if 'instruction' in cleaned and isinstance(cleaned['instruction'], str):
                cleaned['instruction'] = clean_text(cleaned['instruction'])

            if 'output' in cleaned and isinstance(cleaned['output'], str):
                cleaned['output'] = clean_text(cleaned['output'])

        return cleaned

    def filter_dataset(
        self,
        examples: List[Dict[str, Any]],
        format_type: str,
        min_length: int = 10,
        max_length: int = 10000,
        remove_toxic: bool = True,
        toxic_patterns: Optional[List[str]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Filter dataset based on quality criteria

        Returns:
            Tuple of (filtered examples, statistics)
        """
        filtered_examples = []
        stats = {
            'original_count': len(examples),
            'invalid_structure': 0,
            'too_short': 0,
            'too_long': 0,
            'toxic_content': 0,
            'valid_count': 0
        }

        for example in examples:
            # Validate structure
            if format_type == 'chat':
                is_valid, error = self.is_valid_chat_example(example)
            elif format_type == 'instruction':
                is_valid, error = self.is_valid_instruction_example(example)
            else:
                is_valid, error = True, None

            if not is_valid:
                stats['invalid_structure'] += 1
                continue

            # Check length constraints
            if not self.check_minimum_length(example, format_type, min_length):
                stats['too_short'] += 1
                continue

            if not self.check_maximum_length(example, format_type, max_length):
                stats['too_long'] += 1
                continue

            # Check for toxic content
            if remove_toxic:
                text = self.extract_text_for_toxicity_check(example, format_type)
                if self.contains_toxic_content(text, toxic_patterns):
                    stats['toxic_content'] += 1
                    continue

            # Example passed all filters
            cleaned = self.clean_example(example, format_type)
            filtered_examples.append(cleaned)
            stats['valid_count'] += 1

        return filtered_examples, stats

    def extract_text_for_toxicity_check(self, example: Dict[str, Any], format_type: str) -> str:
        """Extract text from example for toxicity checking"""
        if format_type == 'chat':
            messages = example.get('messages', [])
            texts = [msg.get('content', '') for msg in messages]
            return ' '.join(texts)

        elif format_type == 'instruction':
            instruction = example.get('instruction', '')
            output = example.get('output', '')
            return f"{instruction} {output}"

        return ""


class DataAnalyzer:
    """Analyze dataset statistics and quality"""

    def __init__(self):
        pass

    def calculate_length_stats(
        self,
        examples: List[Dict[str, Any]],
        format_type: str
    ) -> Dict[str, Any]:
        """Calculate length statistics for dataset"""
        lengths = []

        for example in examples:
            if format_type == 'chat':
                messages = example.get('messages', [])
                for msg in messages:
                    content = msg.get('content', '')
                    lengths.append(len(content))

            elif format_type == 'instruction':
                instruction = example.get('instruction', '')
                output = example.get('output', '')
                lengths.append(len(instruction))
                lengths.append(len(output))

        if not lengths:
            return {}

        return {
            'min': min(lengths),
            'max': max(lengths),
            'mean': sum(lengths) / len(lengths),
            'median': sorted(lengths)[len(lengths) // 2],
            'total_chars': sum(lengths)
        }

    def analyze_metadata(
        self,
        examples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze metadata across dataset"""
        metadata_keys = Counter()
        metadata_values = {}

        for example in examples:
            metadata = example.get('metadata', {})

            for key, value in metadata.items():
                metadata_keys[key] += 1

                if key not in metadata_values:
                    metadata_values[key] = Counter()

                metadata_values[key][str(value)] += 1

        return {
            'metadata_keys': dict(metadata_keys),
            'metadata_values': {k: dict(v) for k, v in metadata_values.items()}
        }

    def analyze_dataset(
        self,
        examples: List[Dict[str, Any]],
        format_type: str
    ) -> Dict[str, Any]:
        """Comprehensive dataset analysis"""
        return {
            'total_examples': len(examples),
            'format_type': format_type,
            'length_stats': self.calculate_length_stats(examples, format_type),
            'metadata_analysis': self.analyze_metadata(examples)
        }
