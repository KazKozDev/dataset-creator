"""
Augmentation Techniques

Various text augmentation methods
"""

import random
import re
from typing import List, Dict, Optional
from abc import ABC, abstractmethod


class BaseAugmenter(ABC):
    """Base class for augmentation techniques"""

    def __init__(self, config):
        """Initialize augmenter with config"""
        self.config = config

    @abstractmethod
    def augment_text(self, text: str) -> str:
        """Augment text"""
        pass


class SynonymAugmenter(BaseAugmenter):
    """Synonym replacement augmentation"""

    # Simple synonym dictionary for common words
    SYNONYMS = {
        "good": ["great", "excellent", "fine", "nice", "wonderful"],
        "bad": ["poor", "terrible", "awful", "horrible", "dreadful"],
        "big": ["large", "huge", "enormous", "massive", "giant"],
        "small": ["tiny", "little", "mini", "compact", "minute"],
        "fast": ["quick", "rapid", "swift", "speedy", "hasty"],
        "slow": ["sluggish", "leisurely", "gradual", "unhurried"],
        "happy": ["joyful", "cheerful", "glad", "delighted", "pleased"],
        "sad": ["unhappy", "sorrowful", "gloomy", "melancholy", "dejected"],
        "easy": ["simple", "straightforward", "effortless", "uncomplicated"],
        "hard": ["difficult", "challenging", "tough", "demanding", "complex"],
        "new": ["fresh", "novel", "recent", "modern", "latest"],
        "old": ["ancient", "aged", "elderly", "vintage", "antique"],
        "important": ["significant", "crucial", "vital", "essential", "critical"],
        "help": ["assist", "aid", "support", "facilitate", "contribute"],
        "make": ["create", "produce", "build", "construct", "generate"],
        "get": ["obtain", "acquire", "receive", "gain", "secure"],
        "use": ["utilize", "employ", "apply", "implement", "leverage"],
        "say": ["state", "mention", "express", "declare", "articulate"],
        "think": ["believe", "consider", "suppose", "assume", "reckon"],
        "want": ["desire", "wish", "need", "require", "seek"],
        "show": ["display", "demonstrate", "exhibit", "reveal", "present"],
        "give": ["provide", "offer", "supply", "deliver", "grant"],
        "take": ["grab", "seize", "capture", "acquire", "obtain"],
        "find": ["discover", "locate", "identify", "detect", "uncover"],
        "know": ["understand", "comprehend", "recognize", "realize", "grasp"],
        "work": ["function", "operate", "perform", "execute", "run"],
        "start": ["begin", "commence", "initiate", "launch", "kickoff"],
        "end": ["finish", "conclude", "complete", "terminate", "cease"],
        "look": ["appear", "seem", "view", "observe", "examine"],
        "ask": ["inquire", "question", "request", "query", "demand"],
    }

    def augment_text(self, text: str) -> str:
        """Replace random words with synonyms"""
        words = text.split()
        num_to_replace = max(1, int(len(words) * self.config.synonym_ratio))

        # Get replaceable positions
        replaceable = []
        for i, word in enumerate(words):
            word_lower = word.lower().strip(".,!?;:")
            if word_lower in self.SYNONYMS:
                replaceable.append(i)

        if not replaceable:
            return text

        # Randomly select positions to replace
        positions = random.sample(
            replaceable,
            min(num_to_replace, len(replaceable))
        )

        # Replace words
        augmented_words = words.copy()
        for pos in positions:
            original_word = words[pos]
            word_lower = original_word.lower().strip(".,!?;:")

            # Get synonym
            synonyms = self.SYNONYMS.get(word_lower, [])
            if synonyms:
                synonym = random.choice(synonyms)

                # Preserve case
                if original_word[0].isupper():
                    synonym = synonym.capitalize()

                # Preserve punctuation
                punctuation = ""
                for char in original_word[::-1]:
                    if char in ".,!?;:":
                        punctuation = char + punctuation
                    else:
                        break

                augmented_words[pos] = synonym + punctuation

        return " ".join(augmented_words)


class RandomAugmenter(BaseAugmenter):
    """Random insertion, deletion, and swap augmentation"""

    def augment_text(self, text: str) -> str:
        """Apply random transformations"""
        words = text.split()

        if len(words) < 3:
            return text

        # Choose random operation
        operation = random.choice(["swap", "delete", "insert"])

        if operation == "swap":
            return self._random_swap(words)
        elif operation == "delete":
            return self._random_delete(words)
        else:
            return self._random_insert(words)

    def _random_swap(self, words: List[str]) -> str:
        """Randomly swap two words"""
        if len(words) < 2:
            return " ".join(words)

        num_swaps = max(1, int(len(words) * self.config.random_swap_ratio))

        for _ in range(num_swaps):
            idx1 = random.randint(0, len(words) - 1)
            idx2 = random.randint(0, len(words) - 1)
            words[idx1], words[idx2] = words[idx2], words[idx1]

        return " ".join(words)

    def _random_delete(self, words: List[str]) -> str:
        """Randomly delete words"""
        if len(words) < 2:
            return " ".join(words)

        num_to_keep = int(len(words) * (1 - self.config.random_delete_ratio))
        num_to_keep = max(1, num_to_keep)

        indices = list(range(len(words)))
        random.shuffle(indices)
        keep_indices = sorted(indices[:num_to_keep])

        return " ".join([words[i] for i in keep_indices])

    def _random_insert(self, words: List[str]) -> str:
        """Randomly insert duplicate words"""
        num_inserts = max(1, int(len(words) * self.config.random_insert_ratio))

        for _ in range(num_inserts):
            # Pick random word to duplicate
            word_to_insert = random.choice(words)
            # Insert at random position
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, word_to_insert)

        return " ".join(words)


class ParaphraseAugmenter(BaseAugmenter):
    """
    Paraphrase-based augmentation

    Note: This is a rule-based implementation.
    For production, consider using LLM-based paraphrasing.
    """

    # Paraphrasing patterns
    PATTERNS = [
        # Question transformations
        (r"what is (\w+)", r"can you explain \1"),
        (r"how do i (\w+)", r"what's the way to \1"),
        (r"can you (\w+)", r"could you \1"),
        (r"will you (\w+)", r"would you \1"),

        # Statement transformations
        (r"i think (\w+)", r"in my opinion \1"),
        (r"i believe (\w+)", r"i feel \1"),
        (r"this is (\w+)", r"this seems \1"),

        # Connector transformations
        (r" but ", " however "),
        (r" and ", " as well as "),
        (r" so ", " therefore "),
        (r" because ", " since "),
    ]

    def augment_text(self, text: str) -> str:
        """Apply paraphrasing patterns"""
        augmented = text.lower()

        # Randomly select patterns based on diversity setting
        num_patterns = max(1, int(len(self.PATTERNS) * self.config.paraphrase_diversity))
        selected_patterns = random.sample(self.PATTERNS, min(num_patterns, len(self.PATTERNS)))

        for pattern, replacement in selected_patterns:
            if random.random() < 0.5:  # 50% chance to apply each pattern
                augmented = re.sub(pattern, replacement, augmented, count=1)

        # Restore capitalization
        if text and text[0].isupper():
            augmented = augmented[0].upper() + augmented[1:]

        return augmented if augmented != text.lower() else text


class BackTranslationAugmenter(BaseAugmenter):
    """
    Back-translation augmentation

    Note: This is a mock implementation.
    For production, integrate with translation APIs (Google Translate, DeepL, etc.)
    """

    # Mock translation tables for demonstration
    TRANSLATIONS = {
        "es": {  # Spanish
            "hello": "hola",
            "world": "mundo",
            "good": "bueno",
            "morning": "mañana",
            "the": "el",
            "a": "un",
            "is": "es",
            "are": "son",
        },
        "fr": {  # French
            "hello": "bonjour",
            "world": "monde",
            "good": "bon",
            "morning": "matin",
            "the": "le",
            "a": "un",
            "is": "est",
            "are": "sont",
        }
    }

    def augment_text(self, text: str) -> str:
        """
        Simulate back-translation

        In production, this would:
        1. Translate text to target language
        2. Translate back to source language
        """
        # For now, just apply some transformations to simulate back-translation effects
        words = text.split()

        # Simulate translation artifacts
        transformations = [
            lambda w: w,  # Keep unchanged
            lambda w: w.replace("'", " "),  # Contract expansions
            lambda w: w if not w.endswith("ing") else w[:-3] + "e",  # Verb form changes
        ]

        augmented_words = [
            random.choice(transformations)(word)
            if random.random() < 0.3 else word
            for word in words
        ]

        return " ".join(augmented_words)


class ContextualAugmenter(BaseAugmenter):
    """
    Contextual word substitution

    Note: This would use embeddings/LLMs in production.
    This is a simplified rule-based version.
    """

    CONTEXTUAL_REPLACEMENTS = {
        "very good": ["excellent", "outstanding", "superb"],
        "very bad": ["terrible", "awful", "horrible"],
        "very big": ["enormous", "huge", "massive"],
        "very small": ["tiny", "minuscule", "microscopic"],
        "a lot of": ["many", "numerous", "plenty of"],
        "kind of": ["somewhat", "rather", "sort of"],
    }

    def augment_text(self, text: str) -> str:
        """Replace multi-word phrases with contextual alternatives"""
        augmented = text

        for phrase, replacements in self.CONTEXTUAL_REPLACEMENTS.items():
            if phrase in text.lower():
                replacement = random.choice(replacements)
                # Case-sensitive replacement
                pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                augmented = pattern.sub(replacement, augmented, count=1)
                break

        return augmented
