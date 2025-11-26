from datasketch import MinHash, MinHashLSH
from typing import List, Dict, Any, Set
import re

class Deduplicator:
    def __init__(self, threshold: float = 0.8, num_perm: int = 128):
        self.threshold = threshold
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.minhashes = {}

    def _preprocess(self, text: str) -> Set[str]:
        """Convert text to a set of shingles (3-grams)"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        shingles = set()
        for i in range(len(tokens) - 2):
            shingle = " ".join(tokens[i:i+3])
            shingles.add(shingle)
        return shingles

    def _get_minhash(self, text: str) -> MinHash:
        """Compute MinHash for text"""
        m = MinHash(num_perm=self.num_perm)
        shingles = self._preprocess(text)
        for s in shingles:
            m.update(s.encode('utf8'))
        return m

    def add_document(self, doc_id: str, text: str):
        """Add a document to the index"""
        m = self._get_minhash(text)
        self.lsh.insert(doc_id, m)
        self.minhashes[doc_id] = m

    def find_duplicates(self, doc_id: str, text: str) -> List[str]:
        """Find duplicates for a document"""
        m = self._get_minhash(text)
        result = self.lsh.query(m)
        # Filter out self
        return [r for r in result if r != doc_id]

    def clear(self):
        """Clear the index"""
        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self.minhashes = {}

# Global instance
deduplicator = Deduplicator()
