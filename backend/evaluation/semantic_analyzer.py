"""
Semantic Analysis for Dataset Evaluation

Provides semantic clustering and similarity analysis using embeddings.
Optional dependency on sentence-transformers.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SemanticCluster:
    """A cluster of semantically similar examples"""
    cluster_id: int
    size: int
    centroid_text: str  # Representative text for the cluster
    example_indices: List[int]
    keywords: List[str]  # Top keywords in cluster


@dataclass
class SemanticAnalysis:
    """Results of semantic analysis"""
    num_clusters: int
    clusters: List[SemanticCluster]
    semantic_diversity: float  # 0-1, higher = more diverse
    avg_cluster_size: float
    largest_cluster_ratio: float  # Size of largest cluster / total
    outlier_count: int  # Examples that don't fit well in any cluster
    topic_distribution: Dict[str, float]  # Cluster labels with percentages


class SemanticAnalyzer:
    """
    Analyzer for semantic clustering and similarity.
    
    Uses sentence-transformers for embeddings if available,
    falls back to TF-IDF based clustering otherwise.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the semantic analyzer.
        
        Args:
            model_name: Name of sentence-transformers model to use
        """
        self.model_name = model_name
        self.model = None
        self.use_transformers = False
        
        # Try to load sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.use_transformers = True
            logger.info(f"Loaded sentence-transformers model: {model_name}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Using TF-IDF fallback for semantic analysis. "
                "Install with: pip install sentence-transformers"
            )
    
    def extract_text(self, example: Dict) -> str:
        """Extract text content from an example"""
        if "messages" in example:
            texts = []
            for msg in example["messages"]:
                content = msg.get("content", "")
                if content:
                    texts.append(content)
            return " ".join(texts)
        elif "conversation" in example:
            texts = []
            for msg in example["conversation"]:
                content = msg.get("content", "")
                if content:
                    texts.append(content)
            return " ".join(texts)
        elif "text" in example:
            return example["text"]
        elif "prompt" in example and "completion" in example:
            return f"{example['prompt']} {example['completion']}"
        elif "instruction" in example and "output" in example:
            return f"{example['instruction']} {example['output']}"
        return ""
    
    def analyze(
        self,
        examples: List[Dict],
        num_clusters: Optional[int] = None,
        min_cluster_size: int = 2
    ) -> SemanticAnalysis:
        """
        Perform semantic clustering analysis.
        
        Args:
            examples: List of examples to analyze
            num_clusters: Number of clusters (auto-detected if None)
            min_cluster_size: Minimum examples per cluster
        
        Returns:
            SemanticAnalysis with clustering results
        """
        if not examples:
            return self._empty_analysis()
        
        # Extract texts
        texts = [self.extract_text(ex) for ex in examples]
        texts = [t for t in texts if t.strip()]
        
        if len(texts) < 3:
            return self._empty_analysis()
        
        # Auto-detect number of clusters
        if num_clusters is None:
            num_clusters = min(max(3, len(texts) // 20), 15)
        
        if self.use_transformers:
            return self._analyze_with_transformers(texts, num_clusters, min_cluster_size)
        else:
            return self._analyze_with_tfidf(texts, num_clusters, min_cluster_size)
    
    def _analyze_with_transformers(
        self,
        texts: List[str],
        num_clusters: int,
        min_cluster_size: int
    ) -> SemanticAnalysis:
        """Analyze using sentence-transformers embeddings"""
        try:
            import numpy as np
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            from collections import Counter
            
            # Generate embeddings
            embeddings = self.model.encode(texts, show_progress_bar=False)
            
            # Cluster
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            
            # Calculate semantic diversity using silhouette score
            if len(set(labels)) > 1:
                diversity = (silhouette_score(embeddings, labels) + 1) / 2  # Normalize to 0-1
            else:
                diversity = 0.0
            
            # Build clusters
            clusters = []
            label_counts = Counter(labels)
            
            for cluster_id in range(num_clusters):
                indices = [i for i, l in enumerate(labels) if l == cluster_id]
                if len(indices) < min_cluster_size:
                    continue
                
                # Find centroid text (closest to cluster center)
                cluster_embeddings = embeddings[indices]
                centroid = kmeans.cluster_centers_[cluster_id]
                distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
                centroid_idx = indices[np.argmin(distances)]
                centroid_text = texts[centroid_idx][:200] + "..." if len(texts[centroid_idx]) > 200 else texts[centroid_idx]
                
                # Extract keywords (simple word frequency)
                cluster_texts = " ".join([texts[i] for i in indices]).lower()
                words = cluster_texts.split()
                word_counts = Counter(words)
                # Filter common words
                stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                           'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                           'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                           'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                           'as', 'into', 'through', 'during', 'before', 'after', 'above',
                           'below', 'between', 'under', 'again', 'further', 'then', 'once',
                           'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either',
                           'neither', 'not', 'only', 'own', 'same', 'than', 'too', 'very',
                           'can', 'just', 'don', 'now', 'i', 'you', 'he', 'she', 'it',
                           'we', 'they', 'what', 'which', 'who', 'whom', 'this', 'that',
                           'these', 'those', 'am', 'if', 'because', 'until', 'while'}
                keywords = [w for w, c in word_counts.most_common(20) 
                           if w not in stopwords and len(w) > 2][:5]
                
                clusters.append(SemanticCluster(
                    cluster_id=cluster_id,
                    size=len(indices),
                    centroid_text=centroid_text,
                    example_indices=indices,
                    keywords=keywords
                ))
            
            # Calculate statistics
            total = len(texts)
            largest_cluster = max(label_counts.values()) if label_counts else 0
            
            # Topic distribution
            topic_dist = {}
            for cluster in clusters:
                label = f"Cluster {cluster.cluster_id}: {', '.join(cluster.keywords[:3])}"
                topic_dist[label] = round(cluster.size / total * 100, 1)
            
            return SemanticAnalysis(
                num_clusters=len(clusters),
                clusters=clusters,
                semantic_diversity=round(diversity, 4),
                avg_cluster_size=round(total / len(clusters), 1) if clusters else 0,
                largest_cluster_ratio=round(largest_cluster / total, 4) if total > 0 else 0,
                outlier_count=sum(1 for c in label_counts.values() if c < min_cluster_size),
                topic_distribution=topic_dist
            )
            
        except Exception as e:
            logger.error(f"Error in transformer analysis: {e}")
            return self._analyze_with_tfidf(texts, num_clusters, min_cluster_size)
    
    def _analyze_with_tfidf(
        self,
        texts: List[str],
        num_clusters: int,
        min_cluster_size: int
    ) -> SemanticAnalysis:
        """Fallback analysis using TF-IDF"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.cluster import KMeans
            from collections import Counter
            import numpy as np
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Cluster
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(tfidf_matrix)
            
            # Build clusters
            clusters = []
            label_counts = Counter(labels)
            feature_names = vectorizer.get_feature_names_out()
            
            for cluster_id in range(num_clusters):
                indices = [i for i, l in enumerate(labels) if l == cluster_id]
                if len(indices) < min_cluster_size:
                    continue
                
                # Get top terms for cluster
                center = kmeans.cluster_centers_[cluster_id]
                top_indices = center.argsort()[-5:][::-1]
                keywords = [feature_names[i] for i in top_indices]
                
                # Get representative text
                centroid_text = texts[indices[0]][:200] + "..." if len(texts[indices[0]]) > 200 else texts[indices[0]]
                
                clusters.append(SemanticCluster(
                    cluster_id=cluster_id,
                    size=len(indices),
                    centroid_text=centroid_text,
                    example_indices=indices,
                    keywords=keywords
                ))
            
            # Calculate statistics
            total = len(texts)
            largest_cluster = max(label_counts.values()) if label_counts else 0
            
            # Estimate diversity (based on cluster distribution)
            if clusters:
                sizes = [c.size for c in clusters]
                # Simpson's diversity index
                diversity = 1 - sum((s/total)**2 for s in sizes)
            else:
                diversity = 0.0
            
            # Topic distribution
            topic_dist = {}
            for cluster in clusters:
                label = f"Cluster {cluster.cluster_id}: {', '.join(cluster.keywords[:3])}"
                topic_dist[label] = round(cluster.size / total * 100, 1)
            
            return SemanticAnalysis(
                num_clusters=len(clusters),
                clusters=clusters,
                semantic_diversity=round(diversity, 4),
                avg_cluster_size=round(total / len(clusters), 1) if clusters else 0,
                largest_cluster_ratio=round(largest_cluster / total, 4) if total > 0 else 0,
                outlier_count=sum(1 for c in label_counts.values() if c < min_cluster_size),
                topic_distribution=topic_dist
            )
            
        except Exception as e:
            logger.error(f"Error in TF-IDF analysis: {e}")
            return self._empty_analysis()
    
    def _empty_analysis(self) -> SemanticAnalysis:
        """Return empty analysis for edge cases"""
        return SemanticAnalysis(
            num_clusters=0,
            clusters=[],
            semantic_diversity=0.0,
            avg_cluster_size=0.0,
            largest_cluster_ratio=0.0,
            outlier_count=0,
            topic_distribution={}
        )
    
    def find_similar(
        self,
        query: str,
        examples: List[Dict],
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Find examples most similar to a query.
        
        Args:
            query: Query text
            examples: List of examples to search
            top_k: Number of results to return
        
        Returns:
            List of (index, similarity_score) tuples
        """
        if not self.use_transformers or not examples:
            return []
        
        try:
            import numpy as np
            
            texts = [self.extract_text(ex) for ex in examples]
            
            # Encode query and texts
            query_embedding = self.model.encode([query])[0]
            text_embeddings = self.model.encode(texts, show_progress_bar=False)
            
            # Calculate cosine similarities
            similarities = np.dot(text_embeddings, query_embedding) / (
                np.linalg.norm(text_embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Get top-k
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            return [(int(idx), float(similarities[idx])) for idx in top_indices]
            
        except Exception as e:
            logger.error(f"Error finding similar examples: {e}")
            return []
