"""
Item Tower: Article Encoder
Encodes articles into fixed-size embeddings
"""
import numpy as np
from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class ItemEncoder:
    """Encode articles into embeddings"""
    
    def __init__(self, config: Dict):
        self.embedding_dim = config['embedding_dim']
        self.text_weight = config['text_weight']
        self.category_weight = config['category_weight']
        
        self.tfidf_vectorizer = None
        self.category_embeddings = {}
        self.tfidf_matrix = None
        
        np.random.seed(42)
    
    def fit(self, articles: Dict[str, Dict]):
        """
        Fit encoders on article corpus
        
        Args:
            articles: {article_id: {title, category, ...}}
        """
        # Extract titles for TF-IDF
        titles = [articles[aid].get('title', '') for aid in articles.keys()]
        article_ids = list(articles.keys())
        
        # Fit TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.embedding_dim,
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(titles)
        
        # Create category embeddings (random, fixed)
        categories = set()
        for article in articles.values():
            categories.add(article.get('category', 'unknown'))
        
        for category in categories:
            self.category_embeddings[category] = np.random.randn(self.embedding_dim)
        
        # Normalize
        for cat in self.category_embeddings:
            self.category_embeddings[cat] = normalize(
                self.category_embeddings[cat].reshape(1, -1)
            )[0]
    
    def encode_article(self, article: Dict) -> np.ndarray:
        """
        Encode single article
        
        Returns: embedding of shape (embedding_dim,)
        """
        if self.tfidf_vectorizer is None:
            raise ValueError("Encoder not fitted. Call fit() first.")
        
        # Text embedding (TF-IDF)
        title = article.get('title', '')
        text_vec = self.tfidf_vectorizer.transform([title]).toarray()[0]
        text_vec = normalize(text_vec.reshape(1, -1))[0]
        
        # Category embedding
        category = article.get('category', 'unknown')
        cat_vec = self.category_embeddings.get(
            category,
            np.random.randn(self.embedding_dim)
        )
        cat_vec = normalize(cat_vec.reshape(1, -1))[0]
        
        # Combine
        combined = (
            self.text_weight * text_vec +
            self.category_weight * cat_vec
        )
        
        # Normalize final embedding
        embedding = normalize(combined.reshape(1, -1))[0]
        
        return embedding.astype(np.float32)
    
    def encode_batch(self, articles: Dict[str, Dict]) -> Dict[str, np.ndarray]:
        """
        Encode multiple articles
        
        Returns: {article_id: embedding}
        """
        embeddings = {}
        
        for article_id, article in articles.items():
            embeddings[article_id] = self.encode_article(article)
        
        return embeddings
