import numpy as np
import math
from typing import List, Dict, Any, Tuple
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from sentence_transformers import SentenceTransformer

from config.settings import settings
from models.schemas import ChunkFeatures, QueryFeatures
from utils.logger import get_logger

logger = get_logger(__name__)

class FeatureExtractor:
    def __init__(self):
        self.embedding_model = SentenceTransformer(settings.EMBEDDINGS_MODEL)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.fitted_tfidf = False

    def extract_chunk_features(self, chunk: str, chunk_id: str, all_chunks: List[str]) -> ChunkFeatures:
        """Extract comprehensive features for a text chunk"""
        try:
            # Basic statistics
            tokens = chunk.split()
            length = len(tokens)
            unique_words = len(set(tokens))
            lexical_diversity = unique_words / length if length > 0 else 0

            # Information content (entropy)
            word_counts = Counter(tokens)
            total_words = sum(word_counts.values())
            entropy = 0
            for count in word_counts.values():
                prob = count / total_words
                entropy -= prob * math.log2(prob) if prob > 0 else 0

            # Semantic similarity with other chunks
            chunk_embedding = self.embedding_model.encode([chunk])
            if len(all_chunks) > 1:
                all_embeddings = self.embedding_model.encode(all_chunks)
                similarities = cosine_similarity(chunk_embedding, all_embeddings)[0]
                # Remove self-similarity
                other_similarities = [sim for i, sim in enumerate(similarities) if all_chunks[i] != chunk]
                similarity_avg = np.mean(other_similarities) if other_similarities else 0
            else:
                similarity_avg = 0

            # Dependency score calculation
            dependency_score = self._calculate_dependency_score(
                chunk, similarity_avg, chunk_id, all_chunks
            )

            # Content indicators
            numeric_content = bool(re.search(r'\d+', chunk))
            has_examples = self._has_examples(chunk)
            metadata_completeness = self._calculate_metadata_completeness(chunk)

            # Overall complexity score
            complexity_score = self._calculate_complexity_score(
                length, lexical_diversity, entropy, dependency_score,
                similarity_avg, metadata_completeness
            )

            features = ChunkFeatures(
                chunk_id=chunk_id,
                length=self._normalize_feature(length, 0, 1000),
                unique_words=unique_words,
                lexical_diversity=lexical_diversity,
                entropy=entropy,
                similarity_avg=similarity_avg,
                dependency_score=dependency_score,
                numeric_content=numeric_content,
                has_examples=has_examples,
                metadata_completeness=metadata_completeness,
                complexity_score=complexity_score
            )

            logger.info(f"Extracted features for chunk {chunk_id}")
            return features

        except Exception as e:
            logger.error(f"Error extracting chunk features: {e}")
            raise

    def extract_query_features(self, query: str) -> QueryFeatures:
        """Extract query-level features"""
        try:
            # Intent classification (simplified)
            intent_vector = self._classify_intent(query)

            # Ambiguity score (entropy over possible interpretations)
            ambiguity_score = self._calculate_query_ambiguity(query)

            # Entity count
            entity_count = self._count_entities(query)

            # Query length
            query_length = len(query.split())

            features = QueryFeatures(
                query=query,
                intent_vector=intent_vector,
                ambiguity_score=ambiguity_score,
                entity_count=entity_count,
                query_length=query_length
            )

            logger.info("Extracted query features")
            return features

        except Exception as e:
            logger.error(f"Error extracting query features: {e}")
            raise

    def calculate_query_chunk_similarity(self, query: str, chunk: str) -> Dict[str, float]:
        """Calculate various similarity metrics between query and chunk"""
        try:
            # Semantic similarity
            query_emb = self.embedding_model.encode([query])
            chunk_emb = self.embedding_model.encode([chunk])
            semantic_sim = cosine_similarity(query_emb, chunk_emb)[0][0]

            # BM25 similarity (simplified)
            bm25_sim = self._calculate_bm25_similarity(query, chunk)

            # Entity overlap
            query_entities = self._extract_entities(query)
            chunk_entities = self._extract_entities(chunk)
            entity_overlap = self._calculate_entity_overlap(query_entities, chunk_entities)

            return {
                "semantic_similarity": float(semantic_sim),
                "bm25_similarity": bm25_sim,
                "entity_overlap": entity_overlap
            }

        except Exception as e:
            logger.error(f"Error calculating query-chunk similarity: {e}")
            return {"semantic_similarity": 0.0, "bm25_similarity": 0.0, "entity_overlap": 0.0}

    def _calculate_dependency_score(self, chunk: str, similarity_avg: float,
                                  chunk_id: str, all_chunks: List[str]) -> float:
        """Calculate dependency score based on mathematical formula"""
        alpha, beta, gamma = settings.ALPHA, settings.BETA, settings.GAMMA

        # Co-retrieval frequency (simplified - would need historical data)
        co_retrieval = 0.5  # Placeholder

        # Graph degree (simplified - would need document graph)
        graph_degree = 0.3  # Placeholder

        dependency = alpha * (1 - similarity_avg) + beta * co_retrieval + gamma * graph_degree
        return max(0, min(1, dependency))  # Normalize to [0,1]

    def _calculate_complexity_score(self, length: float, diversity: float, entropy: float,
                                  dependency: float, similarity: float, metadata: float) -> float:
        """Calculate overall complexity score using sigmoid function"""
        # Weights for complexity calculation
        w_l, w_d, w_e, w_dep, w_sim, w_meta = 0.15, 0.15, 0.2, 0.2, 0.15, 0.15
        bias = -2.0

        # Normalize inputs
        norm_length = self._normalize_feature(length, 0, 1000)

        # Linear combination
        linear_sum = (w_l * norm_length + w_d * diversity + w_e * entropy/10 +
                     w_dep * dependency + w_sim * (1 - similarity) +
                     w_meta * (1 - metadata) + bias)

        # Sigmoid activation
        complexity = 1 / (1 + math.exp(-linear_sum))
        return complexity

    def _classify_intent(self, query: str) -> List[float]:
        """Classify query intent into [lookup, reasoning, synthesis, numeric]"""
        query_lower = query.lower()

        # Simple keyword-based classification
        lookup_keywords = ['what', 'who', 'when', 'where', 'define', 'explain']
        reasoning_keywords = ['why', 'how', 'analyze', 'compare', 'evaluate']
        synthesis_keywords = ['create', 'generate', 'summarize', 'combine']
        numeric_keywords = ['calculate', 'count', 'measure', 'percentage']

        intent = [0.0, 0.0, 0.0, 0.0]

        for keyword in lookup_keywords:
            if keyword in query_lower:
                intent[0] += 0.2

        for keyword in reasoning_keywords:
            if keyword in query_lower:
                intent[1] += 0.2

        for keyword in synthesis_keywords:
            if keyword in query_lower:
                intent[2] += 0.2

        for keyword in numeric_keywords:
            if keyword in query_lower:
                intent[3] += 0.2

        # Normalize
        total = sum(intent)
        if total > 0:
            intent = [x/total for x in intent]
        else:
            intent = [0.25, 0.25, 0.25, 0.25]  # Default uniform distribution

        return intent

    def _calculate_query_ambiguity(self, query: str) -> float:
        """Calculate query ambiguity using entropy of possible interpretations"""
        # Simplified: based on number of possible meanings of words
        words = query.split()
        polysemy_scores = []

        for word in words:
            # Simplified polysemy score (in practice, use WordNet or similar)
            if len(word) <= 3:
                polysemy_scores.append(0.2)
            elif len(word) <= 6:
                polysemy_scores.append(0.4)
            else:
                polysemy_scores.append(0.6)

        if not polysemy_scores:
            return 0.0

        avg_polysemy = np.mean(polysemy_scores)
        return min(1.0, avg_polysemy)

    def _count_entities(self, text: str) -> int:
        """Count named entities in text (simplified)"""
        # Simplified entity detection using capitalized words
        words = text.split()
        entities = [word for word in words if word[0].isupper() and len(word) > 2]
        return len(entities)

    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from text"""
        words = text.split()
        entities = [word for word in words if word[0].isupper() and len(word) > 2]
        return entities

    def _calculate_entity_overlap(self, entities1: List[str], entities2: List[str]) -> float:
        """Calculate Jaccard similarity between entity sets"""
        set1 = set(entities1)
        set2 = set(entities2)

        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def _calculate_bm25_similarity(self, query: str, document: str) -> float:
        """Simplified BM25 calculation"""
        k1, b = 1.5, 0.75
        query_terms = query.lower().split()
        doc_terms = document.lower().split()
        doc_length = len(doc_terms)
        avgdl = 100  # Average document length (placeholder)

        score = 0.0
        doc_term_counts = Counter(doc_terms)

        for term in query_terms:
            if term in doc_term_counts:
                tf = doc_term_counts[term]
                idf = math.log((1 + 0.5) / (0.5))  # Simplified IDF

                score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avgdl)))

        return min(1.0, score / 10)  # Normalize

    def _has_examples(self, text: str) -> bool:
        """Check if text contains examples"""
        example_indicators = ['example', 'for instance', 'such as', 'like', 'e.g.', 'contoh']
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in example_indicators)

    def _calculate_metadata_completeness(self, text: str) -> float:
        """Calculate metadata completeness score"""
        # Check for various metadata indicators
        indicators = {
            'title': bool(re.search(r'^#|^\*\*|^[A-Z][^.]*$', text, re.MULTILINE)),
            'structure': bool(re.search(r'^\d+\.|^-|^\*', text, re.MULTILINE)),
            'references': bool(re.search(r'\[\d+\]|Reference|Sumber', text)),
            'dates': bool(re.search(r'\d{4}|\d{1,2}/\d{1,2}/\d{4}', text)),
        }

        completeness = sum(indicators.values()) / len(indicators)
        return completeness

    def _normalize_feature(self, value: float, min_val: float, max_val: float) -> float:
        """Min-max normalize a feature value"""
        if max_val == min_val:
            return 0.5
        return (value - min_val) / (max_val - min_val)
