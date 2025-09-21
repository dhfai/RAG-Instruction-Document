from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from models.schemas import DocumentChunk, QueryFeatures
from core.feature_extraction import FeatureExtractor
from utils.logger import get_logger

logger = get_logger(__name__)

class BaseRetrievalStrategy(ABC):
    def __init__(self):
        self.feature_extractor = FeatureExtractor()

    @abstractmethod
    def retrieve(self, query: str, chunks: List[DocumentChunk],
                top_k: int = 10) -> List[DocumentChunk]:
        pass

class S1_SinglePassDense(BaseRetrievalStrategy):
    """S1 - Single-pass Dense Retrieval Strategy"""

    def retrieve(self, query: str, chunks: List[DocumentChunk],
                top_k: int = 10) -> List[DocumentChunk]:
        """Simple dense retrieval using semantic similarity"""
        try:
            logger.info("Executing S1 - Single-pass Dense retrieval")

            # Get query embedding
            query_embedding = self.feature_extractor.embedding_model.encode([query])

            # Calculate similarities
            scored_chunks = []
            for chunk in chunks:
                if chunk.embedding is None:
                    chunk.embedding = self.feature_extractor.embedding_model.encode([chunk.content])

                similarity = cosine_similarity(query_embedding, [chunk.embedding])[0][0]
                scored_chunks.append((chunk, float(similarity)))

            # Sort by similarity and return top_k
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            selected_chunks = [chunk for chunk, _ in scored_chunks[:top_k]]

            logger.info(f"S1 retrieved {len(selected_chunks)} chunks")
            return selected_chunks

        except Exception as e:
            logger.error(f"Error in S1 retrieval: {e}")
            return chunks[:top_k]  # Fallback

class S2_HybridBM25Dense(BaseRetrievalStrategy):
    """S2 - Hybrid BM25+Dense Retrieval Strategy"""

    def retrieve(self, query: str, chunks: List[DocumentChunk],
                top_k: int = 10) -> List[DocumentChunk]:
        """Hybrid retrieval combining BM25 and dense similarity"""
        try:
            logger.info("Executing S2 - Hybrid BM25+Dense retrieval")

            # Get query embedding
            query_embedding = self.feature_extractor.embedding_model.encode([query])

            scored_chunks = []
            for chunk in chunks:
                # Dense similarity
                if chunk.embedding is None:
                    chunk.embedding = self.feature_extractor.embedding_model.encode([chunk.content])

                dense_sim = cosine_similarity(query_embedding, [chunk.embedding])[0][0]

                # BM25 similarity (simplified)
                bm25_sim = self._calculate_bm25_score(query, chunk.content)

                # Entity overlap
                query_entities = self._extract_entities(query)
                chunk_entities = self._extract_entities(chunk.content)
                entity_overlap = self._calculate_entity_overlap(query_entities, chunk_entities)

                # Combined score with weights
                combined_score = (0.4 * dense_sim + 0.4 * bm25_sim + 0.2 * entity_overlap)
                scored_chunks.append((chunk, combined_score))

            # Sort and return top_k
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            selected_chunks = [chunk for chunk, _ in scored_chunks[:top_k]]

            logger.info(f"S2 retrieved {len(selected_chunks)} chunks")
            return selected_chunks

        except Exception as e:
            logger.error(f"Error in S2 retrieval: {e}")
            return chunks[:top_k]

    def _calculate_bm25_score(self, query: str, document: str) -> float:
        """Simplified BM25 calculation"""
        k1, b = 1.5, 0.75
        query_terms = query.lower().split()
        doc_terms = document.lower().split()
        doc_length = len(doc_terms)
        avgdl = 100  # Average document length

        score = 0.0
        from collections import Counter
        doc_term_counts = Counter(doc_terms)

        for term in query_terms:
            if term in doc_term_counts:
                tf = doc_term_counts[term]
                idf = np.log((len(doc_terms) + 0.5) / (tf + 0.5))

                score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avgdl)))

        return min(1.0, score / 10)  # Normalize

    def _extract_entities(self, text: str) -> List[str]:
        """Simple entity extraction"""
        words = text.split()
        entities = [word for word in words if word[0].isupper() and len(word) > 2]
        return entities

    def _calculate_entity_overlap(self, entities1: List[str], entities2: List[str]) -> float:
        """Calculate Jaccard similarity between entity sets"""
        set1, set2 = set(entities1), set(entities2)
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / len(set1 | set2)

class S3_MultiHopIterative(BaseRetrievalStrategy):
    """S3 - Multi-hop Iterative Retrieval Strategy"""

    def retrieve(self, query: str, chunks: List[DocumentChunk],
                top_k: int = 10) -> List[DocumentChunk]:
        """Multi-hop retrieval for complex queries requiring multiple steps"""
        try:
            logger.info("Executing S3 - Multi-hop Iterative retrieval")

            selected_chunks = []
            current_query = query
            remaining_chunks = chunks.copy()
            max_hops = 3

            for hop in range(max_hops):
                if len(selected_chunks) >= top_k or not remaining_chunks:
                    break

                logger.info(f"Hop {hop + 1}: Query = {current_query[:100]}...")

                # Find best chunk for current hop
                hop_chunks = self._retrieve_hop(current_query, remaining_chunks,
                                              min(3, top_k - len(selected_chunks)))

                if not hop_chunks:
                    break

                selected_chunks.extend(hop_chunks)

                # Remove selected chunks from remaining
                selected_ids = {chunk.chunk_id for chunk in hop_chunks}
                remaining_chunks = [c for c in remaining_chunks if c.chunk_id not in selected_ids]

                # Update query for next hop based on selected content
                current_query = self._generate_followup_query(current_query, hop_chunks)

            logger.info(f"S3 retrieved {len(selected_chunks)} chunks over {hop + 1} hops")
            return selected_chunks[:top_k]

        except Exception as e:
            logger.error(f"Error in S3 retrieval: {e}")
            return chunks[:top_k]

    def _retrieve_hop(self, query: str, chunks: List[DocumentChunk], k: int) -> List[DocumentChunk]:
        """Retrieve chunks for a single hop"""
        query_embedding = self.feature_extractor.embedding_model.encode([query])

        scored_chunks = []
        for chunk in chunks:
            if chunk.embedding is None:
                chunk.embedding = self.feature_extractor.embedding_model.encode([chunk.content])

            # Consider both similarity and dependency
            similarity = cosine_similarity(query_embedding, [chunk.embedding])[0][0]
            dependency_bonus = chunk.features.dependency_score if chunk.features else 0.5

            score = 0.7 * similarity + 0.3 * dependency_bonus
            scored_chunks.append((chunk, score))

        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in scored_chunks[:k]]

    def _generate_followup_query(self, original_query: str, selected_chunks: List[DocumentChunk]) -> str:
        """Generate follow-up query based on selected content"""
        # Extract key terms from selected chunks
        all_content = " ".join([chunk.content for chunk in selected_chunks])
        words = all_content.split()

        # Simple keyword extraction (in practice, use more sophisticated methods)
        important_words = [word for word in words if len(word) > 5][:5]

        # Combine with original query
        followup = f"{original_query} {' '.join(important_words)}"
        return followup

class S4_ClusteredSelection(BaseRetrievalStrategy):
    """S4 - Clustered Selection Strategy"""

    def retrieve(self, query: str, chunks: List[DocumentChunk],
                top_k: int = 10) -> List[DocumentChunk]:
        """Retrieve diverse chunks using clustering"""
        try:
            logger.info("Executing S4 - Clustered Selection retrieval")

            if len(chunks) < top_k:
                return chunks

            # Get embeddings for all chunks
            embeddings = []
            for chunk in chunks:
                if chunk.embedding is None:
                    chunk.embedding = self.feature_extractor.embedding_model.encode([chunk.content])
                embeddings.append(chunk.embedding)

            embeddings = np.array(embeddings)

            # Cluster chunks
            n_clusters = min(top_k // 2, len(chunks) // 3, 5)  # Adaptive cluster count
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)

            # Select diverse chunks from each cluster
            query_embedding = self.feature_extractor.embedding_model.encode([query])
            selected_chunks = []

            for cluster_id in range(n_clusters):
                cluster_chunks = [chunks[i] for i in range(len(chunks))
                                if cluster_labels[i] == cluster_id]

                if not cluster_chunks:
                    continue

                # Select best chunk from this cluster
                best_chunk = self._select_best_from_cluster(query_embedding, cluster_chunks)
                selected_chunks.append(best_chunk)

                # If we need more chunks and this cluster has them, add more
                if len(selected_chunks) < top_k and len(cluster_chunks) > 1:
                    remaining = [c for c in cluster_chunks if c.chunk_id != best_chunk.chunk_id]
                    if remaining:
                        second_best = self._select_best_from_cluster(query_embedding, remaining)
                        selected_chunks.append(second_best)

            logger.info(f"S4 retrieved {len(selected_chunks)} chunks from {n_clusters} clusters")
            return selected_chunks[:top_k]

        except Exception as e:
            logger.error(f"Error in S4 retrieval: {e}")
            return chunks[:top_k]

    def _select_best_from_cluster(self, query_embedding: np.ndarray,
                                cluster_chunks: List[DocumentChunk]) -> DocumentChunk:
        """Select best chunk from a cluster"""
        best_chunk = cluster_chunks[0]
        best_score = -1

        for chunk in cluster_chunks:
            similarity = cosine_similarity(query_embedding, [chunk.embedding])[0][0]

            # Add diversity bonus (favor chunks with unique characteristics)
            diversity_bonus = chunk.features.lexical_diversity if chunk.features else 0.5
            score = 0.8 * similarity + 0.2 * diversity_bonus

            if score > best_score:
                best_score = score
                best_chunk = chunk

        return best_chunk

class S5_QueryRewriteDecomposition(BaseRetrievalStrategy):
    """S5 - Query Rewrite/Decomposition Strategy"""

    def retrieve(self, query: str, chunks: List[DocumentChunk],
                top_k: int = 10) -> List[DocumentChunk]:
        """Retrieve using query decomposition and rewriting"""
        try:
            logger.info("Executing S5 - Query Rewrite/Decomposition retrieval")

            # Decompose query into sub-queries
            sub_queries = self._decompose_query(query)

            all_selected = []
            query_weights = [1.0] + [0.7] * (len(sub_queries) - 1)  # Main query gets higher weight

            for i, (sub_query, weight) in enumerate(zip(sub_queries, query_weights)):
                logger.info(f"Sub-query {i + 1}: {sub_query}")

                # Retrieve for each sub-query
                sub_results = self._retrieve_for_subquery(sub_query, chunks, top_k // 2)

                # Weight the results
                for chunk in sub_results:
                    all_selected.append((chunk, weight))

            # Aggregate and deduplicate results
            chunk_scores = {}
            for chunk, weight in all_selected:
                if chunk.chunk_id in chunk_scores:
                    chunk_scores[chunk.chunk_id] = (chunk, chunk_scores[chunk.chunk_id][1] + weight)
                else:
                    chunk_scores[chunk.chunk_id] = (chunk, weight)

            # Sort by aggregated score
            sorted_results = sorted(chunk_scores.values(), key=lambda x: x[1], reverse=True)
            final_chunks = [chunk for chunk, _ in sorted_results[:top_k]]

            logger.info(f"S5 retrieved {len(final_chunks)} chunks from {len(sub_queries)} sub-queries")
            return final_chunks

        except Exception as e:
            logger.error(f"Error in S5 retrieval: {e}")
            return chunks[:top_k]

    def _decompose_query(self, query: str) -> List[str]:
        """Decompose complex query into sub-queries"""
        sub_queries = [query]  # Always include original

        # Simple decomposition based on conjunctions and question words
        if " and " in query.lower():
            parts = query.lower().split(" and ")
            sub_queries.extend(parts)

        if " or " in query.lower():
            parts = query.lower().split(" or ")
            sub_queries.extend(parts)

        # Extract key concepts
        words = query.split()
        if len(words) > 8:  # For long queries
            # Create focused sub-queries from key terms
            key_terms = [word for word in words if len(word) > 4][:3]
            for term in key_terms:
                sub_queries.append(f"What is {term}?")

        return list(set(sub_queries))  # Remove duplicates

    def _retrieve_for_subquery(self, query: str, chunks: List[DocumentChunk],
                              k: int) -> List[DocumentChunk]:
        """Retrieve chunks for a single sub-query"""
        query_embedding = self.feature_extractor.embedding_model.encode([query])

        scored_chunks = []
        for chunk in chunks:
            if chunk.embedding is None:
                chunk.embedding = self.feature_extractor.embedding_model.encode([chunk.content])

            similarity = cosine_similarity(query_embedding, [chunk.embedding])[0][0]
            scored_chunks.append((chunk, similarity))

        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in scored_chunks[:k]]

# Strategy Factory
class RetrievalStrategyFactory:
    @staticmethod
    def create_strategy(strategy_name: str) -> BaseRetrievalStrategy:
        """Create retrieval strategy instance"""
        strategies = {
            "S1": S1_SinglePassDense,
            "S2": S2_HybridBM25Dense,
            "S3": S3_MultiHopIterative,
            "S4": S4_ClusteredSelection,
            "S5": S5_QueryRewriteDecomposition
        }

        if strategy_name not in strategies:
            logger.warning(f"Unknown strategy {strategy_name}, defaulting to S1")
            strategy_name = "S1"

        return strategies[strategy_name]()
