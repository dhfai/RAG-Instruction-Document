import numpy as np
import math
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from config.settings import settings, STRATEGY_COSTS
from models.schemas import ChunkFeatures, QueryFeatures, StrategySelection
from core.feature_extraction import FeatureExtractor
from utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class StrategyUtility:
    strategy_id: str
    utility_score: float
    expected_cost: float
    expected_latency: float
    confidence: float

class RouterEngine:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()

    def select_retrieval_strategy(self, query_features: QueryFeatures,
                                chunk_features: List[ChunkFeatures],
                                query_chunk_similarities: List[Dict[str, float]]) -> str:
        """Select optimal retrieval strategy based on mathematical utility functions"""
        try:
            utilities = {}

            # Calculate average chunk complexity
            avg_complexity = np.mean([cf.complexity_score for cf in chunk_features])
            avg_similarity = np.mean([sim["semantic_similarity"] for sim in query_chunk_similarities])
            avg_dependency = np.mean([cf.dependency_score for cf in chunk_features])

            # S1 - Single-pass Dense
            utilities["S1"] = self._calculate_s1_utility(
                avg_similarity, avg_complexity, chunk_features
            )

            # S2 - Hybrid BM25+Dense
            utilities["S2"] = self._calculate_s2_utility(
                query_chunk_similarities, chunk_features
            )

            # S3 - Multi-hop Iterative
            utilities["S3"] = self._calculate_s3_utility(
                avg_dependency, query_features.ambiguity_score, chunk_features
            )

            # S4 - Clustered Selection
            utilities["S4"] = self._calculate_s4_utility(chunk_features)

            # S5 - Query Rewrite/Decomposition
            utilities["S5"] = self._calculate_s5_utility(
                query_features, chunk_features, query_chunk_similarities
            )

            # Apply cost-aware selection
            best_strategy = self._apply_cost_aware_selection(utilities, "retrieval")

            logger.info(f"Selected retrieval strategy: {best_strategy}")
            return best_strategy

        except Exception as e:
            logger.error(f"Error selecting retrieval strategy: {e}")
            return "S1"  # Fallback to simplest strategy

    def select_generation_strategy(self, query_features: QueryFeatures,
                                 chunk_features: List[ChunkFeatures],
                                 selected_chunks: List[str]) -> str:
        """Select optimal generation strategy"""
        try:
            utilities = {}

            num_docs = len(selected_chunks)
            doc_diversity = self._calculate_document_diversity(chunk_features)
            avg_dependency = np.mean([cf.dependency_score for cf in chunk_features])
            avg_numeric = np.mean([1.0 if cf.numeric_content else 0.0 for cf in chunk_features])

            # G1 - Fusion-in-Decoder
            utilities["G1"] = self._calculate_g1_utility(
                num_docs, doc_diversity, query_features.intent_vector[2]  # synthesis intent
            )

            # G2 - Rerank-then-Generate
            utilities["G2"] = self._calculate_g2_utility(chunk_features, selected_chunks)

            # G3 - Chain-of-Thought
            utilities["G3"] = self._calculate_g3_utility(
                query_features.intent_vector[1], avg_dependency  # reasoning intent
            )

            # G4 - Evidence-Aware
            utilities["G4"] = self._calculate_g4_utility(query_features)

            # G5 - Validator-Augmented
            utilities["G5"] = self._calculate_g5_utility(avg_numeric, query_features)

            best_strategy = self._apply_cost_aware_selection(utilities, "generation")

            logger.info(f"Selected generation strategy: {best_strategy}")
            return best_strategy

        except Exception as e:
            logger.error(f"Error selecting generation strategy: {e}")
            return "G2"  # Fallback to rerank-then-generate

    def create_strategy_selection(self, retrieval_strategy: str, generation_strategy: str,
                                query_features: QueryFeatures,
                                chunk_features: List[ChunkFeatures]) -> StrategySelection:
        """Create strategy selection with reasoning"""

        # Calculate combined expected utility
        retrieval_util = self._get_strategy_utility(retrieval_strategy, query_features, chunk_features)
        generation_util = self._get_strategy_utility(generation_strategy, query_features, chunk_features)
        combined_utility = (retrieval_util + generation_util) / 2

        # Calculate confidence based on feature clarity
        confidence = self._calculate_selection_confidence(query_features, chunk_features)

        # Generate reasoning
        reasoning = self._generate_selection_reasoning(
            retrieval_strategy, generation_strategy, query_features, chunk_features
        )

        return StrategySelection(
            retrieval_strategy=retrieval_strategy,
            generation_strategy=generation_strategy,
            expected_utility=combined_utility,
            confidence=confidence,
            reasoning=reasoning
        )

    def _calculate_s1_utility(self, avg_similarity: float, avg_complexity: float,
                            chunk_features: List[ChunkFeatures]) -> float:
        """S1 - Single-pass Dense utility calculation"""
        weights = settings.S1_WEIGHTS
        avg_length = np.mean([cf.length for cf in chunk_features])
        avg_metadata = np.mean([cf.metadata_completeness for cf in chunk_features])

        utility = (weights[0] * avg_similarity +
                  weights[1] * (1 - avg_length) +
                  weights[2] * avg_metadata)

        return max(0, min(1, utility))

    def _calculate_s2_utility(self, query_chunk_similarities: List[Dict[str, float]],
                            chunk_features: List[ChunkFeatures]) -> float:
        """S2 - Hybrid BM25+Dense utility calculation"""
        weights = settings.S2_WEIGHTS

        avg_semantic = np.mean([sim["semantic_similarity"] for sim in query_chunk_similarities])
        avg_bm25 = np.mean([sim["bm25_similarity"] for sim in query_chunk_similarities])
        avg_entity = np.mean([sim["entity_overlap"] for sim in query_chunk_similarities])

        utility = (weights[0] * avg_semantic +
                  weights[1] * avg_bm25 +
                  weights[2] * avg_entity)

        return max(0, min(1, utility))

    def _calculate_s3_utility(self, avg_dependency: float, ambiguity: float,
                            chunk_features: List[ChunkFeatures]) -> float:
        """S3 - Multi-hop Iterative utility calculation"""
        weights = settings.S3_WEIGHTS

        avg_sim_avg = np.mean([cf.similarity_avg for cf in chunk_features])

        utility = (weights[0] * avg_dependency +
                  weights[1] * (1 - avg_sim_avg) +
                  weights[2] * ambiguity)

        return max(0, min(1, utility))

    def _calculate_s4_utility(self, chunk_features: List[ChunkFeatures]) -> float:
        """S4 - Clustered Selection utility calculation"""
        weights = settings.S4_WEIGHTS

        # Cluster diversity (simplified)
        cluster_diversity = self._calculate_cluster_diversity(chunk_features)

        # Marginal gain (simplified)
        marginal_gain = self._calculate_marginal_gain(chunk_features)

        utility = weights[0] * cluster_diversity + weights[1] * marginal_gain

        return max(0, min(1, utility))

    def _calculate_s5_utility(self, query_features: QueryFeatures,
                            chunk_features: List[ChunkFeatures],
                            query_chunk_similarities: List[Dict[str, float]]) -> float:
        """S5 - Query Rewrite/Decomposition utility calculation"""
        weights = settings.S5_WEIGHTS

        # Coverage estimation (simplified)
        coverage = 1 - np.mean([sim["semantic_similarity"] for sim in query_chunk_similarities])

        utility = (weights[0] * query_features.ambiguity_score +
                  weights[1] * query_features.intent_vector[1] +  # reasoning intent
                  weights[2] * coverage)

        return max(0, min(1, utility))

    def _calculate_g1_utility(self, num_docs: int, doc_diversity: float, synthesis_intent: float) -> float:
        """G1 - Fusion-in-Decoder utility calculation"""
        weights = settings.G1_WEIGHTS

        normalized_num_docs = min(1.0, num_docs / 10.0)  # Normalize to [0,1]

        utility = (weights[0] * normalized_num_docs +
                  weights[1] * doc_diversity +
                  weights[2] * synthesis_intent)

        return max(0, min(1, utility))

    def _calculate_g2_utility(self, chunk_features: List[ChunkFeatures], selected_chunks: List[str]) -> float:
        """G2 - Rerank-then-Generate utility calculation"""
        weights = settings.G2_WEIGHTS

        # Simplified rerank gain
        rerank_gain = 0.7  # Placeholder

        # Top similarity
        similarities = [cf.similarity_avg for cf in chunk_features]
        sim_top1 = max(similarities) if similarities else 0

        # Redundancy calculation
        redundancy = self._calculate_redundancy(chunk_features)

        utility = (weights[0] * rerank_gain +
                  weights[1] * sim_top1 +
                  weights[2] * (1 - redundancy))

        return max(0, min(1, utility))

    def _calculate_g3_utility(self, reasoning_intent: float, avg_dependency: float) -> float:
        """G3 - Chain-of-Thought utility calculation"""
        weights = settings.G3_WEIGHTS

        # Estimated hop count (simplified)
        hop_count_est = avg_dependency * 3  # Scale dependency to hop count
        normalized_hop = min(1.0, hop_count_est / 5.0)

        utility = (weights[0] * reasoning_intent +
                  weights[1] * avg_dependency +
                  weights[2] * normalized_hop)

        return max(0, min(1, utility))

    def _calculate_g4_utility(self, query_features: QueryFeatures) -> float:
        """G4 - Evidence-Aware utility calculation"""
        weights = settings.G4_WEIGHTS

        # Citation need (simplified heuristic)
        citation_need = 0.8 if query_features.query_length > 10 else 0.4

        # Risk profile (simplified)
        risk_profile = 0.6  # Medium risk for educational content

        utility = weights[0] * citation_need + weights[1] * risk_profile

        return max(0, min(1, utility))

    def _calculate_g5_utility(self, avg_numeric: float, query_features: QueryFeatures) -> float:
        """G5 - Validator-Augmented utility calculation"""
        weights = settings.G5_WEIGHTS

        risk_profile = 0.6  # Medium risk

        # Claim density (simplified)
        claim_density = min(1.0, query_features.query_length / 20.0)

        utility = (weights[0] * avg_numeric +
                  weights[1] * risk_profile +
                  weights[2] * claim_density)

        return max(0, min(1, utility))

    def _apply_cost_aware_selection(self, utilities: Dict[str, float], strategy_type: str) -> str:
        """Apply cost-aware optimization to select best strategy"""
        try:
            expected_utilities = {}

            for strategy, utility in utilities.items():
                cost_info = STRATEGY_COSTS.get(strategy, {"latency": 1.0, "tokens": 100})

                # Calculate expected utility with cost penalties
                expected_utility = (utility -
                                  settings.LAMBDA_1 * cost_info["latency"] -
                                  settings.LAMBDA_2 * cost_info["tokens"] / 1000)

                expected_utilities[strategy] = expected_utility

            # Select strategy with highest expected utility
            best_strategy = max(expected_utilities.items(), key=lambda x: x[1])[0]

            # Confidence-based fallback
            max_confidence = max(utilities.values())
            if max_confidence < settings.CONFIDENCE_THRESHOLD:
                fallback = "S1" if strategy_type == "retrieval" else "G2"
                logger.warning(f"Low confidence ({max_confidence:.2f}), using fallback: {fallback}")
                return fallback

            return best_strategy

        except Exception as e:
            logger.error(f"Error in cost-aware selection: {e}")
            return "S1" if strategy_type == "retrieval" else "G2"

    def _calculate_document_diversity(self, chunk_features: List[ChunkFeatures]) -> float:
        """Calculate diversity among document chunks"""
        if len(chunk_features) < 2:
            return 0.0

        similarities = []
        for i in range(len(chunk_features)):
            for j in range(i + 1, len(chunk_features)):
                # Use different features to calculate dissimilarity
                dissim = abs(chunk_features[i].entropy - chunk_features[j].entropy)
                similarities.append(dissim)

        return np.mean(similarities) if similarities else 0.0

    def _calculate_cluster_diversity(self, chunk_features: List[ChunkFeatures]) -> float:
        """Calculate cluster diversity (simplified)"""
        if len(chunk_features) < 2:
            return 0.0

        # Use entropy as proxy for cluster diversity
        entropies = [cf.entropy for cf in chunk_features]
        return np.std(entropies) / (np.mean(entropies) + 1e-6)

    def _calculate_marginal_gain(self, chunk_features: List[ChunkFeatures]) -> float:
        """Calculate marginal gain for chunk selection"""
        if len(chunk_features) < 2:
            return 1.0

        # Simplified: average distance between chunks
        distances = []
        for i in range(len(chunk_features)):
            for j in range(i + 1, len(chunk_features)):
                # Use feature differences as distance proxy
                dist = abs(chunk_features[i].complexity_score - chunk_features[j].complexity_score)
                distances.append(dist)

        return np.mean(distances) if distances else 0.0

    def _calculate_redundancy(self, chunk_features: List[ChunkFeatures]) -> float:
        """Calculate redundancy among chunks"""
        if len(chunk_features) < 2:
            return 0.0

        # Average similarity as redundancy measure
        similarities = [cf.similarity_avg for cf in chunk_features]
        return np.mean(similarities)

    def _get_strategy_utility(self, strategy: str, query_features: QueryFeatures,
                            chunk_features: List[ChunkFeatures]) -> float:
        """Get utility score for a specific strategy"""
        # Simplified utility calculation
        base_utility = 0.5

        # Adjust based on strategy characteristics
        if strategy in ["S3", "G3"]:  # Complex strategies
            base_utility += 0.2 if query_features.ambiguity_score > 0.5 else -0.1
        elif strategy in ["S1", "G2"]:  # Simple strategies
            base_utility += 0.1 if query_features.ambiguity_score < 0.3 else -0.1

        return max(0, min(1, base_utility))

    def _calculate_selection_confidence(self, query_features: QueryFeatures,
                                      chunk_features: List[ChunkFeatures]) -> float:
        """Calculate confidence in strategy selection"""
        factors = [
            1 - query_features.ambiguity_score,  # Lower ambiguity = higher confidence
            min(1.0, query_features.query_length / 10.0),  # Longer queries = more info
            np.mean([cf.metadata_completeness for cf in chunk_features])  # Better metadata
        ]

        return np.mean(factors)

    def _generate_selection_reasoning(self, retrieval_strategy: str, generation_strategy: str,
                                    query_features: QueryFeatures,
                                    chunk_features: List[ChunkFeatures]) -> str:
        """Generate human-readable reasoning for strategy selection"""
        reasons = []

        # Retrieval strategy reasoning
        if retrieval_strategy == "S1":
            reasons.append("Selected simple dense retrieval due to clear query intent")
        elif retrieval_strategy == "S3":
            reasons.append("Selected multi-hop retrieval due to complex dependencies")
        elif retrieval_strategy == "S5":
            reasons.append("Selected query decomposition due to high ambiguity")

        # Generation strategy reasoning
        if generation_strategy == "G1":
            reasons.append("Selected fusion-in-decoder for multi-document synthesis")
        elif generation_strategy == "G3":
            reasons.append("Selected chain-of-thought for reasoning-heavy query")
        elif generation_strategy == "G5":
            reasons.append("Selected validator-augmented due to numeric content")

        return "; ".join(reasons) if reasons else "Default strategy selection"
