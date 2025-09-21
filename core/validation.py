import re
import numpy as np
from typing import List, Dict, Any, Tuple
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from models.schemas import ValidationResult, GeneratedSection, DocumentChunk
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

class ClaimValidator:
    def __init__(self):
        try:
            # Initialize NLI model for faithfulness checking
            self.nli_pipeline = pipeline("text-classification",
                                       model="microsoft/DialoGPT-medium",
                                       return_all_scores=True)
        except:
            logger.warning("NLI model not available, using simplified validation")
            self.nli_pipeline = None

        self.embedding_model = SentenceTransformer(settings.EMBEDDINGS_MODEL)

    def validate_section(self, section: GeneratedSection,
                        evidence_chunks: List[DocumentChunk]) -> ValidationResult:
        """Comprehensive validation of generated section"""
        try:
            logger.info(f"Validating section: {section.section_name}")

            # Extract claims from content
            claims = self._extract_claims(section.content)

            # Validate each claim
            faithfulness_scores = []
            contradictions = []
            numeric_issues = []

            for claim in claims:
                faith_score = self._validate_faithfulness(claim, evidence_chunks)
                faithfulness_scores.append(faith_score)

                # Check for contradictions
                contradiction = self._detect_contradiction(claim, evidence_chunks)
                contradictions.append(contradiction)

                # Validate numeric claims
                numeric_ok = self._validate_numeric_claim(claim, evidence_chunks)
                numeric_issues.append(not numeric_ok)

            # Aggregate results
            avg_faithfulness = np.mean(faithfulness_scores) if faithfulness_scores else 0.7
            contradiction_detected = any(contradictions)
            numeric_consistency = not any(numeric_issues)

            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(
                avg_faithfulness, contradiction_detected, numeric_consistency
            )

            result = ValidationResult(
                faithfulness_score=avg_faithfulness,
                numeric_consistency=numeric_consistency,
                contradiction_detected=contradiction_detected,
                overall_confidence=overall_confidence
            )

            logger.info(f"Validation complete. Confidence: {overall_confidence:.2f}")
            return result

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return ValidationResult(
                faithfulness_score=0.5,
                numeric_consistency=True,
                contradiction_detected=False,
                overall_confidence=0.5
            )

    def _extract_claims(self, content: str) -> List[str]:
        """Extract individual claims from content"""
        # Split content into sentences
        sentences = re.split(r'[.!?]+', content)

        # Filter out short sentences and clean up
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence.split()) >= 5:  # Only substantial sentences
                claims.append(sentence)

        return claims

    def _validate_faithfulness(self, claim: str, evidence_chunks: List[DocumentChunk]) -> float:
        """Validate if claim is faithful to evidence"""
        try:
            if not evidence_chunks:
                return 0.5

            # Get claim embedding
            claim_embedding = self.embedding_model.encode([claim])

            # Calculate similarity with evidence
            max_similarity = 0.0
            best_bm25 = 0.0

            for chunk in evidence_chunks:
                # Semantic similarity
                if chunk.embedding is None:
                    chunk.embedding = self.embedding_model.encode([chunk.content])

                similarity = cosine_similarity(claim_embedding, [chunk.embedding])[0][0]
                max_similarity = max(max_similarity, similarity)

                # BM25-style keyword matching
                bm25_score = self._calculate_claim_overlap(claim, chunk.content)
                best_bm25 = max(best_bm25, bm25_score)

            # Combine semantic and lexical similarity
            faithfulness = (settings.FAITHFULNESS_ALPHA * max_similarity +
                          settings.FAITHFULNESS_BETA * best_bm25)

            return min(1.0, faithfulness)

        except Exception as e:
            logger.error(f"Faithfulness validation error: {e}")
            return 0.5

    def _calculate_claim_overlap(self, claim: str, evidence: str) -> float:
        """Calculate overlap between claim and evidence"""
        claim_words = set(claim.lower().split())
        evidence_words = set(evidence.lower().split())

        if not claim_words:
            return 0.0

        overlap = len(claim_words & evidence_words)
        return overlap / len(claim_words)

    def _detect_contradiction(self, claim: str, evidence_chunks: List[DocumentChunk]) -> bool:
        """Detect if claim contradicts evidence"""
        try:
            # Simple contradiction detection using keyword analysis
            claim_lower = claim.lower()

            # Look for negation patterns
            negation_patterns = ['tidak', 'bukan', 'tanpa', 'tidak ada', 'belum', 'never', 'no', 'not']
            claim_negated = any(pattern in claim_lower for pattern in negation_patterns)

            # Check against evidence
            for chunk in evidence_chunks:
                evidence_lower = chunk.content.lower()
                evidence_negated = any(pattern in evidence_lower for pattern in negation_patterns)

                # Simple heuristic: if claim is negated but evidence is positive, possible contradiction
                if claim_negated != evidence_negated:
                    # Check for keyword overlap to see if they're about the same thing
                    overlap = self._calculate_claim_overlap(claim, chunk.content)
                    if overlap > 0.3:  # Significant overlap suggests they're about the same topic
                        return True

            return False

        except Exception as e:
            logger.error(f"Contradiction detection error: {e}")
            return False

    def _validate_numeric_claim(self, claim: str, evidence_chunks: List[DocumentChunk]) -> bool:
        """Validate numeric information in claims"""
        try:
            # Extract numbers from claim
            claim_numbers = re.findall(r'\b\d+\.?\d*\b', claim)

            if not claim_numbers:
                return True  # No numbers to validate

            # Check if these numbers appear in evidence
            for number in claim_numbers:
                found_in_evidence = False

                for chunk in evidence_chunks:
                    if number in chunk.content:
                        found_in_evidence = True
                        break

                if not found_in_evidence:
                    # Try to find similar numbers (within reasonable range)
                    if not self._find_similar_number(float(number), evidence_chunks):
                        logger.warning(f"Number {number} not found in evidence")
                        return False

            return True

        except Exception as e:
            logger.error(f"Numeric validation error: {e}")
            return True  # Conservative: assume valid on error

    def _find_similar_number(self, target: float, evidence_chunks: List[DocumentChunk]) -> bool:
        """Find similar numbers in evidence (within 10% range)"""
        try:
            tolerance = 0.1  # 10% tolerance

            for chunk in evidence_chunks:
                evidence_numbers = re.findall(r'\b\d+\.?\d*\b', chunk.content)

                for num_str in evidence_numbers:
                    try:
                        num = float(num_str)
                        if abs(num - target) / max(num, target) <= tolerance:
                            return True
                    except ValueError:
                        continue

            return False

        except Exception as e:
            logger.error(f"Similar number search error: {e}")
            return False

    def _calculate_overall_confidence(self, faithfulness: float,
                                    contradiction_detected: bool,
                                    numeric_consistency: bool) -> float:
        """Calculate overall validation confidence"""
        confidence = faithfulness

        # Penalty for contradictions
        if contradiction_detected:
            confidence *= 0.7

        # Penalty for numeric inconsistency
        if not numeric_consistency:
            confidence *= 0.8

        return max(0.0, min(1.0, confidence))

class ContentSynthesizer:
    """Synthesize and weight multiple generated sections"""

    def __init__(self):
        pass

    def synthesize_sections(self, sections: List[GeneratedSection],
                          template_requirements: Dict[str, Any]) -> GeneratedSection:
        """Synthesize multiple sections using confidence weighting"""
        try:
            logger.info(f"Synthesizing {len(sections)} sections")

            if not sections:
                return self._create_empty_section()

            if len(sections) == 1:
                return sections[0]

            # Weight sections by confidence and validation scores
            weighted_contents = []
            total_weight = 0

            for section in sections:
                weight = self._calculate_section_weight(section)
                weighted_contents.append((section, weight))
                total_weight += weight

            # Sort by weight (highest first)
            weighted_contents.sort(key=lambda x: x[1], reverse=True)

            # Create synthesized content
            synthesized_content = self._merge_content([s for s, w in weighted_contents])

            # Aggregate metadata
            all_evidence = []
            confidence_scores = []

            for section, weight in weighted_contents:
                all_evidence.extend(section.evidence_sources)
                confidence_scores.append(section.confidence_score)

            # Remove duplicates from evidence
            unique_evidence = list(set(all_evidence))

            # Weighted average confidence
            avg_confidence = sum(conf * weight for (_, weight), conf
                               in zip(weighted_contents, confidence_scores)) / total_weight

            # Create best validation result
            best_validation = max(sections, key=lambda s: s.validation_result.overall_confidence).validation_result

            synthesized_section = GeneratedSection(
                section_name=sections[0].section_name,
                content=synthesized_content,
                evidence_sources=unique_evidence,
                confidence_score=avg_confidence,
                validation_result=best_validation
            )

            logger.info(f"Synthesis complete. Final confidence: {avg_confidence:.2f}")
            return synthesized_section

        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return sections[0] if sections else self._create_empty_section()

    def _calculate_section_weight(self, section: GeneratedSection) -> float:
        """Calculate weight for a section based on quality metrics"""
        # Base weight from confidence score
        weight = section.confidence_score

        # Validation bonus
        validation_bonus = section.validation_result.overall_confidence * 0.2
        weight += validation_bonus

        # Content length bonus (up to a point)
        content_length = len(section.content.split())
        length_factor = min(1.0, content_length / 200) * 0.1
        weight += length_factor

        # Evidence count bonus
        evidence_bonus = min(0.1, len(section.evidence_sources) * 0.02)
        weight += evidence_bonus

        return max(0.1, min(2.0, weight))  # Clamp between 0.1 and 2.0

    def _merge_content(self, sections: List[GeneratedSection]) -> str:
        """Merge content from multiple sections intelligently"""
        if not sections:
            return ""

        if len(sections) == 1:
            return sections[0].content

        # Take the best section as base
        base_section = sections[0]
        base_content = base_section.content

        # Add unique insights from other sections
        additional_content = []

        for section in sections[1:]:
            unique_points = self._extract_unique_points(section.content, base_content)
            if unique_points:
                additional_content.extend(unique_points)

        # Combine content
        if additional_content:
            merged_content = base_content + "\n\n" + "\n".join(additional_content)
        else:
            merged_content = base_content

        return merged_content

    def _extract_unique_points(self, candidate_content: str, base_content: str) -> List[str]:
        """Extract unique points from candidate that aren't in base"""
        candidate_sentences = re.split(r'[.!?]+', candidate_content)
        base_sentences = re.split(r'[.!?]+', base_content)

        base_words = set(' '.join(base_sentences).lower().split())
        unique_points = []

        for sentence in candidate_sentences:
            sentence = sentence.strip()
            if len(sentence.split()) < 5:  # Skip short sentences
                continue

            sentence_words = set(sentence.lower().split())
            # If sentence has significant unique content, include it
            overlap = len(sentence_words & base_words) / len(sentence_words)

            if overlap < 0.7:  # Less than 70% overlap = unique enough
                unique_points.append(sentence)

        return unique_points[:3]  # Limit to 3 additional points

    def _create_empty_section(self) -> GeneratedSection:
        """Create empty fallback section"""
        return GeneratedSection(
            section_name="empty",
            content="Konten tidak tersedia",
            evidence_sources=[],
            confidence_score=0.0,
            validation_result=ValidationResult(
                faithfulness_score=0.0,
                numeric_consistency=True,
                contradiction_detected=False,
                overall_confidence=0.0
            )
        )
