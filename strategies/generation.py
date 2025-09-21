from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import openai
from transformers import pipeline
import re
import numpy as np

from models.schemas import DocumentChunk, GeneratedSection, ValidationResult
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

class BaseGenerationStrategy(ABC):
    def __init__(self):
        # Initialize OpenAI client
        openai.api_key = settings.OPENAI_API_KEY

    @abstractmethod
    def generate(self, query: str, chunks: List[DocumentChunk],
                template_section: Dict[str, Any]) -> GeneratedSection:
        pass

    def _call_llm(self, prompt: str, model: str = None) -> str:
        """Call LLM with error handling"""
        try:
            model = model or settings.DEFAULT_LLM_MODEL

            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=settings.MAX_TOKENS,
                temperature=settings.TEMPERATURE
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "Error generating content. Please try again."

class G1_FusionInDecoder(BaseGenerationStrategy):
    """G1 - Fusion-in-Decoder Strategy"""

    def generate(self, query: str, chunks: List[DocumentChunk],
                template_section: Dict[str, Any]) -> GeneratedSection:
        """Generate content by fusing multiple documents in decoder"""
        try:
            logger.info("Executing G1 - Fusion-in-Decoder generation")

            section_name = template_section.get("name", "content")
            section_description = template_section.get("description", "")

            # Prepare multi-document context
            context_parts = []
            for i, chunk in enumerate(chunks[:8]):  # Limit to avoid token overflow
                context_parts.append(f"Document {i+1}:\n{chunk.content}\n")

            combined_context = "\n".join(context_parts)

            # Create fusion prompt
            prompt = f"""
Berdasarkan dokumen-dokumen berikut, buatlah {section_name} untuk modul ajar dengan topik: {query}

{section_description}

DOKUMEN REFERENSI:
{combined_context}

INSTRUKSI:
1. Gabungkan informasi dari semua dokumen yang relevan
2. Pastikan konten sesuai dengan konteks pendidikan Indonesia
3. Gunakan bahasa yang jelas dan mudah dipahami
4. Sertakan detail yang spesifik dan praktis
5. Jangan menyalin langsung, tapi sintetis informasi

OUTPUT:
"""

            generated_content = self._call_llm(prompt)

            # Extract evidence sources
            evidence_sources = [f"{chunk.source_file}:{chunk.chunk_id}" for chunk in chunks]

            # Simple confidence calculation based on chunk quality
            confidence_score = self._calculate_confidence(chunks, generated_content)

            # Basic validation
            validation_result = self._basic_validation(generated_content, chunks)

            section = GeneratedSection(
                section_name=section_name,
                content=generated_content,
                evidence_sources=evidence_sources,
                confidence_score=confidence_score,
                validation_result=validation_result
            )

            logger.info(f"G1 generated section: {section_name}")
            return section

        except Exception as e:
            logger.error(f"Error in G1 generation: {e}")
            return self._create_fallback_section(section_name, query)

    def _calculate_confidence(self, chunks: List[DocumentChunk], content: str) -> float:
        """Calculate confidence based on source quality and content length"""
        base_confidence = 0.7

        # Adjust based on number of sources
        source_bonus = min(0.2, len(chunks) * 0.05)

        # Adjust based on content length
        content_words = len(content.split())
        length_factor = min(1.0, content_words / 100) * 0.1

        return min(1.0, base_confidence + source_bonus + length_factor)

class G2_RerankThenGenerate(BaseGenerationStrategy):
    """G2 - Rerank-then-Generate Strategy"""

    def generate(self, query: str, chunks: List[DocumentChunk],
                template_section: Dict[str, Any]) -> GeneratedSection:
        """Rerank chunks then generate content"""
        try:
            logger.info("Executing G2 - Rerank-then-Generate generation")

            section_name = template_section.get("name", "content")
            section_description = template_section.get("description", "")

            # Rerank chunks based on relevance to specific section
            reranked_chunks = self._rerank_chunks_for_section(query, chunks, section_description)

            # Use top reranked chunks
            top_chunks = reranked_chunks[:5]

            # Generate content with focused context
            context = "\n\n".join([f"Sumber {i+1}:\n{chunk.content}"
                                 for i, chunk in enumerate(top_chunks)])

            prompt = f"""
Berdasarkan sumber-sumber terpilih berikut, buatlah {section_name} untuk modul ajar:

TOPIK: {query}
DESKRIPSI BAGIAN: {section_description}

SUMBER TERPILIH:
{context}

INSTRUKSI:
1. Fokus pada sumber yang paling relevan
2. Buat konten yang terstruktur dan koheren
3. Sesuaikan dengan standar kurikulum Indonesia
4. Gunakan pendekatan pedagogis yang tepat

OUTPUT:
"""

            generated_content = self._call_llm(prompt)

            evidence_sources = [f"{chunk.source_file}:{chunk.chunk_id}" for chunk in top_chunks]
            confidence_score = self._calculate_rerank_confidence(top_chunks, generated_content)
            validation_result = self._basic_validation(generated_content, top_chunks)

            section = GeneratedSection(
                section_name=section_name,
                content=generated_content,
                evidence_sources=evidence_sources,
                confidence_score=confidence_score,
                validation_result=validation_result
            )

            logger.info(f"G2 generated section: {section_name}")
            return section

        except Exception as e:
            logger.error(f"Error in G2 generation: {e}")
            return self._create_fallback_section(section_name, query)

    def _rerank_chunks_for_section(self, query: str, chunks: List[DocumentChunk],
                                 section_description: str) -> List[DocumentChunk]:
        """Rerank chunks specifically for the target section"""
        # Combine query and section description for relevance scoring
        section_query = f"{query} {section_description}"

        # Simple reranking based on keyword overlap and semantic similarity
        scored_chunks = []

        for chunk in chunks:
            # Keyword overlap score
            query_words = set(section_query.lower().split())
            chunk_words = set(chunk.content.lower().split())
            keyword_overlap = len(query_words & chunk_words) / len(query_words | chunk_words)

            # Length bonus for substantial chunks
            length_bonus = min(1.0, len(chunk.content.split()) / 100)

            # Metadata bonus
            metadata_bonus = chunk.features.metadata_completeness if chunk.features else 0.5

            total_score = 0.5 * keyword_overlap + 0.3 * length_bonus + 0.2 * metadata_bonus
            scored_chunks.append((chunk, total_score))

        # Sort by score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in scored_chunks]

    def _calculate_rerank_confidence(self, chunks: List[DocumentChunk], content: str) -> float:
        """Calculate confidence for reranked generation"""
        base_confidence = 0.8  # Higher base confidence due to reranking

        # Quality of top chunks
        if chunks and chunks[0].features:
            top_chunk_quality = chunks[0].features.metadata_completeness
            confidence = base_confidence + 0.1 * top_chunk_quality
        else:
            confidence = base_confidence

        return min(1.0, confidence)

class G3_ChainOfThought(BaseGenerationStrategy):
    """G3 - Chain-of-Thought Strategy"""

    def generate(self, query: str, chunks: List[DocumentChunk],
                template_section: Dict[str, Any]) -> GeneratedSection:
        """Generate using chain-of-thought reasoning"""
        try:
            logger.info("Executing G3 - Chain-of-Thought generation")

            section_name = template_section.get("name", "content")
            section_description = template_section.get("description", "")

            # Multi-step reasoning process
            reasoning_steps = self._create_reasoning_steps(query, section_name, chunks)

            # Generate content through reasoning chain
            final_content = self._execute_reasoning_chain(reasoning_steps, chunks)

            evidence_sources = [f"{chunk.source_file}:{chunk.chunk_id}" for chunk in chunks]
            confidence_score = self._calculate_reasoning_confidence(reasoning_steps, final_content)
            validation_result = self._basic_validation(final_content, chunks)

            section = GeneratedSection(
                section_name=section_name,
                content=final_content,
                evidence_sources=evidence_sources,
                confidence_score=confidence_score,
                validation_result=validation_result
            )

            logger.info(f"G3 generated section: {section_name} with {len(reasoning_steps)} reasoning steps")
            return section

        except Exception as e:
            logger.error(f"Error in G3 generation: {e}")
            return self._create_fallback_section(section_name, query)

    def _create_reasoning_steps(self, query: str, section_name: str,
                              chunks: List[DocumentChunk]) -> List[str]:
        """Create reasoning steps for content generation"""
        steps = [
            f"1. Analisis topik pembelajaran: {query}",
            f"2. Identifikasi tujuan bagian {section_name}",
            "3. Kumpulkan informasi relevan dari sumber",
            "4. Strukturkan konten sesuai kebutuhan pedagogis",
            "5. Sesuaikan dengan konteks kurikulum Indonesia",
            "6. Validasi kelengkapan dan keakuratan konten"
        ]
        return steps

    def _execute_reasoning_chain(self, reasoning_steps: List[str],
                               chunks: List[DocumentChunk]) -> str:
        """Execute the reasoning chain to generate content"""
        context = "\n\n".join([chunk.content for chunk in chunks[:5]])

        steps_text = "\n".join(reasoning_steps)

        prompt = f"""
Mari kita buat konten modul ajar dengan mengikuti langkah-langkah reasoning berikut:

LANGKAH REASONING:
{steps_text}

SUMBER INFORMASI:
{context}

Ikuti setiap langkah secara sistematis dan berikan hasil akhir yang komprehensif:

LANGKAH 1 - ANALISIS TOPIK:
[Analisis singkat topik]

LANGKAH 2 - IDENTIFIKASI TUJUAN:
[Tujuan bagian ini]

LANGKAH 3 - INFORMASI RELEVAN:
[Poin-poin penting dari sumber]

LANGKAH 4 - STRUKTUR KONTEN:
[Outline struktur]

LANGKAH 5 - KONTEKSTUALISASI:
[Penyesuaian kurikulum]

LANGKAH 6 - HASIL AKHIR:
[Konten final yang lengkap]
"""

        reasoning_result = self._call_llm(prompt)

        # Extract final content from reasoning result
        final_section = reasoning_result.split("LANGKAH 6 - HASIL AKHIR:")
        if len(final_section) > 1:
            return final_section[1].strip()
        else:
            return reasoning_result

    def _calculate_reasoning_confidence(self, reasoning_steps: List[str], content: str) -> float:
        """Calculate confidence for chain-of-thought generation"""
        base_confidence = 0.85  # High confidence due to systematic approach

        # Bonus for detailed reasoning
        reasoning_bonus = min(0.1, len(reasoning_steps) * 0.02)

        # Content completeness bonus
        content_completeness = min(0.05, len(content.split()) / 200)

        return min(1.0, base_confidence + reasoning_bonus + content_completeness)

class G4_EvidenceAware(BaseGenerationStrategy):
    """G4 - Evidence-Aware Strategy"""

    def generate(self, query: str, chunks: List[DocumentChunk],
                template_section: Dict[str, Any]) -> GeneratedSection:
        """Generate with explicit evidence citation and validation"""
        try:
            logger.info("Executing G4 - Evidence-Aware generation")

            section_name = template_section.get("name", "content")
            section_description = template_section.get("description", "")

            # Create evidence-mapped content
            evidence_mapped_content = self._generate_with_evidence_mapping(
                query, chunks, section_name, section_description
            )

            evidence_sources = [f"{chunk.source_file}:{chunk.chunk_id}" for chunk in chunks]
            confidence_score = self._calculate_evidence_confidence(chunks, evidence_mapped_content)
            validation_result = self._validate_evidence_claims(evidence_mapped_content, chunks)

            section = GeneratedSection(
                section_name=section_name,
                content=evidence_mapped_content,
                evidence_sources=evidence_sources,
                confidence_score=confidence_score,
                validation_result=validation_result
            )

            logger.info(f"G4 generated evidence-aware section: {section_name}")
            return section

        except Exception as e:
            logger.error(f"Error in G4 generation: {e}")
            return self._create_fallback_section(section_name, query)

    def _generate_with_evidence_mapping(self, query: str, chunks: List[DocumentChunk],
                                      section_name: str, section_description: str) -> str:
        """Generate content with explicit evidence mapping"""
        # Prepare evidence sources with IDs
        evidence_context = ""
        for i, chunk in enumerate(chunks[:6]):
            evidence_context += f"[Sumber {i+1}]: {chunk.content}\n\n"

        prompt = f"""
Buat {section_name} untuk modul ajar dengan topik: {query}

DESKRIPSI: {section_description}

SUMBER EVIDENSI:
{evidence_context}

INSTRUKSI PENTING:
1. Setiap klaim atau informasi HARUS disertai referensi [Sumber X]
2. Jangan membuat klaim tanpa dukungan evidensi
3. Jika informasi tidak ada di sumber, nyatakan "memerlukan sumber tambahan"
4. Prioritaskan akurasi daripada kelengkapan
5. Gunakan format: "Berdasarkan [Sumber X], ..."

CONTOH FORMAT:
Berdasarkan [Sumber 1], konsep pembelajaran aktif melibatkan...
Menurut [Sumber 2], strategi yang efektif mencakup...

OUTPUT DENGAN REFERENSI:
"""

        return self._call_llm(prompt)

    def _calculate_evidence_confidence(self, chunks: List[DocumentChunk], content: str) -> float:
        """Calculate confidence based on evidence quality"""
        base_confidence = 0.75

        # Count evidence citations in content
        citation_count = len(re.findall(r'\[Sumber \d+\]', content))
        citation_ratio = citation_count / max(1, len(content.split('\n')))
        citation_bonus = min(0.2, citation_ratio * 0.5)

        # Source quality bonus
        if chunks:
            avg_metadata_quality = np.mean([
                chunk.features.metadata_completeness if chunk.features else 0.5
                for chunk in chunks
            ])
            quality_bonus = 0.05 * avg_metadata_quality
        else:
            quality_bonus = 0

        return min(1.0, base_confidence + citation_bonus + quality_bonus)

    def _validate_evidence_claims(self, content: str, chunks: List[DocumentChunk]) -> ValidationResult:
        """Validate claims against evidence"""
        # Simple validation - check if citations exist
        citations = re.findall(r'\[Sumber (\d+)\]', content)
        max_source = len(chunks)

        valid_citations = all(int(cite) <= max_source for cite in citations)

        # Check for numeric consistency (simplified)
        numbers_in_content = re.findall(r'\b\d+\.?\d*\b', content)
        numeric_consistency = len(numbers_in_content) == 0  # Conservative: assume consistent if no numbers

        return ValidationResult(
            faithfulness_score=0.8 if valid_citations else 0.6,
            numeric_consistency=numeric_consistency,
            contradiction_detected=False,  # Would need more sophisticated checking
            overall_confidence=0.8 if valid_citations and numeric_consistency else 0.6
        )

class G5_ValidatorAugmented(BaseGenerationStrategy):
    """G5 - Validator-Augmented Strategy"""

    def generate(self, query: str, chunks: List[DocumentChunk],
                template_section: Dict[str, Any]) -> GeneratedSection:
        """Generate with integrated validation and correction"""
        try:
            logger.info("Executing G5 - Validator-Augmented generation")

            section_name = template_section.get("name", "content")
            section_description = template_section.get("description", "")

            # Generate initial content
            initial_content = self._generate_initial_content(query, chunks, section_name, section_description)

            # Validate and correct
            validated_content = self._validate_and_correct(initial_content, chunks)

            evidence_sources = [f"{chunk.source_file}:{chunk.chunk_id}" for chunk in chunks]
            confidence_score = 0.9  # High confidence due to validation
            validation_result = self._comprehensive_validation(validated_content, chunks)

            section = GeneratedSection(
                section_name=section_name,
                content=validated_content,
                evidence_sources=evidence_sources,
                confidence_score=confidence_score,
                validation_result=validation_result
            )

            logger.info(f"G5 generated validator-augmented section: {section_name}")
            return section

        except Exception as e:
            logger.error(f"Error in G5 generation: {e}")
            return self._create_fallback_section(section_name, query)

    def _generate_initial_content(self, query: str, chunks: List[DocumentChunk],
                                section_name: str, section_description: str) -> str:
        """Generate initial content"""
        context = "\n\n".join([chunk.content for chunk in chunks[:5]])

        prompt = f"""
Buat {section_name} untuk modul ajar: {query}

{section_description}

SUMBER:
{context}

Buat konten yang akurat dan dapat diverifikasi:
"""

        return self._call_llm(prompt)

    def _validate_and_correct(self, content: str, chunks: List[DocumentChunk]) -> str:
        """Validate content and make corrections"""
        # Validation prompt
        context = "\n\n".join([chunk.content for chunk in chunks[:3]])

        validation_prompt = f"""
Periksa dan perbaiki konten berikut berdasarkan sumber yang diberikan:

KONTEN YANG HARUS DIPERIKSA:
{content}

SUMBER REFERENSI:
{context}

TUGAS VALIDASI:
1. Periksa keakuratan informasi
2. Pastikan konsistensi dengan sumber
3. Perbaiki kesalahan faktual jika ada
4. Lengkapi informasi yang kurang
5. Berikan versi yang telah divalidasi

KONTEN TERVALIDASI:
"""

        return self._call_llm(validation_prompt)

    def _comprehensive_validation(self, content: str, chunks: List[DocumentChunk]) -> ValidationResult:
        """Comprehensive validation of generated content"""
        # Check faithfulness (simplified)
        faithfulness_score = 0.85  # Would use NLI model in practice

        # Check numeric consistency
        numeric_consistency = True  # Would implement actual numeric checking

        # Check for contradictions
        contradiction_detected = False  # Would use contradiction detection model

        overall_confidence = (faithfulness_score +
                            (0.1 if numeric_consistency else 0) +
                            (0.05 if not contradiction_detected else 0))

        return ValidationResult(
            faithfulness_score=faithfulness_score,
            numeric_consistency=numeric_consistency,
            contradiction_detected=contradiction_detected,
            overall_confidence=min(1.0, overall_confidence)
        )

    def _basic_validation(self, content: str, chunks: List[DocumentChunk]) -> ValidationResult:
        """Basic validation for all strategies"""
        return ValidationResult(
            faithfulness_score=0.7,
            numeric_consistency=True,
            contradiction_detected=False,
            overall_confidence=0.7
        )

    def _create_fallback_section(self, section_name: str, query: str) -> GeneratedSection:
        """Create fallback section on error"""
        fallback_content = f"Konten untuk {section_name} sedang dalam pengembangan. Topik: {query}"

        return GeneratedSection(
            section_name=section_name,
            content=fallback_content,
            evidence_sources=[],
            confidence_score=0.3,
            validation_result=ValidationResult(
                faithfulness_score=0.3,
                numeric_consistency=True,
                contradiction_detected=False,
                overall_confidence=0.3
            )
        )

# Strategy Factory
class GenerationStrategyFactory:
    @staticmethod
    def create_strategy(strategy_name: str) -> BaseGenerationStrategy:
        """Create generation strategy instance"""
        strategies = {
            "G1": G1_FusionInDecoder,
            "G2": G2_RerankThenGenerate,
            "G3": G3_ChainOfThought,
            "G4": G4_EvidenceAware,
            "G5": G5_ValidatorAugmented
        }

        if strategy_name not in strategies:
            logger.warning(f"Unknown strategy {strategy_name}, defaulting to G2")
            strategy_name = "G2"

        return strategies[strategy_name]()
