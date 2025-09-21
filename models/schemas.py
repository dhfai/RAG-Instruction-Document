from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"

class ModulAjarRequest(BaseModel):
    nama_guru: str
    nama_sekolah: str
    mata_pelajaran: str
    topik: str
    sub_topik: str
    alokasi_waktu: str
    kelas: str
    fase: str
    llm_used: LLMProvider = LLMProvider.OPENAI

class ChunkFeatures(BaseModel):
    chunk_id: str
    length: float
    unique_words: int
    lexical_diversity: float
    entropy: float
    similarity_avg: float
    dependency_score: float
    numeric_content: bool
    has_examples: bool
    metadata_completeness: float
    complexity_score: float

class QueryFeatures(BaseModel):
    query: str
    intent_vector: List[float]  # [lookup, reasoning, synthesis, numeric]
    ambiguity_score: float
    entity_count: int
    query_length: int

class StrategySelection(BaseModel):
    retrieval_strategy: str
    generation_strategy: str
    expected_utility: float
    confidence: float
    reasoning: str

class ValidationResult(BaseModel):
    faithfulness_score: float
    numeric_consistency: bool
    contradiction_detected: bool
    overall_confidence: float

class GeneratedSection(BaseModel):
    section_name: str
    content: str
    evidence_sources: List[str]
    confidence_score: float
    validation_result: ValidationResult

class ModulAjarResponse(BaseModel):
    request_id: str
    generated_sections: Dict[str, GeneratedSection]
    overall_quality_score: float
    total_cost: float
    processing_time: float
    strategies_used: Dict[str, StrategySelection]
    created_at: datetime = datetime.now()

class DocumentChunk(BaseModel):
    chunk_id: str
    content: str
    source_file: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    features: Optional[ChunkFeatures] = None

class ProcessingLog(BaseModel):
    timestamp: datetime
    level: str
    component: str
    message: str
    metadata: Optional[Dict[str, Any]] = None
