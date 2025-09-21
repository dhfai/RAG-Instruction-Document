import os
from pathlib import Path
from pydantic import BaseSettings
from typing import Dict, List, Any

class Settings(BaseSettings):
    # Database Configuration
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "rag_modul_ajar"

    # Vector Database
    FAISS_INDEX_PATH: str = "storage/faiss_index"
    EMBEDDINGS_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # LLM Configuration
    OPENAI_API_KEY: str = ""
    DEFAULT_LLM_MODEL: str = "gpt-3.5-turbo"
    MAX_TOKENS: int = 2000
    TEMPERATURE: float = 0.7

    # Data Paths
    DATA_PATH: str = "data/modul_ajar"
    TEMPLATE_PATH: str = "templates"

    # Feature Engineering Parameters
    ALPHA: float = 0.3
    BETA: float = 0.4
    GAMMA: float = 0.3

    # Strategy Parameters
    SIMILARITY_THRESHOLD: float = 0.8
    COMPLEXITY_THRESHOLD: float = 0.3
    AMBIGUITY_THRESHOLD: float = 0.7
    DEPENDENCY_THRESHOLD: float = 0.7
    REDUNDANCY_THRESHOLD: float = 0.6

    # Cost Parameters
    LAMBDA_1: float = 0.1  # Latency penalty
    LAMBDA_2: float = 0.05  # Token cost penalty

    # Strategy Weights - Retrieval
    S1_WEIGHTS: List[float] = [0.6, 0.3, 0.1]  # [similarity, length, metadata]
    S2_WEIGHTS: List[float] = [0.4, 0.4, 0.2]  # [similarity, BM25, entity_overlap]
    S3_WEIGHTS: List[float] = [0.5, 0.3, 0.2]  # [dependency, sim_avg, ambiguity]
    S4_WEIGHTS: List[float] = [0.6, 0.4]       # [cluster_diversity, marginal_gain]
    S5_WEIGHTS: List[float] = [0.4, 0.3, 0.3]  # [ambiguity, intent_reasoning, coverage]

    # Strategy Weights - Generation
    G1_WEIGHTS: List[float] = [0.4, 0.3, 0.3]  # [num_docs, doc_diversity, intent_synthesis]
    G2_WEIGHTS: List[float] = [0.4, 0.4, 0.2]  # [rerank_gain, sim_top1, redundancy]
    G3_WEIGHTS: List[float] = [0.5, 0.3, 0.2]  # [intent_reasoning, dependency, hop_count]
    G4_WEIGHTS: List[float] = [0.6, 0.4]       # [citation_need, risk_profile]
    G5_WEIGHTS: List[float] = [0.4, 0.3, 0.3]  # [numeric, risk_profile, claim_density]

    # Validation Parameters
    FAITHFULNESS_ALPHA: float = 0.7
    FAITHFULNESS_BETA: float = 0.3
    CONFIDENCE_THRESHOLD: float = 0.6

    # Web Scraping
    MAX_GOOGLE_RESULTS: int = 10
    WEB_SCRAPING_DELAY: float = 1.0

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/rag_system.log"

    class Config:
        env_file = ".env"
        case_sensitive = True

# Strategy Cost Configuration
STRATEGY_COSTS = {
    "S1": {"latency": 0.5, "tokens": 100},
    "S2": {"latency": 0.8, "tokens": 150},
    "S3": {"latency": 1.5, "tokens": 300},
    "S4": {"latency": 1.2, "tokens": 250},
    "S5": {"latency": 2.0, "tokens": 400},
    "G1": {"latency": 2.0, "tokens": 800},
    "G2": {"latency": 1.5, "tokens": 600},
    "G3": {"latency": 3.0, "tokens": 1200},
    "G4": {"latency": 1.8, "tokens": 700},
    "G5": {"latency": 2.5, "tokens": 1000},
}

# Template Configuration
TEMPLATE_SECTIONS = [
    "identitas",
    "tujuan_pembelajaran",
    "profil_pelajar_pancasila",
    "sarana_prasarana",
    "target_peserta_didik",
    "model_pembelajaran",
    "kegiatan_pembelajaran",
    "asesmen",
    "pengayaan_remedial",
    "refleksi_guru"
]

settings = Settings()
