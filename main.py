import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Dict, Any

from models.schemas import ModulAjarRequest, ModulAjarResponse
from core.orchestrator import OrchestratedRAGSystem
from utils.logger import get_logger, log_info, log_error, print_success, print_error

logger = get_logger(__name__)

# Global RAG system instance
rag_system = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global rag_system

    try:
        # Startup
        log_info("Starting Orchestrated RAG System", "STARTUP")
        rag_system = OrchestratedRAGSystem()

        success = await rag_system.initialize_system()
        if not success:
            raise RuntimeError("Failed to initialize RAG system")

        print_success("API Server ready to serve requests!")
        yield

    except Exception as e:
        log_error(f"Startup failed: {e}", "STARTUP")
        raise
    finally:
        # Shutdown
        if rag_system:
            rag_system.cleanup()
        log_info("RAG System shutdown completed", "SHUTDOWN")

# Create FastAPI app
app = FastAPI(
    title="Orchestrated RAG System - Modul Ajar Generator",
    description="Sistem RAG multi-strategi untuk pembuatan modul ajar otomatis dengan optimisasi matematis",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Orchestrated RAG System - Modul Ajar Generator",
        "version": "1.0.0",
        "status": "operational" if rag_system and rag_system.initialized else "initializing",
        "features": [
            "Multi-strategy retrieval (S1-S5)",
            "Multi-strategy generation (G1-G5)",
            "Mathematical feature engineering",
            "Cost-aware routing",
            "Claim validation",
            "Web content enhancement",
            "FAISS vector search",
            "MongoDB storage"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not rag_system or not rag_system.initialized:
        raise HTTPException(status_code=503, detail="System not ready")

    return {
        "status": "healthy",
        "components": {
            "rag_system": "operational",
            "vector_db": "loaded" if rag_system.chunks_loaded else "empty",
            "mongodb": "connected",
            "templates": "loaded"
        }
    }

@app.post("/generate", response_model=ModulAjarResponse)
async def generate_modul_ajar(request: ModulAjarRequest, background_tasks: BackgroundTasks):
    """Generate modul ajar based on request"""
    try:
        if not rag_system or not rag_system.initialized:
            raise HTTPException(status_code=503, detail="System not initialized")

        # Validate request
        if not request.topik.strip():
            raise HTTPException(status_code=400, detail="Topik tidak boleh kosong")

        if not request.mata_pelajaran.strip():
            raise HTTPException(status_code=400, detail="Mata pelajaran tidak boleh kosong")

        log_info(f"Received generation request: {request.topik} - {request.mata_pelajaran}", "API")

        # Generate modul ajar
        response = await rag_system.generate_modul_ajar(request)

        if response.overall_quality_score == 0.0:
            raise HTTPException(status_code=500, detail="Generation failed")

        log_info(f"Successfully generated modul ajar. Quality: {response.overall_quality_score:.2f}", "API")
        return response

    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Generation request failed: {e}", "API")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/history")
async def get_generation_history(limit: int = 10):
    """Get recent generation history"""
    try:
        if not rag_system or not rag_system.mongodb:
            raise HTTPException(status_code=503, detail="Database not available")

        results = await rag_system.mongodb.get_recent_results(limit)
        return {
            "history": results,
            "count": len(results)
        }

    except Exception as e:
        log_error(f"History request failed: {e}", "API")
        raise HTTPException(status_code=500, detail="Failed to fetch history")

@app.get("/strategies")
async def get_available_strategies():
    """Get information about available strategies"""
    return {
        "retrieval_strategies": {
            "S1": {
                "name": "Single-pass Dense",
                "description": "Retrieval sederhana menggunakan semantic similarity",
                "use_case": "Query yang jelas dan spesifik"
            },
            "S2": {
                "name": "Hybrid BM25+Dense",
                "description": "Kombinasi keyword matching dan semantic similarity",
                "use_case": "Query dengan term spesifik"
            },
            "S3": {
                "name": "Multi-hop Iterative",
                "description": "Retrieval bertahap untuk query kompleks",
                "use_case": "Query yang membutuhkan reasoning multi-step"
            },
            "S4": {
                "name": "Clustered Selection",
                "description": "Seleksi dokumen beragam dari cluster berbeda",
                "use_case": "Membutuhkan perspektif yang beragam"
            },
            "S5": {
                "name": "Query Rewrite/Decomposition",
                "description": "Dekomposisi query menjadi sub-query",
                "use_case": "Query ambigu atau kompleks"
            }
        },
        "generation_strategies": {
            "G1": {
                "name": "Fusion-in-Decoder",
                "description": "Fusi multi-dokumen dalam decoder",
                "use_case": "Sintesis dari banyak sumber"
            },
            "G2": {
                "name": "Rerank-then-Generate",
                "description": "Reranking dokumen sebelum generasi",
                "use_case": "Presisi tinggi dengan sumber terbatas"
            },
            "G3": {
                "name": "Chain-of-Thought",
                "description": "Generasi dengan reasoning eksplisit",
                "use_case": "Konten yang membutuhkan logical reasoning"
            },
            "G4": {
                "name": "Evidence-Aware",
                "description": "Generasi dengan sitasi eksplisit",
                "use_case": "Konten yang membutuhkan dukungan evidensi"
            },
            "G5": {
                "name": "Validator-Augmented",
                "description": "Generasi dengan validasi terintegrasi",
                "use_case": "Konten dengan angka atau klaim faktual"
            }
        }
    }

@app.get("/template")
async def get_template_structure():
    """Get modul ajar template structure"""
    try:
        if not rag_system or not rag_system.template_manager:
            raise HTTPException(status_code=503, detail="Template manager not available")

        template = rag_system.template_manager.load_template()
        return {
            "template": template,
            "section_count": len(template.get('sections', [])),
            "sections": [section['name'] for section in template.get('sections', [])]
        }

    except Exception as e:
        log_error(f"Template request failed: {e}", "API")
        raise HTTPException(status_code=500, detail="Failed to fetch template")

@app.get("/stats")
async def get_system_stats():
    """Get system statistics"""
    try:
        if not rag_system:
            raise HTTPException(status_code=503, detail="System not available")

        stats = {
            "system_status": "operational" if rag_system.initialized else "initializing",
            "vector_db": {
                "chunks_loaded": len(rag_system.vector_db.chunks) if rag_system.chunks_loaded else 0,
                "index_ready": rag_system.vector_db.index is not None
            },
            "components": {
                "feature_extractor": "ready",
                "routing_engine": "ready",
                "claim_validator": "ready",
                "content_synthesizer": "ready",
                "mongodb": "connected" if rag_system.mongodb.client else "disconnected"
            }
        }

        return stats

    except Exception as e:
        log_error(f"Stats request failed: {e}", "API")
        raise HTTPException(status_code=500, detail="Failed to fetch statistics")

if __name__ == "__main__":
    print_success("Starting Orchestrated RAG System API Server...")

    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info"
    )
