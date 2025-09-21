import time
import uuid
from datetime import datetime
from typing import List, Dict, Any

from models.schemas import (
    ModulAjarRequest, ModulAjarResponse, DocumentChunk,
    GeneratedSection, StrategySelection, ProcessingLog
)
from core.feature_extraction import FeatureExtractor
from core.routing_engine import RouterEngine
from core.validation import ClaimValidator, ContentSynthesizer
from strategies.retrieval import RetrievalStrategyFactory
from strategies.generation import GenerationStrategyFactory
from storage.database import VectorDatabase, MongoDBManager, DocumentProcessor, TemplateManager
from utils.web_scraper import ContentEnhancer
from utils.logger import (
    get_logger, print_banner, print_section_header, print_progress,
    print_success, print_error, print_strategy_selection,
    print_validation_result, log_system_startup, log_component_status
)
from config.settings import settings

logger = get_logger(__name__)

class OrchestratedRAGSystem:
    """Main orchestrator for the multi-strategy RAG system"""

    def __init__(self):
        # Initialize components
        self.feature_extractor = FeatureExtractor()
        self.routing_engine = RouterEngine()
        self.claim_validator = ClaimValidator()
        self.content_synthesizer = ContentSynthesizer()
        self.vector_db = VectorDatabase()
        self.mongodb = MongoDBManager()
        self.doc_processor = DocumentProcessor()
        self.template_manager = TemplateManager()
        self.content_enhancer = ContentEnhancer()

        # System state
        self.initialized = False
        self.chunks_loaded = False

    async def initialize_system(self) -> bool:
        """Initialize the RAG system"""
        try:
            print_banner()
            log_system_startup()

            print_section_header("System Initialization")

            # Connect to MongoDB
            print_progress("Connecting to MongoDB", 1, 5)
            if not self.mongodb.connect():
                log_component_status("MongoDB", "ERROR", "Connection failed")
                return False
            log_component_status("MongoDB", "READY")

            # Load or build vector index
            print_progress("Loading vector database", 2, 5)
            if not await self._initialize_vector_db():
                log_component_status("VectorDB", "ERROR", "Initialization failed")
                return False
            log_component_status("VectorDB", "READY", f"{len(self.vector_db.chunks)} chunks")

            # Initialize LLM components
            print_progress("Initializing AI models", 3, 5)
            log_component_status("FeatureExtractor", "READY")
            log_component_status("RouterEngine", "READY")

            # Load templates
            print_progress("Loading templates", 4, 5)
            self.template = self.template_manager.load_template()
            log_component_status("TemplateManager", "READY", f"{len(self.template.get('sections', []))} sections")

            print_progress("System initialization complete", 5, 5)
            self.initialized = True
            print_success("RAG System initialized successfully! 🎉")

            return True

        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            log_component_status("SYSTEM", "ERROR", str(e))
            return False

    async def _initialize_vector_db(self) -> bool:
        """Initialize vector database"""
        try:
            # Try to load existing index
            if self.vector_db.load_index():
                self.chunks_loaded = True
                return True

            # Build new index from documents
            print_progress("Processing local documents")
            chunks = self.doc_processor.process_local_documents()

            if not chunks:
                logger.warning("No local documents found, system will rely on web content")
                return True  # Allow system to continue

            print_progress(f"Building vector index for {len(chunks)} chunks")
            self.vector_db.build_index(chunks)
            self.chunks_loaded = True

            return True

        except Exception as e:
            logger.error(f"Vector DB initialization failed: {e}")
            return False

    async def generate_modul_ajar(self, request: ModulAjarRequest) -> ModulAjarResponse:
        """Generate modul ajar based on request"""
        try:
            if not self.initialized:
                raise RuntimeError("System not initialized. Call initialize_system() first.")

            # Generate unique request ID
            request_id = str(uuid.uuid4())
            start_time = time.time()

            print_section_header(f"Generating Modul Ajar - {request.topik}")
            logger.info(f"Processing request {request_id}: {request.topik} - {request.sub_topik}")

            # Step 1: Retrieve relevant content
            print_progress("Retrieving relevant content", 1, 6)
            chunks = await self._retrieve_content(request)

            if not chunks:
                logger.warning("No relevant content found")
                return self._create_error_response(request_id, "No relevant content found")

            # Step 2: Extract features
            print_progress("Analyzing content features", 2, 6)
            query = f"{request.topik} {request.sub_topik} {request.mata_pelajaran}"
            query_features, chunk_features, similarities = await self._extract_features(query, chunks)

            # Step 3: Select strategies
            print_progress("Selecting optimal strategies", 3, 6)
            strategies = await self._select_strategies(query_features, chunk_features, similarities)

            # Step 4: Generate content sections
            print_progress("Generating content sections", 4, 6)
            generated_sections = await self._generate_sections(
                request, chunks, strategies, chunk_features
            )

            # Step 5: Validate content
            print_progress("Validating generated content", 5, 6)
            validated_sections = await self._validate_sections(generated_sections, chunks)

            # Step 6: Create final response
            print_progress("Finalizing response", 6, 6)
            response = await self._create_response(
                request_id, request, validated_sections, strategies, start_time
            )

            # Save to database
            self.mongodb.save_result(response)

            print_success(f"Modul Ajar generated successfully! Quality Score: {response.overall_quality_score:.2f}")
            return response

        except Exception as e:
            logger.error(f"Error generating modul ajar: {e}")
            return self._create_error_response(
                request_id if 'request_id' in locals() else str(uuid.uuid4()),
                f"Generation failed: {str(e)}"
            )

    async def _retrieve_content(self, request: ModulAjarRequest) -> List[DocumentChunk]:
        """Retrieve relevant content for the request"""
        try:
            query = f"{request.topik} {request.sub_topik} {request.mata_pelajaran}"

            # Search vector database
            chunks = []
            if self.chunks_loaded:
                chunks = self.vector_db.search(query, top_k=15)

            # Enhance with web content if needed
            enhanced_chunks = self.content_enhancer.enhance_chunks(
                chunks, request.topik, request.sub_topik
            )

            logger.info(f"Retrieved {len(enhanced_chunks)} chunks ({len(chunks)} local, {len(enhanced_chunks) - len(chunks)} web)")
            return enhanced_chunks[:20]  # Limit to top 20 chunks

        except Exception as e:
            logger.error(f"Content retrieval failed: {e}")
            return []

    async def _extract_features(self, query: str, chunks: List[DocumentChunk]):
        """Extract features from query and chunks"""
        try:
            # Extract query features
            query_features = self.feature_extractor.extract_query_features(query)

            # Extract chunk features
            chunk_contents = [chunk.content for chunk in chunks]
            chunk_features = []

            for chunk in chunks:
                features = self.feature_extractor.extract_chunk_features(
                    chunk.content, chunk.chunk_id, chunk_contents
                )
                chunk.features = features
                chunk_features.append(features)

            # Calculate similarities
            similarities = []
            for chunk in chunks:
                similarity = self.feature_extractor.calculate_query_chunk_similarity(
                    query, chunk.content
                )
                similarities.append(similarity)

            return query_features, chunk_features, similarities

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise

    async def _select_strategies(self, query_features, chunk_features, similarities):
        """Select optimal retrieval and generation strategies"""
        try:
            # Select retrieval strategy
            retrieval_strategy = self.routing_engine.select_retrieval_strategy(
                query_features, chunk_features, similarities
            )

            # Select generation strategy
            generation_strategy = self.routing_engine.select_generation_strategy(
                query_features, chunk_features, [chunk.content for chunk in self.vector_db.chunks[:10]]
            )

            # Create strategy selection
            strategy_selection = self.routing_engine.create_strategy_selection(
                retrieval_strategy, generation_strategy, query_features, chunk_features
            )

            print_strategy_selection(
                retrieval_strategy, generation_strategy, strategy_selection.confidence
            )

            return strategy_selection

        except Exception as e:
            logger.error(f"Strategy selection failed: {e}")
            raise

    async def _generate_sections(self, request: ModulAjarRequest, chunks: List[DocumentChunk],
                               strategies: StrategySelection, chunk_features) -> Dict[str, GeneratedSection]:
        """Generate all sections of the modul ajar"""
        try:
            generated_sections = {}

            # Get retrieval and generation strategies
            retrieval_strategy = RetrievalStrategyFactory.create_strategy(strategies.retrieval_strategy)
            generation_strategy = GenerationStrategyFactory.create_strategy(strategies.generation_strategy)

            # Generate content for each template section
            template_sections = self.template.get('sections', [])

            for i, section_template in enumerate(template_sections):
                section_name = section_template['name']
                print_progress(f"Generating {section_name}", i + 1, len(template_sections))

                try:
                    # Apply retrieval strategy
                    relevant_chunks = retrieval_strategy.retrieve(
                        f"{request.topik} {section_template.get('description', '')}",
                        chunks,
                        top_k=8
                    )

                    # Apply generation strategy
                    generated_section = generation_strategy.generate(
                        f"{request.topik} {request.sub_topik}",
                        relevant_chunks,
                        section_template
                    )

                    # Add request context to identity section
                    if section_name == "identitas":
                        generated_section.content = self._create_identity_section(request)

                    generated_sections[section_name] = generated_section

                except Exception as e:
                    logger.error(f"Error generating section {section_name}: {e}")
                    # Create fallback section
                    generated_sections[section_name] = GeneratedSection(
                        section_name=section_name,
                        content=f"Bagian {section_name} sedang dalam pengembangan.",
                        evidence_sources=[],
                        confidence_score=0.3,
                        validation_result=self.claim_validator.validate_section(
                            GeneratedSection(section_name=section_name, content="", evidence_sources=[], confidence_score=0.3, validation_result=None),
                            []
                        )
                    )

            return generated_sections

        except Exception as e:
            logger.error(f"Section generation failed: {e}")
            raise

    async def _validate_sections(self, sections: Dict[str, GeneratedSection],
                               chunks: List[DocumentChunk]) -> Dict[str, GeneratedSection]:
        """Validate generated sections"""
        try:
            validated_sections = {}

            for section_name, section in sections.items():
                print_progress(f"Validating {section_name}")

                # Validate section content
                validation_result = self.claim_validator.validate_section(section, chunks)
                section.validation_result = validation_result

                print_validation_result(validation_result)

                validated_sections[section_name] = section

            return validated_sections

        except Exception as e:
            logger.error(f"Section validation failed: {e}")
            return sections  # Return unvalidated sections

    async def _create_response(self, request_id: str, request: ModulAjarRequest,
                             sections: Dict[str, GeneratedSection],
                             strategies: StrategySelection, start_time: float) -> ModulAjarResponse:
        """Create final response"""
        try:
            # Calculate overall quality score
            quality_scores = [section.confidence_score for section in sections.values()]
            overall_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

            # Calculate processing time and cost
            processing_time = time.time() - start_time
            total_cost = self._estimate_cost(sections, strategies)

            # Create strategies used dict
            strategies_used = {
                "main_strategy": strategies
            }

            response = ModulAjarResponse(
                request_id=request_id,
                generated_sections=sections,
                overall_quality_score=overall_quality,
                total_cost=total_cost,
                processing_time=processing_time,
                strategies_used=strategies_used,
                created_at=datetime.now()
            )

            return response

        except Exception as e:
            logger.error(f"Response creation failed: {e}")
            raise

    def _create_identity_section(self, request: ModulAjarRequest) -> str:
        """Create identity section from request data"""
        return f"""
IDENTITAS MODUL AJAR

Nama Guru: {request.nama_guru}
Nama Sekolah: {request.nama_sekolah}
Mata Pelajaran: {request.mata_pelajaran}
Topik: {request.topik}
Sub Topik: {request.sub_topik}
Kelas: {request.kelas}
Fase: {request.fase}
Alokasi Waktu: {request.alokasi_waktu}

Modul ajar ini disusun untuk mendukung pembelajaran {request.mata_pelajaran}
dengan fokus pada topik "{request.topik}" khususnya sub-topik "{request.sub_topik}".
        """.strip()

    def _estimate_cost(self, sections: Dict[str, GeneratedSection],
                      strategies: StrategySelection) -> float:
        """Estimate processing cost"""
        # Simple cost estimation based on content length and strategies
        total_tokens = sum(len(section.content.split()) for section in sections.values())
        base_cost = total_tokens * 0.001  # $0.001 per token estimate

        # Strategy cost multiplier
        strategy_multiplier = 1.2 if strategies.retrieval_strategy in ["S3", "S5"] else 1.0
        strategy_multiplier *= 1.3 if strategies.generation_strategy in ["G3", "G5"] else 1.0

        return base_cost * strategy_multiplier

    def _create_error_response(self, request_id: str, error_message: str) -> ModulAjarResponse:
        """Create error response"""
        print_error(error_message)

        return ModulAjarResponse(
            request_id=request_id,
            generated_sections={},
            overall_quality_score=0.0,
            total_cost=0.0,
            processing_time=0.0,
            strategies_used={},
            created_at=datetime.now()
        )

    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.mongodb:
                self.mongodb.close()
            if self.content_enhancer and self.content_enhancer.web_scraper:
                self.content_enhancer.web_scraper.close()
            logger.info("System cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
