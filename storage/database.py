import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import pickle
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from docx import Document
import PyPDF2

from models.schemas import DocumentChunk, ModulAjarResponse, ProcessingLog
from core.feature_extraction import FeatureExtractor
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

class DocumentProcessor:
    """Process and chunk documents from local files"""

    def __init__(self):
        self.feature_extractor = FeatureExtractor()

    def process_local_documents(self) -> List[DocumentChunk]:
        """Process all documents in the data directory"""
        try:
            logger.info("Processing local documents")

            data_path = Path(settings.DATA_PATH)
            if not data_path.exists():
                logger.error(f"Data path not found: {data_path}")
                return []

            chunks = []

            # Process all document files
            for file_path in data_path.glob("**/*"):
                if file_path.is_file() and file_path.suffix.lower() in ['.docx', '.pdf', '.txt']:
                    logger.info(f"Processing file: {file_path.name}")

                    try:
                        file_chunks = self._process_single_file(file_path)
                        chunks.extend(file_chunks)
                        logger.info(f"Extracted {len(file_chunks)} chunks from {file_path.name}")
                    except Exception as e:
                        logger.error(f"Error processing {file_path.name}: {e}")

            logger.info(f"Total chunks processed: {len(chunks)}")
            return chunks

        except Exception as e:
            logger.error(f"Error processing local documents: {e}")
            return []

    def _process_single_file(self, file_path: Path) -> List[DocumentChunk]:
        """Process a single file and return chunks"""
        try:
            # Extract text based on file type
            if file_path.suffix.lower() == '.docx':
                text = self._extract_from_docx(file_path)
            elif file_path.suffix.lower() == '.pdf':
                text = self._extract_from_pdf(file_path)
            elif file_path.suffix.lower() == '.txt':
                text = self._extract_from_txt(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_path.suffix}")
                return []

            if not text.strip():
                logger.warning(f"No text extracted from {file_path.name}")
                return []

            # Chunk the text
            chunks = self._chunk_text(text, str(file_path))

            return chunks

        except Exception as e:
            logger.error(f"Error processing single file {file_path}: {e}")
            return []

    def _extract_from_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            paragraphs = []

            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    paragraphs.append(paragraph.text.strip())

            return '\n\n'.join(paragraphs)

        except Exception as e:
            logger.error(f"Error extracting from DOCX {file_path}: {e}")
            return ""

    def _extract_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_parts = []

                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text.strip():
                        text_parts.append(text.strip())

                return '\n\n'.join(text_parts)

        except Exception as e:
            logger.error(f"Error extracting from PDF {file_path}: {e}")
            return ""

    def _extract_from_txt(self, file_path: Path) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()

        except Exception as e:
            logger.error(f"Error extracting from TXT {file_path}: {e}")
            return ""

    def _chunk_text(self, text: str, source_file: str) -> List[DocumentChunk]:
        """Chunk text into smaller pieces"""
        try:
            # Simple chunking by paragraphs or sentences
            paragraphs = text.split('\n\n')
            chunks = []

            current_chunk = ""
            chunk_id = 0

            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue

                # If adding this paragraph would make chunk too long, start new chunk
                if len(current_chunk) + len(paragraph) > 500 and current_chunk:
                    # Create chunk
                    chunk = DocumentChunk(
                        chunk_id=f"{Path(source_file).stem}_{chunk_id:03d}",
                        content=current_chunk.strip(),
                        source_file=str(source_file),
                        metadata={
                            "file_name": Path(source_file).name,
                            "chunk_index": chunk_id,
                            "word_count": len(current_chunk.split())
                        }
                    )
                    chunks.append(chunk)

                    current_chunk = paragraph
                    chunk_id += 1
                else:
                    current_chunk += "\n\n" + paragraph if current_chunk else paragraph

            # Add final chunk
            if current_chunk.strip():
                chunk = DocumentChunk(
                    chunk_id=f"{Path(source_file).stem}_{chunk_id:03d}",
                    content=current_chunk.strip(),
                    source_file=str(source_file),
                    metadata={
                        "file_name": Path(source_file).name,
                        "chunk_index": chunk_id,
                        "word_count": len(current_chunk.split())
                    }
                )
                chunks.append(chunk)

            return chunks

        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            return []

class VectorDatabase:
    """FAISS vector database for semantic search"""

    def __init__(self):
        self.index = None
        self.chunks = []
        self.embeddings = None
        self.feature_extractor = FeatureExtractor()
        self.index_path = Path(settings.FAISS_INDEX_PATH)

    def build_index(self, chunks: List[DocumentChunk]) -> None:
        """Build FAISS index from document chunks"""
        try:
            logger.info(f"Building FAISS index for {len(chunks)} chunks")

            if not chunks:
                logger.warning("No chunks provided for indexing")
                return

            # Generate embeddings for all chunks
            embeddings = []
            valid_chunks = []

            for chunk in chunks:
                try:
                    # Generate embedding
                    embedding = self.feature_extractor.embedding_model.encode([chunk.content])
                    embeddings.append(embedding[0])

                    # Store embedding in chunk
                    chunk.embedding = embedding[0].tolist()
                    valid_chunks.append(chunk)

                except Exception as e:
                    logger.error(f"Error generating embedding for chunk {chunk.chunk_id}: {e}")

            if not embeddings:
                logger.error("No valid embeddings generated")
                return

            embeddings = np.array(embeddings).astype('float32')

            # Create FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)

            # Add to index
            self.index.add(embeddings)

            self.chunks = valid_chunks
            self.embeddings = embeddings

            # Save index and chunks
            self._save_index()

            logger.info(f"FAISS index built successfully with {len(valid_chunks)} chunks")

        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")

    def search(self, query: str, top_k: int = 10) -> List[DocumentChunk]:
        """Search for similar chunks using FAISS"""
        try:
            if self.index is None:
                logger.error("Index not built. Call build_index first.")
                return []

            # Generate query embedding
            query_embedding = self.feature_extractor.embedding_model.encode([query])
            query_embedding = query_embedding.astype('float32')
            faiss.normalize_L2(query_embedding)

            # Search
            scores, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))

            # Return matching chunks
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and idx < len(self.chunks):  # Valid index
                    chunk = self.chunks[idx]
                    results.append(chunk)

            logger.info(f"Found {len(results)} results for query")
            return results

        except Exception as e:
            logger.error(f"Error searching FAISS index: {e}")
            return []

    def _save_index(self) -> None:
        """Save FAISS index and chunks to disk"""
        try:
            # Create storage directory
            self.index_path.parent.mkdir(parents=True, exist_ok=True)

            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path / "faiss.index"))

            # Save chunks metadata
            chunks_data = []
            for chunk in self.chunks:
                chunk_dict = {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "source_file": chunk.source_file,
                    "metadata": chunk.metadata,
                    "embedding": chunk.embedding
                }
                chunks_data.append(chunk_dict)

            with open(self.index_path / "chunks.pkl", 'wb') as f:
                pickle.dump(chunks_data, f)

            logger.info("Index and chunks saved successfully")

        except Exception as e:
            logger.error(f"Error saving index: {e}")

    def load_index(self) -> bool:
        """Load FAISS index and chunks from disk"""
        try:
            index_file = self.index_path / "faiss.index"
            chunks_file = self.index_path / "chunks.pkl"

            if not index_file.exists() or not chunks_file.exists():
                logger.info("No saved index found")
                return False

            # Load FAISS index
            self.index = faiss.read_index(str(index_file))

            # Load chunks
            with open(chunks_file, 'rb') as f:
                chunks_data = pickle.load(f)

            # Reconstruct chunks
            self.chunks = []
            for chunk_dict in chunks_data:
                chunk = DocumentChunk(
                    chunk_id=chunk_dict["chunk_id"],
                    content=chunk_dict["content"],
                    source_file=chunk_dict["source_file"],
                    metadata=chunk_dict["metadata"],
                    embedding=chunk_dict["embedding"]
                )
                self.chunks.append(chunk)

            logger.info(f"Loaded index with {len(self.chunks)} chunks")
            return True

        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False

class MongoDBManager:
    """MongoDB manager for storing results and logs"""

    def __init__(self):
        self.client = None
        self.db = None
        self.async_client = None
        self.async_db = None

    def connect(self) -> bool:
        """Connect to MongoDB"""
        try:
            # Synchronous client
            self.client = MongoClient(settings.MONGODB_URL)
            self.db = self.client[settings.MONGODB_DB_NAME]

            # Test connection
            self.client.admin.command('ping')

            # Async client
            self.async_client = AsyncIOMotorClient(settings.MONGODB_URL)
            self.async_db = self.async_client[settings.MONGODB_DB_NAME]

            logger.info("Connected to MongoDB successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False

    def save_result(self, result: ModulAjarResponse) -> bool:
        """Save modul ajar result to MongoDB"""
        try:
            if not self.db:
                logger.error("Not connected to MongoDB")
                return False

            # Convert to dict
            result_dict = result.dict()

            # Insert into collection
            collection = self.db.modul_ajar_results
            inserted = collection.insert_one(result_dict)

            logger.info(f"Saved result with ID: {inserted.inserted_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving result to MongoDB: {e}")
            return False

    def save_log(self, log: ProcessingLog) -> bool:
        """Save processing log to MongoDB"""
        try:
            if not self.db:
                return False

            log_dict = log.dict()
            collection = self.db.processing_logs
            collection.insert_one(log_dict)

            return True

        except Exception as e:
            logger.error(f"Error saving log: {e}")
            return False

    async def get_recent_results(self, limit: int = 10) -> List[Dict]:
        """Get recent results from MongoDB"""
        try:
            if not self.async_db:
                return []

            collection = self.async_db.modul_ajar_results
            cursor = collection.find().sort("created_at", -1).limit(limit)

            results = []
            async for document in cursor:
                # Convert ObjectId to string
                document['_id'] = str(document['_id'])
                results.append(document)

            return results

        except Exception as e:
            logger.error(f"Error getting recent results: {e}")
            return []

    def close(self):
        """Close database connections"""
        if self.client:
            self.client.close()
        if self.async_client:
            self.async_client.close()

class TemplateManager:
    """Manage modul ajar templates"""

    def __init__(self):
        self.template_path = Path(settings.TEMPLATE_PATH)

    def load_template(self, template_name: str = "template.json") -> Dict[str, Any]:
        """Load template from JSON file"""
        try:
            template_file = self.template_path / template_name

            if not template_file.exists():
                logger.error(f"Template file not found: {template_file}")
                return self._get_default_template()

            with open(template_file, 'r', encoding='utf-8') as f:
                template = json.load(f)

            logger.info(f"Loaded template: {template_name}")
            return template

        except Exception as e:
            logger.error(f"Error loading template: {e}")
            return self._get_default_template()

    def _get_default_template(self) -> Dict[str, Any]:
        """Get default template structure"""
        return {
            "sections": [
                {
                    "name": "identitas",
                    "description": "Identitas modul ajar termasuk nama guru, sekolah, mata pelajaran, topik, kelas, dan alokasi waktu"
                },
                {
                    "name": "tujuan_pembelajaran",
                    "description": "Tujuan pembelajaran yang spesifik, terukur, dan dapat dicapai peserta didik"
                },
                {
                    "name": "profil_pelajar_pancasila",
                    "description": "Profil Pelajar Pancasila yang akan dikembangkan melalui pembelajaran ini"
                },
                {
                    "name": "sarana_prasarana",
                    "description": "Sarana dan prasarana yang dibutuhkan untuk pelaksanaan pembelajaran"
                },
                {
                    "name": "target_peserta_didik",
                    "description": "Karakteristik dan target peserta didik yang mengikuti pembelajaran"
                },
                {
                    "name": "model_pembelajaran",
                    "description": "Model dan metode pembelajaran yang akan digunakan"
                },
                {
                    "name": "kegiatan_pembelajaran",
                    "description": "Langkah-langkah kegiatan pembelajaran dari pembukaan hingga penutup"
                },
                {
                    "name": "asesmen",
                    "description": "Rencana asesmen untuk mengukur ketercapaian tujuan pembelajaran"
                },
                {
                    "name": "pengayaan_remedial",
                    "description": "Program pengayaan dan remedial untuk peserta didik"
                },
                {
                    "name": "refleksi_guru",
                    "description": "Refleksi guru terhadap pembelajaran yang telah dilaksanakan"
                }
            ]
        }
