# rag_agricultural_enhanced.py - Sistema RAG mejorado
import os
import sys
import json
import requests
import hashlib
import re
import tempfile
import base64
import time
import logging
import pickle
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
from pathlib import Path
import numpy as np
from textwrap import dedent

# Configurar logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("RAG_Agricultural_Enhanced")

# Metadata requerida para el pipeline
metadata = {
    "name": "rag_agricultural_enhanced",
    "description": "Sistema RAG mejorado con Chroma, CrossEncoder y chunking semantico para documentos agricolas",
    "type": "llm",
    "version": "5.0.0"
}

# Configuracion
CHROMA_DIR = "./chroma_agricultural"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBED_MODEL = "text-embedding-3-small"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class AdvancedPDFExtractor:
    def __init__(self):
        self.available_methods = []
        self._check_available_libraries()
    
    def _check_available_libraries(self):
        libraries = [
            ("pymupdf", "fitz"),
            ("pdfplumber", "pdfplumber"),
            ("pypdf2", "PyPDF2"),
            ("pdfminer", "pdfminer.high_level"),
            ("ocr", ["pytesseract", "PIL", "pdf2image"])
        ]
        
        for lib_name, import_name in libraries:
            try:
                if isinstance(import_name, list):
                    for imp in import_name:
                        __import__(imp)
                else:
                    __import__(import_name)
                self.available_methods.append(lib_name)
                logger.info(f"Disponible: {lib_name}")
            except ImportError:
                logger.debug(f"No disponible: {lib_name}")
    
    def extract_comprehensive(self, file_path: str) -> Dict[str, Any]:
        results = {
            "text_content": "",
            "extraction_quality": 0.0,
            "best_method": None,
            "page_metadata": []
        }
        
        extraction_methods = [
            ("pymupdf", self._extract_with_pymupdf),
            ("pdfplumber", self._extract_with_pdfplumber),
            ("pdfminer", self._extract_with_pdfminer),
            ("pypdf2", self._extract_with_pypdf2)
        ]
        
        best_quality = 0
        best_content = ""
        best_method = None
        
        for method_name, method_func in extraction_methods:
            if method_name in self.available_methods:
                try:
                    logger.info(f"Intentando extraccion con {method_name}")
                    content = method_func(file_path)
                    
                    if content:
                        quality = self._evaluate_content_quality(content)
                        logger.info(f"{method_name}: {len(content)} chars, calidad: {quality:.2f}")
                        
                        if quality > best_quality:
                            best_quality = quality
                            best_content = content
                            best_method = method_name
                        
                except Exception as e:
                    logger.error(f"Error con {method_name}: {e}")
        
        if best_quality < 0.6 and "ocr" in self.available_methods:
            try:
                logger.info("Calidad baja, intentando OCR...")
                ocr_content = self._extract_with_ocr(file_path)
                if ocr_content:
                    ocr_quality = self._evaluate_content_quality(ocr_content)
                    if ocr_quality > best_quality:
                        best_quality = ocr_quality
                        best_content = ocr_content
                        best_method = "ocr"
            except Exception as e:
                logger.error(f"Error con OCR: {e}")
        
        results["text_content"] = best_content
        results["extraction_quality"] = best_quality
        results["best_method"] = best_method
        
        logger.info(f"Mejor extraccion: {best_method} (calidad: {best_quality:.2f})")
        return results
    
    def _extract_with_pymupdf(self, file_path: str) -> str:
        import fitz
        content = ""
        doc = fitz.open(file_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            if page_text.strip():
                content += f"\n--- Pagina {page_num + 1} ---\n{page_text}\n"
        
        doc.close()
        return content.strip()
    
    def _extract_with_pdfplumber(self, file_path: str) -> str:
        import pdfplumber
        content = ""
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    content += f"\n--- Pagina {page_num + 1} ---\n{text}\n"
        
        return content.strip()
    
    def _extract_with_pdfminer(self, file_path: str) -> str:
        try:
            from pdfminer.high_level import extract_text
            return extract_text(file_path)
        except ImportError:
            return ""
    
    def _extract_with_pypdf2(self, file_path: str) -> str:
        import PyPDF2
        content = ""
        
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        content += f"\n--- Pagina {page_num + 1} ---\n{page_text}\n"
                except Exception as e:
                    logger.debug(f"Error extrayendo pagina {page_num + 1}: {e}")
        
        return content.strip()
    
    def _extract_with_ocr(self, file_path: str) -> str:
        try:
            import pytesseract
            from PIL import Image
            import pdf2image
            
            pages = pdf2image.convert_from_path(file_path, dpi=300)
            content = ""
            
            for page_num, page_image in enumerate(pages):
                text = pytesseract.image_to_string(page_image, lang='spa')
                if text.strip():
                    content += f"\n--- Pagina {page_num + 1} (OCR) ---\n{text}\n"
            
            return content.strip()
        except ImportError:
            logger.error("OCR no disponible")
            return ""
    
    def _evaluate_content_quality(self, content: str) -> float:
        if not content:
            return 0.0
        
        total_chars = len(content)
        if total_chars == 0:
            return 0.0
        
        printable_chars = sum(1 for c in content if c.isprintable() or c.isspace())
        printable_ratio = printable_chars / total_chars
        
        words = content.split()
        if not words:
            return 0.0
        
        valid_words = sum(1 for word in words if len(word) > 1 and any(c.isalpha() for c in word))
        word_ratio = valid_words / len(words) if words else 0
        
        agricultural_indicators = [
            r'aplicaci[oo]n', r'dosis', r'temporada', r'cultivo', r'hectarea',
            r'tratamiento', r'plaga', r'fungicida', r'herbicida', r'insecticida',
            r'maximo', r'minimo', r'dias', r'frecuencia', r'carencia'
        ]
        
        agricultural_score = 0
        content_lower = content.lower()
        for indicator in agricultural_indicators:
            if re.search(indicator, content_lower):
                agricultural_score += 0.03
        
        agricultural_score = min(agricultural_score, 0.4)
        quality_score = (printable_ratio * 0.3) + (word_ratio * 0.4) + agricultural_score
        
        return min(quality_score, 1.0)


class SemanticChunker:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.agricultural_headers = [
            r"dosificaci[oo]n", r"dosis", r"aplicaci[oo]n",
            r"modo\s+de\s+aplicaci[oo]n", r"periodo\s+de\s+carencia",
            r"intervalo\s+entre\s+aplicaciones", r"maximo\s+de\s+aplicaciones",
            r"instrucciones\s+de\s+uso", r"precauciones"
        ]
    
    def semantic_chunk(self, text: str, source_file: str = "", page_metadata: List[Dict] = None) -> List[Dict]:
        chunks = []
        
        # Intentar split por headings agricolas
        pattern = r"(?:\n\s*)(?=(?:" + "|".join(self.agricultural_headers) + r").{0,120}:?)"
        headings = re.split(pattern, text, flags=re.IGNORECASE)
        
        if len(headings) <= 2:
            return self._character_based_chunking(text, source_file)
        
        for i, segment in enumerate(headings):
            segment = segment.strip()
            if not segment or len(segment) < 50:
                continue
            
            section_type = self._detect_section_type(segment)
            
            if len(segment) > self.chunk_size * 1.5:
                sub_chunks = self._split_large_segment(segment, section_type, i)
                chunks.extend(sub_chunks)
            else:
                chunk_data = {
                    "id": f"chunk_{hashlib.md5(f'{source_file}_{i}_{segment[:100]}'.encode()).hexdigest()[:12]}",
                    "text": segment,
                    "metadata": {
                        "source_file": source_file,
                        "chunk_index": i,
                        "section_type": section_type,
                        "semantic_chunking": True,
                        "char_count": len(segment),
                        "agricultural_relevance": self._calculate_agricultural_relevance(segment)
                    }
                }
                chunks.append(chunk_data)
        
        return chunks
    
    def _character_based_chunking(self, text: str, source_file: str) -> List[Dict]:
        chunks = []
        words = text.split()
        
        if not words:
            return chunks
        
        current_chunk = []
        current_length = 0
        chunk_index = 0
        
        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1
            
            if current_length >= self.chunk_size:
                chunk_text = ' '.join(current_chunk)
                chunk_data = {
                    "id": f"chunk_{hashlib.md5(f'{source_file}_{chunk_index}_{chunk_text[:100]}'.encode()).hexdigest()[:12]}",
                    "text": chunk_text,
                    "metadata": {
                        "source_file": source_file,
                        "chunk_index": chunk_index,
                        "section_type": "character_split",
                        "semantic_chunking": False,
                        "char_count": len(chunk_text),
                        "agricultural_relevance": self._calculate_agricultural_relevance(chunk_text)
                    }
                }
                chunks.append(chunk_data)
                
                overlap_words = int(len(current_chunk) * (self.chunk_overlap / self.chunk_size))
                current_chunk = current_chunk[-overlap_words:] if overlap_words > 0 else []
                current_length = sum(len(word) + 1 for word in current_chunk)
                chunk_index += 1
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) > 50:
                chunk_data = {
                    "id": f"chunk_{hashlib.md5(f'{source_file}_{chunk_index}_{chunk_text[:100]}'.encode()).hexdigest()[:12]}",
                    "text": chunk_text,
                    "metadata": {
                        "source_file": source_file,
                        "chunk_index": chunk_index,
                        "section_type": "character_split",
                        "semantic_chunking": False,
                        "char_count": len(chunk_text),
                        "agricultural_relevance": self._calculate_agricultural_relevance(chunk_text)
                    }
                }
                chunks.append(chunk_data)
        
        return chunks
    
    def _split_large_segment(self, segment: str, section_type: str, base_index: int) -> List[Dict]:
        chunks = []
        words = segment.split()
        current_chunk = []
        current_length = 0
        sub_index = 0
        
        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1
            
            if current_length >= self.chunk_size:
                chunk_text = ' '.join(current_chunk)
                chunk_data = {
                    "id": f"chunk_{base_index}_{sub_index}_{hashlib.md5(chunk_text[:100].encode()).hexdigest()[:8]}",
                    "text": chunk_text,
                    "metadata": {
                        "chunk_index": f"{base_index}_{sub_index}",
                        "section_type": section_type,
                        "semantic_chunking": True,
                        "is_subsection": True,
                        "char_count": len(chunk_text),
                        "agricultural_relevance": self._calculate_agricultural_relevance(chunk_text)
                    }
                }
                chunks.append(chunk_data)
                
                overlap_words = int(len(current_chunk) * 0.2)
                current_chunk = current_chunk[-overlap_words:]
                current_length = sum(len(word) + 1 for word in current_chunk)
                sub_index += 1
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_data = {
                "id": f"chunk_{base_index}_{sub_index}_{hashlib.md5(chunk_text[:100].encode()).hexdigest()[:8]}",
                "text": chunk_text,
                "metadata": {
                    "chunk_index": f"{base_index}_{sub_index}",
                    "section_type": section_type,
                    "semantic_chunking": True,
                    "is_subsection": True,
                    "char_count": len(chunk_text),
                    "agricultural_relevance": self._calculate_agricultural_relevance(chunk_text)
                }
            }
            chunks.append(chunk_data)
        
        return chunks
    
    def _detect_section_type(self, text: str) -> str:
        text_lower = text.lower()
        
        section_types = [
            (r'dosificaci[oo]n|dosis', 'dosificacion'),
            (r'aplicaci[oo]n', 'aplicacion'),
            (r'periodo\s+de\s+carencia', 'carencia'),
            (r'intervalo', 'intervalo'),
            (r'maximo.*aplicaciones', 'limite_aplicaciones'),
            (r'precauci[oo]n|advertencia', 'seguridad'),
            (r'instrucciones', 'instrucciones')
        ]
        
        for pattern, section_type in section_types:
            if re.search(pattern, text_lower):
                return section_type
        
        return 'general'
    
    def _calculate_agricultural_relevance(self, text: str) -> float:
        text_lower = text.lower()
        
        high_relevance_terms = [
            'aplicacion', 'dosis', 'tratamiento', 'cultivo', 'plaga',
            'fungicida', 'herbicida', 'insecticida', 'temporada',
            'hectarea', 'maximo', 'minimo', 'dias', 'frecuencia',
            'carencia', 'intervalo', 'litros', 'gramos'
        ]
        
        score = 0.0
        words = text_lower.split()
        total_words = len(words)
        
        if total_words == 0:
            return 0.0
        
        for term in high_relevance_terms:
            count = text_lower.count(term)
            score += (count / total_words) * 2.0
        
        return min(score, 1.0)


class ChromaVectorStore:
    def __init__(self, persist_directory: str = CHROMA_DIR, embedding_model: str = EMBED_MODEL):
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.embedding_cache = {}
        self._load_cache()
        
        try:
            import chromadb
            from chromadb.config import Settings
            
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(allow_reset=True, anonymized_telemetry=False)
            )
            
            self.collection = self.client.get_or_create_collection(
                name="agricultural_documents",
                metadata={"description": "Documentos agricolas procesados"}
            )
            
            logger.info(f"Chroma inicializado: {self.collection.count()} documentos")
            
        except ImportError:
            logger.warning("Chroma no disponible, usando vector store simple")
            self.client = None
            self.collection = None
            self._init_simple_store()
    
    def _init_simple_store(self):
        self.documents = {}
        self.embeddings = {}
        self.metadata = {}
        
        store_file = os.path.join(self.persist_directory, "simple_store.pkl")
        if os.path.exists(store_file):
            try:
                with open(store_file, "rb") as f:
                    data = pickle.load(f)
                    self.documents = data.get("documents", {})
                    self.embeddings = data.get("embeddings", {})
                    self.metadata = data.get("metadata", {})
                logger.info(f"Vector store simple cargado: {len(self.documents)} documentos")
            except Exception as e:
                logger.error(f"Error cargando store simple: {e}")
    
    def add_documents(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            return
        
        logger.info(f"Agregando {len(chunks)} chunks al vector store...")
        
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self._get_embeddings_batch(texts)
        
        if not embeddings:
            logger.error("No se pudieron generar embeddings")
            return
        
        if self.collection:
            self._add_to_chroma(chunks, embeddings)
        else:
            self._add_to_simple_store(chunks, embeddings)
    
    def _add_to_chroma(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        ids = [chunk["id"] for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        try:
            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            logger.info(f"{len(chunks)} chunks agregados a Chroma")
        except Exception as e:
            logger.error(f"Error agregando a Chroma: {e}")
    
    def _add_to_simple_store(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        for i, chunk in enumerate(chunks):
            chunk_id = chunk["id"]
            self.documents[chunk_id] = chunk["text"]
            self.embeddings[chunk_id] = embeddings[i]
            self.metadata[chunk_id] = chunk["metadata"]
        
        os.makedirs(self.persist_directory, exist_ok=True)
        store_file = os.path.join(self.persist_directory, "simple_store.pkl")
        try:
            data = {
                "documents": self.documents,
                "embeddings": self.embeddings,
                "metadata": self.metadata
            }
            with open(store_file, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"{len(chunks)} chunks agregados al store simple")
        except Exception as e:
            logger.error(f"Error guardando store simple: {e}")
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        query_embeddings = self._get_embeddings_batch([query])
        if not query_embeddings:
            return []
        
        query_embedding = query_embeddings[0]
        
        if self.collection:
            return self._search_chroma(query_embedding, top_k)
        else:
            return self._search_simple_store(query_embedding, top_k)
    
    def _search_chroma(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            chunks = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    chunks.append({
                        "id": metadata.get('id', f'unknown_{i}'),
                        "text": doc,
                        "metadata": metadata,
                        "similarity": 1 - distance,
                        "rank": i + 1
                    })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error buscando en Chroma: {e}")
            return []
    
    def _search_simple_store(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        if not self.embeddings:
            return []
        
        similarities = []
        
        for doc_id, doc_embedding in self.embeddings.items():
            dot_product = sum(a * b for a, b in zip(query_embedding, doc_embedding))
            norm_a = sum(a * a for a in query_embedding) ** 0.5
            norm_b = sum(b * b for b in doc_embedding) ** 0.5
            
            similarity = dot_product / (norm_a * norm_b) if norm_a * norm_b > 0 else 0
            similarities.append((similarity, doc_id))
        
        similarities.sort(reverse=True)
        similarities = similarities[:top_k]
        
        chunks = []
        for i, (similarity, doc_id) in enumerate(similarities):
            chunks.append({
                "id": doc_id,
                "text": self.documents[doc_id],
                "metadata": self.metadata[doc_id],
                "similarity": similarity,
                "rank": i + 1
            })
        
        return chunks
    
    def count(self) -> int:
        if self.collection:
            return self.collection.count()
        else:
            return len(self.documents)
    
    def _get_embeddings_batch(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = []
            texts_to_embed = []
            cached_embeddings = []
            
            for text in batch_texts:
                text_hash = hashlib.md5(text.encode()).hexdigest()
                if text_hash in self.embedding_cache:
                    cached_embeddings.append((len(batch_embeddings), self.embedding_cache[text_hash]))
                    batch_embeddings.append(None)
                else:
                    texts_to_embed.append((len(batch_embeddings), text, text_hash))
                    batch_embeddings.append(None)
            
            if texts_to_embed:
                new_embeddings = self._call_embedding_api([item[1] for item in texts_to_embed])
                
                for j, (idx, text, text_hash) in enumerate(texts_to_embed):
                    if j < len(new_embeddings):
                        embedding = new_embeddings[j]
                        batch_embeddings[idx] = embedding
                        self.embedding_cache[text_hash] = embedding
            
            for idx, embedding in cached_embeddings:
                batch_embeddings[idx] = embedding
            
            embeddings.extend([emb for emb in batch_embeddings if emb is not None])
            
            if i + batch_size < len(texts):
                time.sleep(0.2)
        
        self._save_cache()
        return embeddings
    
    def _call_embedding_api(self, texts: List[str]) -> List[List[float]]:
        if not self.api_key:
            logger.error("API key no disponible")
            return []
        
        try:
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "input": [text[:8000] for text in texts],
                    "model": self.embedding_model
                },
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                embeddings = [item["embedding"] for item in data["data"]]
                logger.info(f"{len(embeddings)} embeddings generados")
                return embeddings
            else:
                logger.error(f"Error en embedding API: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error llamando API de embeddings: {e}")
            return []
    
    def _load_cache(self):
        cache_file = os.path.join(self.persist_directory, "embedding_cache.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Cache cargado: {len(self.embedding_cache)} embeddings")
            except Exception as e:
                logger.error(f"Error cargando cache: {e}")
                self.embedding_cache = {}
    
    def _save_cache(self):
        os.makedirs(self.persist_directory, exist_ok=True)
        cache_file = os.path.join(self.persist_directory, "embedding_cache.pkl")
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(self.embedding_cache, f)
        except Exception as e:
            logger.error(f"Error guardando cache: {e}")


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANK_MODEL):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name, device='cpu')
            logger.info(f"CrossEncoder cargado: {self.model_name}")
        except ImportError:
            logger.warning("sentence-transformers no disponible, reranking deshabilitado")
        except Exception as e:
            logger.error(f"Error cargando CrossEncoder: {e}")
    
    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.model or not chunks:
            return chunks[:top_k]
        
        try:
            pairs = [[query, chunk["text"]] for chunk in chunks]
            scores = self.model.predict(pairs)
            
            scored_chunks = [(score, chunk) for score, chunk in zip(scores, chunks)]
            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            
            reranked = []
            for i, (score, chunk) in enumerate(scored_chunks[:top_k]):
                chunk_copy = chunk.copy()
                chunk_copy["rerank_score"] = float(score)
                chunk_copy["rerank_rank"] = i + 1
                reranked.append(chunk_copy)
            
            logger.info(f"Rerankeados {len(reranked)} chunks")
            return reranked
            
        except Exception as e:
            logger.error(f"Error en reranking: {e}")
            return chunks[:top_k]


class EnhancedRAGSystem:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        if not self.api_key:
            logger.error("OPENAI_API_KEY no configurada")
        
        self.pdf_extractor = AdvancedPDFExtractor()
        self.chunker = SemanticChunker()
        self.vector_store = ChromaVectorStore()
        self.reranker = CrossEncoderReranker()
        
        logger.info("Sistema RAG mejorado inicializado")
    
    def detect_files(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        files = []
        logger.info("Detectando archivos en request...")
        
        locations = [
            ("files", request.get("files", [])),
            ("body.files", request.get("body", {}).get("files", [])),
            ("attachments", request.get("attachments", [])),
            ("documents", request.get("documents", [])),
            ("file", [request.get("file")] if request.get("file") else [])
        ]
        
        for location_name, location_data in locations:
            if isinstance(location_data, list):
                for file_info in location_data:
                    if self._is_valid_file(file_info):
                        files.append(self._extract_file_info(file_info))
                        logger.info(f"Archivo detectado en {location_name}: {file_info.get('name', 'unknown')}")
            elif location_data:
                if self._is_valid_file(location_data):
                    files.append(self._extract_file_info(location_data))
                    logger.info(f"Archivo detectado en {location_name}: {location_data.get('name', 'unknown')}")
        
        logger.info(f"Total archivos detectados: {len(files)}")
        return files
    
    def _is_valid_file(self, file_info: Any) -> bool:
        if not isinstance(file_info, dict):
            return False

        name = file_info.get("name") or file_info.get("filename") or ""
        path = file_info.get("path") or ""
        content = file_info.get("content")
        data = file_info.get("data")
        size = file_info.get("size", 0)

        valid_text_exts = (".pdf", ".txt", ".docx", ".doc", ".md")
        has_any_payload = bool(path or data or content or size > 0)
        has_valid_ext = any((name or path).lower().endswith(ext) for ext in valid_text_exts if (name or path))

        is_valid = has_any_payload and has_valid_ext
        return is_valid
    
    def _extract_file_info(self, file_info: Dict) -> Dict[str, Any]:
        return {
            "name": file_info.get("name") or file_info.get("filename") or "unknown_file",
            "id": file_info.get("id", ""),
            "content": file_info.get("content", ""),
            "data": file_info.get("data", ""),
            "path": file_info.get("path", ""),
            "url": file_info.get("url", ""),
            "size": file_info.get("size", 0),
            "type": file_info.get("type", ""),
            "mime_type": file_info.get("mime_type", ""),
        }
    
    def process_file(self, file_data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        filename = file_data.get("name", "unknown")
        logger.info(f"Procesando archivo: {filename}")
        
        if filename.lower().endswith('.pdf'):
            extraction_result = self._process_pdf_enhanced(file_data)
            if not extraction_result:
                return None
            content = extraction_result["text_content"]
        else:
            content = self._process_text_file(file_data)
        
        if not content:
            logger.warning(f"No se pudo extraer contenido de {filename}")
            return None
        
        chunks = self.chunker.semantic_chunk(content, filename)
        
        if chunks:
            logger.info(f"{filename}: {len(chunks)} chunks creados")
            for chunk in chunks:
                chunk["metadata"]["source_file"] = filename
                chunk["metadata"]["file_size"] = file_data.get("size", 0)
                chunk["metadata"]["processing_timestamp"] = datetime.now().isoformat()
            return chunks
        else:
            logger.warning(f"No se crearon chunks para {filename}")
            return None
    
    def _process_pdf_enhanced(self, file_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        filename = file_data.get("name", "unknown")
        temp_path = None
        
        try:
            if file_data.get("path") and os.path.exists(file_data["path"]):
                temp_path = file_data["path"]
                logger.info(f"Usando archivo desde ruta: {temp_path}")
            elif file_data.get("data"):
                logger.info("Creando archivo temporal desde 'data'...")
                data = file_data["data"]
                try:
                    if isinstance(data, str) and data.startswith("data:"):
                        data = data.split(",", 1)[1]
                    if isinstance(data, str):
                        file_bytes = base64.b64decode(data)
                    else:
                        file_bytes = data
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(file_bytes)
                        temp_path = tmp.name
                    logger.info(f"Archivo temporal creado: {temp_path}")
                except Exception as e:
                    logger.error(f"Error procesando data: {e}")
                    return None

            if not temp_path:
                logger.warning(f"No se pudo obtener path temporal para {filename}")
                return None

            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                logger.error(f"Archivo temporal invalido: {temp_path}")
                return None

            return self.pdf_extractor.extract_comprehensive(temp_path)

        except Exception as e:
            logger.error(f"Error procesando PDF {filename}: {e}")
            return None
        finally:
            if temp_path and temp_path != file_data.get("path"):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logger.warning(f"No se pudo eliminar archivo temporal: {e}")
    
    def _process_text_file(self, file_data: Dict[str, Any]) -> Optional[str]:
        content = file_data.get("content", "")
        if content:
            return content
        
        path = file_data.get("path", "")
        if path and os.path.exists(path):
            encodings = ['utf-8', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    with open(path, 'r', encoding=encoding) as file:
                        return file.read()
                except UnicodeDecodeError:
                    continue
        return None
    
    def search_and_rerank(self, query: str, top_k: int = 10, rerank_k: int = 5) -> List[Dict[str, Any]]:
        results = self.vector_store.search(query, top_k)
        
        if not results:
            logger.info("No se encontraron resultados en la busqueda vectorial")
            return []
        
        logger.info(f"Encontrados {len(results)} resultados en busqueda vectorial")
        reranked_results = self.reranker.rerank(query, results, rerank_k)
        logger.info(f"Top {len(reranked_results)} resultados despues del reranking")
        
        return reranked_results
    
    def generate_grounded_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        if not self.api_key:
            return "Error: API key no configurada"
        
        if not context_chunks:
            return "No se encontraron documentos relevantes para responder la consulta."
        
        context_parts = []
        for chunk in context_chunks:
            chunk_id = chunk.get("id", "unknown")
            text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})
            source_file = metadata.get("source_file", "documento")
            
            context_parts.append(f"[CHUNK_ID:{chunk_id}] [FUENTE:{source_file}]\n{text}")
        
        context = "\n---\n".join(context_parts)
        
        system_prompt = """Eres un asistente experto en documentacion agricola y fitosanitaria.

INSTRUCCIONES CRITICAS:
1) Usa UNICAMENTE la informacion dentro de los CHUNKS provistos abajo
2) Si la pregunta no puede ser respondida con esos CHUNKS, responde: "No hay informacion suficiente en los documentos provistos"
3) SIEMPRE incluye las referencias a los CHUNK_ID que usaste para cada afirmacion (ej: [CHUNK_ID:abc123])
4) NO inventes datos, dosis, fechas o numeros
5) Si mencionas un valor especifico (dosis, dias, limites), debe estar explicitamente en el documento

Responde de forma clara y concisa, priorizando informacion sobre:
- Limites maximos de aplicacion
- Intervalos entre aplicaciones  
- Dosis y concentraciones
- Periodos de carencia
- Restricciones y precauciones"""

        user_prompt = f"""CONTEXTO:
{context}

PREGUNTA: {query}

RESPUESTA:"""

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.0,
                    "max_tokens": 800
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                logger.error(f"Error en respuesta: {response.status_code}")
                return f"Error al generar respuesta: {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error generando respuesta: {e}")
            return f"Error al procesar la consulta: {e}"


class Pipeline:
    def __init__(self):
        self.name = "RAG Agricultural Enhanced System"
        self.rag_system = None
        logger.info("Pipeline RAG Agricola mejorado inicializado")
    
    def on_startup(self):
        try:
            self.rag_system = EnhancedRAGSystem()
            logger.info("Sistema RAG mejorado inicializado correctamente")
        except Exception as e:
            logger.error(f"Error inicializando sistema RAG: {e}")
    
    def on_shutdown(self):
        logger.info("Cerrando pipeline RAG Agricola mejorado")
    
    def pipe(self, body: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not self.rag_system:
                self.rag_system = EnhancedRAGSystem()
            
            result = run_enhanced(body)
            
            return {
                "reply": result.get("reply", "Error procesando consulta"),
                "metadata": {
                    "processed_files": result.get("processed_files", 0),
                    "chunks_created": result.get("chunks_created", 0),
                    "relevant_chunks": result.get("relevant_chunks", 0),
                    "vector_store_count": result.get("vector_store_count", 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error en pipeline: {e}")
            return {
                "reply": f"Error procesando la consulta: {str(e)}",
                "metadata": {"error": str(e)}
            }

    def __call__(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return self.pipe(body)


pipeline = Pipeline()


def run_enhanced(request: Dict[str, Any]) -> Dict[str, Any]:
    try:
        rag_system = EnhancedRAGSystem()
        
        messages = request.get("messages", []) or request.get("body", {}).get("messages", [])
        
        if not messages:
            return {
                "reply": "No se encontraron mensajes en el request.",
                "processed_files": 0,
                "chunks_created": 0,
                "vector_store_count": rag_system.vector_store.count()
            }
        
        user_query = ""
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content", "").strip()
                
                if content.startswith("### Task:"):
                    import re
                    chat_history_match = re.search(r'<chat_history>.*?USER:\s*(.*?)\s*(?:ASSISTANT:|</chat_history>)', content, re.DOTALL)
                    if chat_history_match:
                        extracted_query = chat_history_match.group(1).strip()
                        if extracted_query and not extracted_query.startswith(("### Task:", "SYSTEM:", "ASSISTANT:")):
                            user_query = extracted_query
                            break
                elif content and not content.startswith(("SYSTEM:", "ASSISTANT:")):
                    user_query = content
                    break
        
        if not user_query:
            return {
                "reply": "No se encontro consulta valida del usuario.",
                "processed_files": 0,
                "chunks_created": 0,
                "vector_store_count": rag_system.vector_store.count()
            }
        
        logger.info(f"Consulta del usuario: {user_query}")
        
        files = rag_system.detect_files(request)
        total_chunks = 0
        processed_files = 0
        
        for file_data in files:
            logger.info(f"Procesando: {file_data['name']}")
            
            chunks = rag_system.process_file(file_data)
            
            if chunks:
                rag_system.vector_store.add_documents(chunks)
                total_chunks += len(chunks)
                processed_files += 1
                logger.info(f"{file_data['name']}: {len(chunks)} chunks procesados")
            else:
                logger.warning(f"No se pudieron procesar chunks para {file_data['name']}")
        
        relevant_chunks = rag_system.search_and_rerank(user_query, top_k=12, rerank_k=6)
        
        if relevant_chunks:
            logger.info(f"{len(relevant_chunks)} chunks relevantes despues del reranking")
            
            response = rag_system.generate_grounded_response(user_query, relevant_chunks)
            
            debug_info = f"\n\n--- Informacion de procesamiento ---\n"
            debug_info += f"Archivos procesados: {processed_files}\n"
            debug_info += f"Chunks creados: {total_chunks}\n"
            debug_info += f"Total chunks en base: {rag_system.vector_store.count()}\n"
            debug_info += f"Chunks relevantes encontrados: {len(relevant_chunks)}\n"
            
            if relevant_chunks:
                debug_info += "\nFuentes utilizadas (con reranking):\n"
                for i, chunk in enumerate(relevant_chunks[:3], 1):
                    chunk_id = chunk.get("id", "unknown")
                    filename = chunk.get("metadata", {}).get("source_file", "documento")
                    similarity = chunk.get("similarity", 0.0)
                    rerank_score = chunk.get("rerank_score", 0.0)
                    debug_info += f"- {i}. {filename} [{chunk_id}] - Sim: {similarity:.2f}, Rerank: {rerank_score:.2f}\n"
            
            return {
                "reply": response + debug_info,
                "processed_files": processed_files,
                "chunks_created": total_chunks,
                "relevant_chunks": len(relevant_chunks),
                "vector_store_count": rag_system.vector_store.count()
            }
        else:
            if total_chunks > 0:
                response = f"Se procesaron {processed_files} archivos con {total_chunks} chunks, pero no se encontro informacion especifica para: '{user_query}'\n\n"
                response += "Esto puede indicar que:\n"
                response += "- La informacion solicitada no esta en los documentos\n"
                response += "- La consulta necesita ser mas especifica\n"
                response += "- Los terminos de busqueda no coinciden con el contenido"
            else:
                response = f"No se encontro informacion relevante para responder: '{user_query}'"
            
            return {
                "reply": response + f"\n\nArchivos procesados: {processed_files}\nChunks creados: {total_chunks}\nTotal chunks en base: {rag_system.vector_store.count()}",
                "processed_files": processed_files,
                "chunks_created": total_chunks,
                "relevant_chunks": 0,
                "vector_store_count": rag_system.vector_store.count()
            }
    
    except Exception as e:
        logger.error(f"Error en funcion principal: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "reply": f"Error procesando la consulta: {str(e)}",
            "processed_files": 0,
            "chunks_created": 0,
            "error": str(e)
        }


def run(request: Dict[str, Any]) -> Dict[str, Any]:
    return run_enhanced(request)


def pipe(body: Dict[str, Any]) -> Dict[str, Any]:
    return pipeline.pipe(body)


if __name__ == "__main__":
    sample_request = {
        "messages": [
            {
                "role": "user", 
                "content": "¿Cual es el numero maximo de aplicaciones del producto Zoro por temporada?"
            }
        ],
        "files": []
    }
    
    result = run_enhanced(sample_request)
    print(json.dumps(result, indent=2, ensure_ascii=False))