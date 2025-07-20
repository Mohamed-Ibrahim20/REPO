import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import execute_values
import re
import uuid
import pdfplumber
import pytesseract
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, DonutProcessor, VisionEncoderDecoderModel

# Try to import unstructured with error handling for missing system dependencies
try:
    from unstructured.partition.pdf import partition_pdf
    UNSTRUCTURED_AVAILABLE = True
except ImportError as e:
    UNSTRUCTURED_AVAILABLE = False
    logging.warning(f"Unstructured library not available due to missing system dependencies: {e}")
    logging.warning("PDF processing with unstructured will be disabled. Using fallback methods.")
    partition_pdf = None

# Load environment variables
load_dotenv("config.env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSystem:
    """Full RAG system for PDF loading, chunking, embedding, and psql vector storage."""
    
    def __init__(self):
        # set variables from environment
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
        self.knowledge_base_dir = os.getenv("KNOWLEDGE_BASE_DIR", "../knowledge_base")
        
        # database configuration
        self.db_config = {
            'host': os.getenv('PGHOST', 'localhost'),
            'port': int(os.getenv('PGPORT', '5433')),
            'database': 'test',
            'user': os.getenv('PGUSER', 'postgres'),
            'password': os.getenv('PGPASSWORD', 'password')
        }
        
        # initialize components
        
        # Load embedding model
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize database
        self._init_database()
        self.ensure_index_exists()
    
    def _get_connection(self):
        """Get database connection."""
        return psycopg2.connect(**self.db_config)
    
    def _init_database(self):
        """Initialize database tables for embeddings and chat history."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as eng:
                    # Enable pgvector extension
                    eng.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    
                    # Check if table exists
                    eng.execute("""
                        SELECT COUNT(*) 
                        FROM information_schema.tables 
                        WHERE table_name = 'document_embeddings'
                    """)
                    table_exists = eng.fetchone()[0] > 0
                    
                    if table_exists:
                        # Check if table has data
                        eng.execute("SELECT COUNT(*) FROM document_embeddings")
                        
                        
                    # create embeddings table
                    eng.execute(f"""
                        CREATE TABLE IF NOT EXISTS document_embeddings (
                            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            content TEXT NOT NULL,
                            embedding vector({self.embedding_dim}) NOT NULL,
                            source_file VARCHAR(255),
                            page_number INTEGER,
                            chunk_index INTEGER,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                    
                    # Create index for vector similarity search
                    eng.execute("""
                        CREATE INDEX IF NOT EXISTS document_embeddings_vector_idx 
                        ON document_embeddings USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100);
                    """)
                    
                    # Create chat sessions table
                    eng.execute("""
                        CREATE TABLE IF NOT EXISTS chat_sessions (
                            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            title VARCHAR(255)
                        );
                    """)
                    # Drop timestamp columns if they exist
                    eng.execute("ALTER TABLE chat_sessions DROP COLUMN IF EXISTS created_at;")
                    eng.execute("ALTER TABLE chat_sessions DROP COLUMN IF EXISTS updated_at;")
                    # Create chat messages table
                    eng.execute("""
                        CREATE TABLE IF NOT EXISTS chat_messages (
                            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            session_id UUID REFERENCES chat_sessions(id) ON DELETE CASCADE,
                            role VARCHAR(255) NOT NULL CHECK (role IN ('user', 'assistant')),
                            content TEXT NOT NULL,
                            rag_context TEXT,
                            similarity_score FLOAT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                    
                    conn.commit()
            logger.info(" Database tables initialized successfully")
            
            # Validate the setup
            with self._get_connection() as conn:
                with conn.cursor() as eng:
                    eng.execute("SELECT COUNT(*) FROM document_embeddings")
                    count = eng.fetchone()[0]
                    logger.info(f" Vector database ready - {count} embeddings stored")
                    
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def load_pdf_elements(self, file_path: str) -> List:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        logger.info(f"Loading PDF with unstructured: {file_path}")
        elements = partition_pdf(file_path, strategy="hi_res")
        logger.info(f"Extracted {len(elements)} elements from PDF")
        return elements

    def group_semantic_chunks(self, elements, file_path):
        chunks = []
        current_section = {
            "title": "Untitled Section",
            "subsections": [],
            "content": [],
            "page_numbers": set()
        }

        current_subsection = None

        def save_current_section():
            if current_section["content"] or current_section["subsections"]:
                chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "title": current_section["title"],
                    "content": "\n".join(current_section["content"]).strip(),
                    "subsections": current_section["subsections"],
                    "source": os.path.basename(file_path),
                    "page_numbers": sorted(list(current_section["page_numbers"])),
                })

        def save_current_subsection():
            if current_subsection and current_subsection["content"]:
                current_section["subsections"].append({
                    "title": current_subsection["title"],
                    "content": "\n".join(current_subsection["content"]).strip(),
                    "page_numbers": sorted(list(current_subsection["page_numbers"])),
                })

        for el in elements:
            category = el.category or "Unknown"
            text = (el.text or "").strip()

            # Skip completely if it's not useful
            if category in ["Header", "Footer", "Image", "PageBreak", "UncategorizedText"]:
                continue
            if not text:
                continue

            # Clean up the text and page info
            page = el.metadata.page_number or 1

            # Skip "Contents" section — common in PDFs
            if category == "Title" and text.lower().strip() in ["contents", "table of contents"]:
                continue

            # Start new section if we find a Title
            if category == "Title":
                save_current_subsection()
                save_current_section()
                current_section = {
                    "title": text,
                    "subsections": [],
                    "content": [],
                    "page_numbers": {page}
                }
                current_subsection = None
                continue

            # Optional: Detect and handle subsections (e.g., by pattern or category)
            if category in ["SectionHeader", "Heading"]:
                save_current_subsection()
                current_subsection = {
                    "title": text,
                    "content": [],
                    "page_numbers": {page}
                }
                continue

            # Otherwise, add as regular content
            if current_subsection:
                current_subsection["content"].append(text)
                current_subsection["page_numbers"].add(page)
            else:
                current_section["content"].append(text)
                current_section["page_numbers"].add(page)

        # Save the last chunk
        save_current_subsection()
        save_current_section()

        return chunks
    def summarize_table(self, rows, max_rows=5):
        if not rows:
            return "No rows found."

        headers = list(rows[0].keys())
        lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]

        for i, row in enumerate(rows):
            if i >= max_rows:
                lines.append("...and more rows.")
                break
            line = " | ".join(row.get(h, "") for h in headers)
            lines.append(line)

        return "\n".join(lines)
    def extract_tables_and_images(self, elements, file_path):
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cpu")

        donut_processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
        donut_model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa").to("cpu")

        def generate_blip_caption(image: Image.Image) -> str:
            inputs = blip_processor(image, return_tensors="pt").to("cpu")
            out = blip_model.generate(**inputs, max_new_tokens=30)
            return blip_processor.decode(out[0], skip_special_tokens=True)

        def run_donut(image: Image.Image, prompt: str = "What does this table or image contain?") -> str:
            task_prompt = f"<s_docvqa><s_question>{prompt}</s_question><s_answer>"
            inputs = donut_processor(image, task_prompt, return_tensors="pt").to("cpu")
            outputs = donut_model.generate(**inputs, max_length=512)
            return donut_processor.batch_decode(outputs, skip_special_tokens=True)[0]

        chunks = []
        doc_name = os.path.basename(file_path)

        page_titles = {}
        for el in elements:
            if el.category == "Title" and el.metadata and el.metadata.page_number:
                pg = el.metadata.page_number
                page_titles.setdefault(pg, []).append(el.text.strip())

        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                found_table = False

                try:
                    tables = page.extract_tables()
                    print(f"[PDF Page {page_num}] Found {len(tables)} tables")

                    for table in tables:
                        if table and len(table) > 1:
                            found_table = True
                            headers = table[0]
                            rows = table[1:]

                            structured_data = []
                            for row in rows:
                                record = {headers[i]: (cell or "-") for i, cell in enumerate(row) if i < len(headers)}
                                structured_data.append(record)

                            title_texts = page_titles.get(page_num, [])
                            title = title_texts[-1] if title_texts else f"Table (Page {page_num})"

                            chunks.append({
                                "chunk_id": str(uuid.uuid4()),
                                "title": title,
                                "type": "table",
                                "content": self.summarize_table(structured_data),
                                "raw_table": structured_data,
                                "page_numbers": [page_num],
                                "source": doc_name
                            })
                except Exception as e:
                    print(f"[Warning] Table extraction failed on page {page_num}: {e}")

                if not found_table:
                    try:
                        full_image = page.to_image(resolution=300).original.convert("RGB")
                        donut_caption = run_donut(full_image, "What does the table or image on this page contain?")
                        if donut_caption.strip():
                            chunks.append({
                                "chunk_id": str(uuid.uuid4()),
                                "title": f"Visual Summary (Page {page_num})",
                                "type": "image-table",
                                "caption": donut_caption,
                                "page_numbers": [page_num],
                                "source": doc_name
                            })
                    except Exception as e:
                        print(f"[Warning] Donut failed on page {page_num}: {e}")

                try:
                    for img_dict in page.images:
                        x0, top, x1, bottom = img_dict["x0"], img_dict["top"], img_dict["x1"], img_dict["bottom"]
                        cropped_image = page.to_image(resolution=300).original.crop((x0, top, x1, bottom)).convert("RGB")

                        ocr_caption = pytesseract.image_to_string(cropped_image).strip()
                        blip_caption = None

                        if len(ocr_caption) < 10:
                            try:
                                blip_caption = generate_blip_caption(cropped_image)
                            except:
                                blip_caption = "[BLIP failed]"

                        final_caption = blip_caption or ocr_caption or "[No caption detected]"

                        chunks.append({
                            "chunk_id": str(uuid.uuid4()),
                            "title": f"Image (Page {page_num})",
                            "type": "image",
                            "caption": final_caption,
                            "ocr_caption": ocr_caption,
                            "blip_caption": blip_caption,
                            "page_numbers": [page_num],
                            "source": doc_name
                        })

                except Exception as e:
                    print(f"[Warning] Image OCR/Captioning failed on page {page_num}: {e}")

        return chunks
    def parse_pdf_combined(self, file_path):
        """Parse PDF using combined semantic and visual extraction."""
        from unstructured.partition.pdf import partition_pdf

        # Step 1: Semantic elements
        elements = partition_pdf(filename=file_path, strategy="hi_res", ocr_languages="eng")
        semantic_chunks = self.group_semantic_chunks(elements, file_path)

        # Step 2: Visual elements (tables/images)
        visual_chunks = self.extract_tables_and_images(elements, file_path)

        # Step 3: Combine and return
        return semantic_chunks + visual_chunks

    # def group_semantic_chunks(self, elements, file_path):
    #     chunks = []
    #     current_title = "Untitled Section"
    #     current_content = []
    #     current_pages = set()
    #     for el in elements:
    #         if el.category in ["Header", "Footer"]:
    #             continue
    #         text = el.text.strip()
    #         if not text:
    #             continue
    #         # Preprocess: remove >5 newlines and >10 dots in a row
    #         text = re.sub(r'\n{6,}', '\n', text)
    #         text = re.sub(r'\.{11,}', '', text)
    #         if hasattr(el.metadata, 'page_number') and el.metadata.page_number:
    #             current_pages.add(el.metadata.page_number)
    #         if el.category == "Title":
    #             if current_content:
    #                 chunks.append({
    #                     "title": current_title,
    #                     "content": "\n".join(current_content),
    #                     "source": os.path.basename(file_path),
    #                     "pages": list(current_pages)
    #                 })
    #                 current_content = []
    #                 current_pages = set()
    #             current_title = text
    #         else:
    #             current_content.append(text)
    #     if current_content:
    #         chunks.append({
    #             "title": current_title,
    #             "content": "\n".join(current_content),
    #             "source": os.path.basename(file_path),
    #             "pages": list(current_pages)
    #         })
    #     return chunks

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)
        return embeddings
    
    def store_embeddings(self, chunked_data: List[Tuple[str, dict]], embeddings: np.ndarray):
        """Store text chunks and their embeddings in the database."""
        logger.info(f"Storing {len(chunked_data)} embeddings in database")
        
        try:
            with self._get_connection() as conn:
                with conn.cursor() as eng:
                    # Clear existing embeddings for this source file
                    source_file = chunked_data[0][1]['source_file'] if chunked_data else 'unknown'
                    eng.execute("DELETE FROM document_embeddings WHERE source_file = %s", (source_file,))
                    
                    # Prepare data for batch insert
                    insert_data = []
                    for (text, metadata), embedding in zip(chunked_data, embeddings):
                        insert_data.append((
                            text,
                            embedding.tolist(),
                            metadata['source_file'],
                            metadata['page_number'],
                            metadata['chunk_index']
                        ))
                    
                    # Batch insert
                    execute_values(
                        eng,
                        """INSERT INTO document_embeddings 
                           (content, embedding, source_file, page_number, chunk_index) 
                           VALUES %s""",
                        insert_data,
                        template=None,
                        page_size=100
                    )
                    
                    conn.commit()
            logger.info("Embeddings stored successfully")
        except Exception as e:
            logger.error(f"Error storing embeddings: {e}")
            raise
    
    def build_index(self, dir_path: Optional[str] = None):
        """Complete pipeline: load PDFs from directory, chunk, embed, and store."""
        logger.info("Building RAG index...")
        
        dir_path = dir_path or self.knowledge_base_dir
        if not os.path.exists(dir_path):
            logger.error(f"Knowledge base directory not found: {dir_path}")
            return
        
        pdf_files = [f for f in os.listdir(dir_path) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {dir_path}")
            return
        
        for pdf_file in pdf_files:
            full_path = os.path.join(dir_path, pdf_file)
            try:
                # Use the new combined parsing function
                all_chunks = self.parse_pdf_combined(full_path)
                
                chunked_data = []
                for i, chunk in enumerate(all_chunks):
                    # Handle different chunk types
                    if chunk.get('type') in ['table', 'image', 'image-table']:
                        # For visual chunks, use caption or content
                        if chunk.get('type') == 'table':
                            text = f"{chunk['title']}\n{chunk['content']}"
                        else:
                            text = f"{chunk['title']}\n{chunk.get('caption', '')}"
                    else:
                        # For semantic chunks, combine title and content
                        text = f"{chunk['title']}\n{chunk['content']}"
                    
                    # Get page number from the new structure
                    page_numbers = chunk.get('page_numbers', [])
                    page_number = min(page_numbers) if page_numbers else None
                    
                    metadata = {
                        'chunk_index': i,
                        'page_number': page_number,
                        'source_file': chunk['source']
                    }
                    chunked_data.append((text, metadata))
                
                if not chunked_data:
                    logger.warning(f"No chunks created from {full_path}")
                    continue
                
                texts = [chunk[0] for chunk in chunked_data]
                embeddings = self.generate_embeddings(texts)
                
                self.store_embeddings(chunked_data, embeddings)
                logger.info(f"Successfully processed {full_path} with {len(all_chunks)} chunks")
            except Exception as e:
                logger.error(f"Error processing {full_path}: {str(e)}")
        
        logger.info("RAG index built successfully")
    
    def ensure_index_exists(self):
        """Ensure the RAG index exists, rebuild if necessary."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as eng:
                    eng.execute("SELECT COUNT(*) FROM document_embeddings")
                    count = eng.fetchone()[0]
                    
                    if count == 0:
                        logger.info("No embeddings found, rebuilding index...")
                        self.build_index()
                        return True
                    else:
                        logger.info(f"Index exists with {count} embeddings")
                        return False
        except Exception as e:
            logger.error(f"Error checking index: {e}")
            return False

    def search_similar(self, query: str, k: int = None) -> List[Tuple[str, float]]:
        """Search for similar text chunks using vector similarity."""
        k = k or int(os.getenv("RETRIEVAL_K", "4"))
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)[0]
        
        try:
            with self._get_connection() as conn:
                with conn.cursor() as eng:
                    # First check if we have any embeddings
                    eng.execute("SELECT COUNT(*) FROM document_embeddings")
                    count = eng.fetchone()[0]
                    logger.info(f"Database contains {count} embeddings for search")
                    
                    if count == 0:
                        logger.warning("No embeddings found in database!")
                        return []
                    
                    # Perform similarity search
                    eng.execute("""
                        SELECT content, (1 - (embedding <=> %s::vector)) as similarity
                        FROM document_embeddings
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                    """, (query_embedding.tolist(), query_embedding.tolist(), k))
                    
                    results = eng.fetchall()
                    logger.info(f"Found {len(results)} similar chunks for query: '{query[:50]}...'")
                    if results:
                        logger.info(f"Top similarity score: {results[0][1]:.3f}")
                    return [(content, float(similarity)) for content, similarity in results]
        except Exception as e:
            logger.error(f"Error searching similar texts: {e}")
            return []
    
    def get_rag_context(self, query: str, similarity_threshold: float = None) -> Tuple[str, float]:
        """Get RAG context for a query if similarity score is above threshold."""
        similarity_threshold = similarity_threshold or float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
        
        similar_chunks = self.search_similar(query)
        
        if not similar_chunks:
            return "", 0.0
        
        # Get the highest similarity score
        max_similarity = similar_chunks[0][1] if similar_chunks else 0.0
        
        # Return context only if above threshold
        if max_similarity >= similarity_threshold:
            context_parts = []
            for content, similarity in similar_chunks:
                context_parts.append(f"[Similarity: {similarity:.3f}]\n{content}")
            
            context = "\n\n---\n\n".join(context_parts)
            return context, max_similarity
        
        return "", max_similarity


if __name__ == "__main__":
    # Test the RAG system
    rag = RAGSystem()
    
    # Build index
    rag.build_index()
    
    # Test search
    test_query = "What is the dress code policy?"
    context, similarity = rag.get_rag_context(test_query)
    
    print(f"Query: {test_query}")
    print(f"Max Similarity: {similarity:.3f}")
    print(f"Context: {context[:200]}..." if context else "No context above threshold")
