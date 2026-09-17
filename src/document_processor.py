import os
import pymupdf as fitz  # PyMuPDF
from typing import List, Dict, Any, Union, BinaryIO
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_SEPARATORS


class DocumentProcessor:
    """
    Handles PDF loading, text extraction, page-level metadata tracking,
    and recursive document chunking for LexiQuery RAG pipeline.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=DEFAULT_SEPARATORS,
            length_function=len,
            is_separator_regex=False,
        )

    def load_pdf(
        self, file_source: Union[str, bytes, BinaryIO], filename: str = "document.pdf"
    ) -> List[Document]:
        """
        Loads a PDF from a file path, raw bytes, or file-like object using PyMuPDF (fitz).
        Extracts page-by-page text with metadata.
        """
        documents: List[Document] = []

        try:
            if isinstance(file_source, str):
                if not os.path.exists(file_source):
                    raise FileNotFoundError(f"PDF file not found: {file_source}")
                doc = fitz.open(file_source)
                filename = os.path.basename(file_source)
            elif isinstance(file_source, bytes):
                doc = fitz.open(stream=file_source, filetype="pdf")
            elif hasattr(file_source, "read"):
                stream_bytes = file_source.read()
                doc = fitz.open(stream=stream_bytes, filetype="pdf")
            else:
                raise ValueError("Unsupported file_source type provided.")

            total_pages = len(doc)
            for page_num in range(total_pages):
                page = doc.load_page(page_num)
                text = page.get_text("text")

                # Clean up excess null characters or control codes
                clean_text = text.replace("\x00", "").strip()
                if clean_text:
                    metadata = {
                        "source": filename,
                        "page": page_num + 1,  # 1-indexed for readability
                        "total_pages": total_pages,
                    }
                    documents.append(Document(page_content=clean_text, metadata=metadata))

            doc.close()
        except Exception as e:
            raise RuntimeError(f"Failed to process PDF '{filename}': {str(e)}") from e

        return documents

    def chunk_documents(
        self,
        documents: List[Document],
        chunk_size: int = None,
        chunk_overlap: int = None,
    ) -> List[Document]:
        """
        Splits a list of Documents into chunks preserving metadata and generating chunk IDs.
        """
        if not documents:
            return []

        c_size = chunk_size if chunk_size is not None else self.chunk_size
        c_overlap = chunk_overlap if chunk_overlap is not None else self.chunk_overlap

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=c_size,
            chunk_overlap=c_overlap,
            separators=DEFAULT_SEPARATORS,
            length_function=len,
            is_separator_regex=False,
        )

        raw_chunks = splitter.split_documents(documents)
        processed_chunks: List[Document] = []

        for idx, chunk in enumerate(raw_chunks):
            # Enrich metadata with unique chunk identifier and position
            metadata = dict(chunk.metadata)
            metadata["chunk_index"] = idx
            metadata["chunk_id"] = f"{metadata.get('source', 'doc')}_p{metadata.get('page', 1)}_c{idx}"
            metadata["character_count"] = len(chunk.page_content)
            
            processed_chunks.append(
                Document(page_content=chunk.page_content, metadata=metadata)
            )

        return processed_chunks

    @staticmethod
    def get_document_stats(chunks: List[Document]) -> Dict[str, Any]:
        """
        Computes summary statistics for a set of processed document chunks.
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_pages": 0,
                "avg_chunk_size": 0,
                "sources": [],
            }

        sources = sorted(list(set(c.metadata.get("source", "unknown") for c in chunks)))
        pages = set((c.metadata.get("source"), c.metadata.get("page")) for c in chunks)
        total_chars = sum(len(c.page_content) for c in chunks)

        return {
            "total_chunks": len(chunks),
            "total_pages": len(pages),
            "avg_chunk_size": round(total_chars / len(chunks), 1) if chunks else 0,
            "sources": sources,
        }
