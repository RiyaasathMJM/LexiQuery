import pytest
import pymupdf as fitz
from src.document_processor import DocumentProcessor
from langchain_core.documents import Document


@pytest.fixture
def sample_pdf_bytes():
    """Generates a multi-page PDF in memory using PyMuPDF for testing."""
    doc = fitz.open()

    # Page 1
    page1 = doc.new_page()
    page1.insert_text(
        (50, 50),
        "LexiQuery Document Processor Test - Page 1.\n"
        "This is a privacy-first local RAG system using LangChain and Ollama.\n"
        "ChromaDB stores vector embeddings locally without sending data to third parties.",
    )

    # Page 2
    page2 = doc.new_page()
    page2.insert_text(
        (50, 50),
        "LexiQuery Document Processor Test - Page 2.\n"
        "Recursive text chunking splits documents while preserving page metadata.\n"
        "This ensures grounded retrieval and hallucination-resistant Q&A.",
    )

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_load_pdf(sample_pdf_bytes):
    processor = DocumentProcessor()
    docs = processor.load_pdf(sample_pdf_bytes, filename="test_sample.pdf")

    assert len(docs) == 2
    assert docs[0].metadata["page"] == 1
    assert docs[0].metadata["source"] == "test_sample.pdf"
    assert "LexiQuery" in docs[0].page_content
    assert docs[1].metadata["page"] == 2
    assert "hallucination-resistant" in docs[1].page_content


def test_chunk_documents(sample_pdf_bytes):
    processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
    docs = processor.load_pdf(sample_pdf_bytes, filename="test_sample.pdf")
    chunks = processor.chunk_documents(docs)

    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, Document)
        assert "chunk_id" in chunk.metadata
        assert "chunk_index" in chunk.metadata
        assert "source" in chunk.metadata
        assert chunk.metadata["source"] == "test_sample.pdf"
        assert len(chunk.page_content) <= 150  # Allowance for boundary splits


def test_get_document_stats(sample_pdf_bytes):
    processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
    docs = processor.load_pdf(sample_pdf_bytes, filename="test_sample.pdf")
    chunks = processor.chunk_documents(docs)

    stats = processor.get_document_stats(chunks)
    assert stats["total_chunks"] == len(chunks)
    assert stats["total_pages"] == 2
    assert "test_sample.pdf" in stats["sources"]
    assert stats["avg_chunk_size"] > 0


def test_empty_documents_handling():
    processor = DocumentProcessor()
    chunks = processor.chunk_documents([])
    assert chunks == []

    stats = processor.get_document_stats([])
    assert stats["total_chunks"] == 0
    assert stats["sources"] == []
