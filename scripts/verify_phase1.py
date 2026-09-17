import os
import sys
import pymupdf as fitz

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.document_processor import DocumentProcessor
from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


def generate_demo_pdf(output_path: str):
    """Creates a sample PDF file for verification."""
    doc = fitz.open()

    page1 = doc.new_page()
    page1.insert_text(
        (50, 50),
        "LexiQuery Architecture Specification - Chapter 1: Introduction\n\n"
        "LexiQuery is a privacy-first, local-first RAG (Retrieval-Augmented Generation) document assistant.\n"
        "It enables natural language Q&A over complex PDFs using LangChain, Ollama (Llama 3 8B), and ChromaDB.\n"
        "No document content leaves the user local machine during indexing, embedding, or inference.",
    )

    page2 = doc.new_page()
    page2.insert_text(
        (50, 50),
        "LexiQuery Architecture Specification - Chapter 2: Technical Design\n\n"
        "1. Document Ingestion: PyMuPDF extracts text page by page.\n"
        "2. Text Chunking: RecursiveCharacterTextSplitter divides text into chunks of 1000 characters with 200 overlap.\n"
        "3. Vector Storage: Nomic Embed Text generates 768-dim embeddings stored in ChromaDB vector collection.\n"
        "4. Prompt Pipeline: Hallucination-resistant grounded prompt enforces strict context fidelity.",
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc.save(output_path)
    doc.close()
    print(f"Generated demo PDF at: {output_path}")


def main():
    print("=" * 60)
    print("LexiQuery Phase 1 Verification: Document Processing Engine")
    print("=" * 60)

    demo_pdf_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../data/lexiquery_spec.pdf")
    )
    generate_demo_pdf(demo_pdf_path)

    processor = DocumentProcessor(
        chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP
    )

    print("\n1. Loading PDF...")
    documents = processor.load_pdf(demo_pdf_path)
    print(f"Loaded {len(documents)} page documents.")
    for doc in documents:
        print(f"  - Page {doc.metadata['page']}/{doc.metadata['total_pages']} ({len(doc.page_content)} characters)")

    print("\n2. Chunking Documents...")
    chunks = processor.chunk_documents(documents)
    print(f"Generated {len(chunks)} chunks.")

    print("\n3. Inspecting Chunks & Metadata:")
    for idx, chunk in enumerate(chunks):
        print(f"\n--- Chunk #{idx + 1} [{chunk.metadata['chunk_id']}] ---")
        print(f"Source: {chunk.metadata['source']} | Page: {chunk.metadata['page']} | Chars: {chunk.metadata['character_count']}")
        print(f"Content snippet: {chunk.page_content[:120]}...")

    print("\n4. Summary Statistics:")
    stats = processor.get_document_stats(chunks)
    print(f"  Total Chunks: {stats['total_chunks']}")
    print(f"  Total Pages: {stats['total_pages']}")
    print(f"  Avg Chunk Size: {stats['avg_chunk_size']} chars")
    print(f"  Sources: {stats['sources']}")

    print("\n[SUCCESS] Phase 1 Document Processing Verification Completed Successfully!")


if __name__ == "__main__":
    main()
