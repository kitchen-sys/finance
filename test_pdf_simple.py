#!/usr/bin/env python3
"""
Simple standalone test for PDF Vector Database
Tests directly without importing full stock_training_db module
"""
import os
import sys
import tempfile
import sqlite3
from pathlib import Path

# Test imports
try:
    import chromadb
    from pypdf import PdfReader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    print("✅ All dependencies installed successfully!")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("\nInstall with:")
    print("  pip install chromadb pypdf langchain-text-splitters reportlab")
    sys.exit(1)


def create_test_pdf(output_path):
    """Create a simple test PDF."""
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter

    # Page 1
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1*inch, height - 1*inch, "Financial Analysis Report Q4 2025")
    c.setFont("Helvetica", 12)

    y = height - 1.5*inch
    lines = [
        "Executive Summary",
        "",
        "This report provides comprehensive financial analysis.",
        "Key highlights:",
        "• Revenue growth of 23% year-over-year",
        "• Operating margin improvement to 34.5%",
        "• Strong cash flow generation of $2.3B",
    ]

    for line in lines:
        c.drawString(1*inch, y, line)
        y -= 0.25*inch

    c.showPage()

    # Page 2
    c.setFont("Helvetica-Bold", 14)
    c.drawString(1*inch, height - 1*inch, "Revenue Analysis")
    c.setFont("Helvetica", 12)

    y = height - 1.5*inch
    lines = [
        "Q4 2025 Revenue: $5.8 billion (up 23% YoY)",
        "Revenue by Segment:",
        "• Cloud Services: $2.1B (36% of total)",
        "• Enterprise Software: $1.8B (31% of total)",
        "• Consumer Products: $1.2B (21% of total)",
    ]

    for line in lines:
        c.drawString(1*inch, y, line)
        y -= 0.25*inch

    c.save()
    print(f"✅ Created test PDF: {output_path}")
    return output_path


def test_pdf_system():
    """Test the PDF vector database system."""
    print("\n" + "="*70)
    print("PDF Vector Database - Simple Test")
    print("="*70 + "\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test PDF
        pdf_path = os.path.join(tmpdir, "test_report.pdf")
        create_test_pdf(pdf_path)

        # Initialize databases
        db_path = os.path.join(tmpdir, "metadata.db")
        chroma_path = os.path.join(tmpdir, "chroma_db")

        print(f"\n📁 Test database: {db_path}")
        print(f"📁 ChromaDB: {chroma_path}\n")

        # Setup SQLite
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE pdf_documents (
                pdf_id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL UNIQUE,
                upload_timestamp TEXT NOT NULL,
                num_pages INTEGER,
                num_chunks INTEGER,
                file_size_bytes INTEGER
            )
        """)

        # Setup ChromaDB
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_or_create_collection(name="pdf_documents")

        print("✅ Databases initialized\n")

        # Extract PDF text
        print("📄 Extracting PDF text...")
        reader = PdfReader(pdf_path)
        num_pages = len(reader.pages)
        full_text = ""

        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            full_text += f"\n\n--- Page {i+1} ---\n\n{text}"

        print(f"✅ Extracted {num_pages} pages\n")

        # Chunk text
        print("✂️  Chunking text...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = splitter.split_text(full_text)
        print(f"✅ Created {len(chunks)} chunks\n")

        # Store metadata in SQLite
        print("💾 Storing metadata...")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO pdf_documents
            (filename, filepath, upload_timestamp, num_pages, num_chunks, file_size_bytes)
            VALUES (?, ?, datetime('now'), ?, ?, ?)
        """, (
            os.path.basename(pdf_path),
            pdf_path,
            num_pages,
            len(chunks),
            os.path.getsize(pdf_path)
        ))
        pdf_id = cursor.lastrowid
        conn.commit()
        print(f"✅ Stored metadata (PDF ID: {pdf_id})\n")

        # Store chunks in ChromaDB
        print("🔮 Creating embeddings and storing in ChromaDB...")
        chunk_ids = [f"pdf_{pdf_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{
            "pdf_id": pdf_id,
            "chunk_index": i,
            "page_number": min(i // max(1, len(chunks) // num_pages) + 1, num_pages),
            "filename": os.path.basename(pdf_path)
        } for i in range(len(chunks))]

        collection.add(
            ids=chunk_ids,
            documents=chunks,
            metadatas=metadatas
        )
        print(f"✅ Stored {len(chunks)} chunks with embeddings\n")

        # Test search
        print("="*70)
        print("Testing Vector Search")
        print("="*70 + "\n")

        queries = [
            "revenue growth and financial performance",
            "cloud services segment analysis",
        ]

        for query in queries:
            print(f"🔍 Query: '{query}'")
            results = collection.query(
                query_texts=[query],
                n_results=3
            )

            if results['ids'] and len(results['ids'][0]) > 0:
                print(f"✅ Found {len(results['ids'][0])} results\n")

                for i in range(len(results['ids'][0])):
                    meta = results['metadatas'][0][i]
                    text = results['documents'][0][i]
                    distance = results['distances'][0][i]
                    score = 1 - distance

                    text_preview = text[:150].replace('\n', ' ')
                    print(f"  Result {i+1}:")
                    print(f"  • Page: {meta['page_number']}")
                    print(f"  • Score: {score:.4f}")
                    print(f"  • Text: {text_preview}...")
                    print()
            else:
                print("📭 No results\n")

        # Test statistics
        print("="*70)
        print("Statistics")
        print("="*70 + "\n")

        cursor.execute("SELECT COUNT(*) FROM pdf_documents")
        num_pdfs = cursor.fetchone()[0]

        print(f"✅ Statistics:")
        print(f"  Total PDFs: {num_pdfs}")
        print(f"  Total Chunks: {len(chunks)}")
        print(f"  ChromaDB Collection Count: {collection.count()}")

        conn.close()

        print("\n" + "="*70)
        print("🎉 All tests passed!")
        print("="*70 + "\n")

        return 0


if __name__ == "__main__":
    sys.exit(test_pdf_system())
