#!/usr/bin/env python3
"""
Test script for PDF Vector Database System
Tests all functionality: ingestion, search, list, delete, stats
"""
import os
import sys
import tempfile
from pathlib import Path

# Try to import reportlab for PDF generation
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Import the PDF vector store
try:
    from stock_training_db import PDFVectorStore, PDF_SUPPORT_AVAILABLE
except ImportError as e:
    print(f"Error importing PDFVectorStore: {e}")
    sys.exit(1)


def create_test_pdf(output_path: str) -> str:
    """Create a test PDF with financial content.

    Args:
        output_path: Path where PDF should be created

    Returns:
        Path to created PDF
    """
    if not REPORTLAB_AVAILABLE:
        print("⚠️  reportlab not available, using alternative method")
        # For testing without reportlab, we'll just note it
        # In real usage, user would provide their own PDFs
        return None

    # Create PDF with financial content
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter

    # Page 1: Introduction
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1*inch, height - 1*inch, "Financial Analysis Report Q4 2025")

    c.setFont("Helvetica", 12)
    y = height - 1.5*inch

    intro_text = [
        "Executive Summary",
        "",
        "This report provides a comprehensive analysis of our financial performance",
        "for Q4 2025. Key highlights include:",
        "",
        "• Revenue growth of 23% year-over-year",
        "• Operating margin improvement to 34.5%",
        "• Strong cash flow generation of $2.3B",
        "• Successful product launches in emerging markets",
        "",
        "The technology sector continues to show robust growth, with particular",
        "strength in cloud computing and artificial intelligence segments.",
    ]

    for line in intro_text:
        c.drawString(1*inch, y, line)
        y -= 0.25*inch

    c.showPage()

    # Page 2: Revenue Analysis
    c.setFont("Helvetica-Bold", 14)
    c.drawString(1*inch, height - 1*inch, "Revenue Analysis")

    c.setFont("Helvetica", 12)
    y = height - 1.5*inch

    revenue_text = [
        "Quarterly Revenue Breakdown:",
        "",
        "Q4 2025 Revenue: $5.8 billion (up 23% YoY)",
        "Q3 2025 Revenue: $5.2 billion",
        "Q2 2025 Revenue: $4.9 billion",
        "Q1 2025 Revenue: $4.7 billion",
        "",
        "Revenue by Segment:",
        "• Cloud Services: $2.1B (36% of total)",
        "• Enterprise Software: $1.8B (31% of total)",
        "• Consumer Products: $1.2B (21% of total)",
        "• Professional Services: $0.7B (12% of total)",
        "",
        "The cloud services segment showed exceptional growth at 45% YoY,",
        "driven by increased adoption of our AI and machine learning platforms.",
    ]

    for line in revenue_text:
        c.drawString(1*inch, y, line)
        y -= 0.25*inch

    c.showPage()

    # Page 3: Cost Analysis
    c.setFont("Helvetica-Bold", 14)
    c.drawString(1*inch, height - 1*inch, "Cost Structure and Profitability")

    c.setFont("Helvetica", 12)
    y = height - 1.5*inch

    cost_text = [
        "Operating Expenses Analysis:",
        "",
        "Cost of Revenue: $2.4B (41% of revenue)",
        "Research & Development: $1.1B (19% of revenue)",
        "Sales & Marketing: $0.8B (14% of revenue)",
        "General & Administrative: $0.5B (9% of revenue)",
        "",
        "Operating Income: $2.0B (34.5% margin)",
        "Net Income: $1.6B (27.6% margin)",
        "",
        "Key Metrics:",
        "• Gross Margin: 59%",
        "• EBITDA: $2.2B",
        "• Free Cash Flow: $1.8B",
        "• Return on Equity: 28%",
        "",
        "We maintained strong profitability while investing heavily in R&D",
        "for next-generation AI capabilities and infrastructure expansion.",
    ]

    for line in cost_text:
        c.drawString(1*inch, y, line)
        y -= 0.25*inch

    c.showPage()

    # Page 4: Risk Factors
    c.setFont("Helvetica-Bold", 14)
    c.drawString(1*inch, height - 1*inch, "Risk Factors and Outlook")

    c.setFont("Helvetica", 12)
    y = height - 1.5*inch

    risk_text = [
        "Key Risk Factors:",
        "",
        "1. Market Competition: Increasing competition in cloud services",
        "   may pressure pricing and market share.",
        "",
        "2. Regulatory Environment: Evolving AI regulations could impact",
        "   product development timelines and costs.",
        "",
        "3. Cybersecurity: Growing threats require continuous investment",
        "   in security infrastructure and personnel.",
        "",
        "4. Economic Conditions: Potential recession could reduce enterprise",
        "   IT spending and delay purchasing decisions.",
        "",
        "Outlook for 2026:",
        "• Expected revenue growth: 18-22%",
        "• Target operating margin: 35-37%",
        "• Capital expenditures: $3-3.5B",
        "• Focus on AI/ML product expansion",
    ]

    for line in risk_text:
        c.drawString(1*inch, y, line)
        y -= 0.25*inch

    c.save()
    print(f"✅ Created test PDF: {output_path}")
    return output_path


def test_pdf_dependencies():
    """Test if required dependencies are installed."""
    print("\n" + "="*70)
    print("Testing PDF System Dependencies")
    print("="*70)

    if not PDF_SUPPORT_AVAILABLE:
        print("❌ PDF support not available!")
        print("\nInstall required packages:")
        print("  pip install chromadb pypdf langchain-text-splitters")
        return False

    print("✅ chromadb installed")
    print("✅ pypdf installed")
    print("✅ langchain-text-splitters installed")

    if REPORTLAB_AVAILABLE:
        print("✅ reportlab installed (for test PDF generation)")
    else:
        print("⚠️  reportlab not installed (optional, for test PDF generation)")
        print("   Install with: pip install reportlab")

    return True


def test_pdf_ingestion(store, pdf_path):
    """Test PDF ingestion."""
    print("\n" + "="*70)
    print("Testing PDF Ingestion")
    print("="*70)

    try:
        result = store.ingest_pdf(pdf_path)
        print(f"\n✅ Ingestion successful!")
        print(f"   PDF ID: {result['pdf_id']}")
        print(f"   Filename: {result['filename']}")
        print(f"   Pages: {result['num_pages']}")
        print(f"   Chunks: {result['num_chunks']}")
        return result['pdf_id']
    except Exception as e:
        print(f"\n❌ Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_list_pdfs(store):
    """Test listing PDFs."""
    print("\n" + "="*70)
    print("Testing List PDFs")
    print("="*70)

    try:
        pdfs = store.list_pdfs()
        print(f"\n✅ Found {len(pdfs)} PDF(s)")

        if pdfs:
            print(f"\n{'ID':<6} {'Filename':<30} {'Pages':<8} {'Chunks':<8}")
            print("-" * 60)
            for pdf in pdfs:
                filename = pdf['filename'][:27] + "..." if len(pdf['filename']) > 30 else pdf['filename']
                print(f"{pdf['pdf_id']:<6} {filename:<30} {pdf['num_pages']:<8} {pdf['num_chunks']:<8}")

        return True
    except Exception as e:
        print(f"\n❌ List failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_search(store):
    """Test vector search."""
    print("\n" + "="*70)
    print("Testing Vector Search")
    print("="*70)

    test_queries = [
        "revenue growth and financial performance",
        "operating expenses and cost analysis",
        "risk factors and market competition",
        "cloud services and AI platforms",
    ]

    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")

        try:
            results = store.search(query, n_results=3)

            if not results:
                print("   📭 No results found")
                continue

            print(f"   ✅ Found {len(results)} result(s)\n")

            for i, result in enumerate(results, 1):
                meta = result['metadata']
                text_preview = result['text'][:150].replace('\n', ' ')
                score = 1 - result['distance'] if result['distance'] else 1.0

                print(f"   Result {i}:")
                print(f"   • File: {meta['filename']}")
                print(f"   • Page: {meta['page_number']}")
                print(f"   • Score: {score:.4f}")
                print(f"   • Text: {text_preview}...")
                print()

        except Exception as e:
            print(f"   ❌ Search failed: {e}")
            return False

    return True


def test_stats(store):
    """Test statistics."""
    print("\n" + "="*70)
    print("Testing Statistics")
    print("="*70)

    try:
        stats = store.get_stats()
        print(f"\n✅ Statistics retrieved:")
        print(f"   Total PDFs: {stats['num_pdfs']}")
        print(f"   Total Chunks: {stats['num_chunks']}")
        print(f"   Total Size: {stats['total_size_mb']:.2f} MB")
        print(f"   ChromaDB Count: {stats['chroma_collection_count']}")
        return True
    except Exception as e:
        print(f"\n❌ Stats failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_delete(store, pdf_id):
    """Test PDF deletion."""
    print("\n" + "="*70)
    print("Testing PDF Deletion")
    print("="*70)

    try:
        store.delete_pdf(pdf_id)
        print(f"\n✅ PDF {pdf_id} deleted successfully")

        # Verify deletion
        pdfs = store.list_pdfs()
        if not any(pdf['pdf_id'] == pdf_id for pdf in pdfs):
            print("✅ Verified: PDF no longer in database")
            return True
        else:
            print("❌ Error: PDF still in database after deletion")
            return False
    except Exception as e:
        print(f"\n❌ Delete failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("PDF Vector Database System - Comprehensive Test Suite")
    print("="*70)

    # Test dependencies
    if not test_pdf_dependencies():
        print("\n❌ Dependency check failed. Please install required packages.")
        return 1

    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize store in temp directory
        db_path = os.path.join(tmpdir, "test_pdf_metadata.db")
        chroma_path = os.path.join(tmpdir, "test_chroma_db")

        print(f"\n📁 Test database path: {db_path}")
        print(f"📁 Test ChromaDB path: {chroma_path}")

        try:
            store = PDFVectorStore(db_path=db_path, chroma_path=chroma_path)
            print("✅ PDFVectorStore initialized")
        except Exception as e:
            print(f"❌ Failed to initialize PDFVectorStore: {e}")
            return 1

        # Create test PDF
        pdf_path = None
        if REPORTLAB_AVAILABLE:
            pdf_path = os.path.join(tmpdir, "test_financial_report.pdf")
            pdf_path = create_test_pdf(pdf_path)

        if not pdf_path:
            print("\n⚠️  Cannot create test PDF without reportlab")
            print("   Install with: pip install reportlab")
            print("   Or provide your own PDF for testing")
            return 1

        # Run tests
        tests_passed = 0
        tests_total = 5

        # Test 1: Ingestion
        pdf_id = test_pdf_ingestion(store, pdf_path)
        if pdf_id:
            tests_passed += 1

        # Test 2: List
        if test_list_pdfs(store):
            tests_passed += 1

        # Test 3: Search
        if test_search(store):
            tests_passed += 1

        # Test 4: Stats
        if test_stats(store):
            tests_passed += 1

        # Test 5: Delete
        if pdf_id and test_delete(store, pdf_id):
            tests_passed += 1

        # Final summary
        print("\n" + "="*70)
        print("Test Results Summary")
        print("="*70)
        print(f"\nTests passed: {tests_passed}/{tests_total}")

        if tests_passed == tests_total:
            print("\n🎉 All tests passed! PDF Vector Database system is working correctly.")
            return 0
        else:
            print(f"\n⚠️  {tests_total - tests_passed} test(s) failed.")
            return 1


if __name__ == "__main__":
    sys.exit(main())
