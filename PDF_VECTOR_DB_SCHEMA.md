# PDF Vector Database Schema Documentation

## Overview

The PDF Vector Database system provides intelligent document ingestion, chunking, and semantic search capabilities using a hybrid storage approach:

- **SQLite**: Stores PDF metadata and chunk references
- **ChromaDB**: Stores vector embeddings for semantic similarity search

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    PDF Document                         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   Text Extraction     │
         │   (pypdf)             │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   Text Chunking       │
         │   (1000 chars/200     │
         │    overlap)           │
         └───────────┬───────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌────────────────┐      ┌────────────────┐
│  SQLite DB     │      │  ChromaDB      │
│  (Metadata)    │      │  (Vectors)     │
└────────────────┘      └────────────────┘
```

## Database Schemas

### SQLite Schema

#### Table: `pdf_documents`

Stores metadata about ingested PDF documents.

| Column            | Type    | Constraints      | Description                           |
|-------------------|---------|------------------|---------------------------------------|
| pdf_id            | INTEGER | PRIMARY KEY      | Auto-incrementing unique identifier   |
| filename          | TEXT    | NOT NULL         | Name of the PDF file                  |
| filepath          | TEXT    | NOT NULL, UNIQUE | Full absolute path to PDF file        |
| upload_timestamp  | TEXT    | NOT NULL         | ISO 8601 timestamp of ingestion       |
| num_pages         | INTEGER |                  | Total number of pages in PDF          |
| num_chunks        | INTEGER |                  | Total number of text chunks created   |
| file_size_bytes   | INTEGER |                  | Size of PDF file in bytes             |
| title             | TEXT    |                  | PDF document title (from metadata)    |
| author            | TEXT    |                  | PDF document author (from metadata)   |
| subject           | TEXT    |                  | PDF document subject (from metadata)  |

**Indexes:**
- Primary key index on `pdf_id`
- Unique index on `filepath`

**Example:**
```sql
INSERT INTO pdf_documents VALUES (
    1,
    'financial_report.pdf',
    '/home/user/docs/financial_report.pdf',
    '2025-12-06T10:30:00',
    45,
    89,
    2457600,
    'Q4 Financial Report 2025',
    'Finance Team',
    'Quarterly earnings and analysis'
);
```

#### Table: `pdf_chunks`

Stores individual text chunks and references to parent PDF documents.

| Column       | Type    | Constraints                            | Description                              |
|--------------|---------|----------------------------------------|------------------------------------------|
| chunk_id     | TEXT    | PRIMARY KEY                            | Unique ID: `pdf_{pdf_id}_chunk_{index}` |
| pdf_id       | INTEGER | NOT NULL, FOREIGN KEY                  | Reference to `pdf_documents.pdf_id`      |
| chunk_index  | INTEGER | NOT NULL                               | Sequential index within document (0-N)   |
| page_number  | INTEGER |                                        | Approximate page number for chunk        |
| chunk_text   | TEXT    | NOT NULL                               | Full text content of the chunk           |
| chunk_length | INTEGER |                                        | Length of chunk text in characters       |

**Foreign Keys:**
- `pdf_id` → `pdf_documents(pdf_id)` with CASCADE DELETE

**Example:**
```sql
INSERT INTO pdf_chunks VALUES (
    'pdf_1_chunk_0',
    1,
    0,
    1,
    'This is the beginning of the financial report...',
    987
);
```

### ChromaDB Schema

#### Collection: `pdf_documents`

Stores vector embeddings for semantic search.

**Metadata per Document (Chunk):**
```json
{
    "pdf_id": 1,
    "chunk_index": 0,
    "page_number": 1,
    "filename": "financial_report.pdf",
    "title": "Q4 Financial Report 2025"
}
```

**Document ID Format:** `pdf_{pdf_id}_chunk_{index}`

**Embedding Model:** ChromaDB default (sentence-transformers/all-MiniLM-L6-v2)

**Distance Metric:** L2 (Euclidean distance)

## Text Chunking Strategy

### Parameters

| Parameter         | Value                          | Purpose                                    |
|-------------------|--------------------------------|--------------------------------------------|
| chunk_size        | 1000 characters                | Optimal balance of context and specificity |
| chunk_overlap     | 200 characters                 | Preserves context across chunk boundaries  |
| length_function   | `len()`                        | Character count                            |
| separators        | `["\n\n", "\n", " ", ""]`      | Split on paragraphs, lines, words, chars   |

### Algorithm: RecursiveCharacterTextSplitter

1. Try to split on paragraph breaks (`\n\n`)
2. If chunks too large, split on line breaks (`\n`)
3. If still too large, split on spaces (` `)
4. If still too large, split by character

This ensures chunks break at natural boundaries when possible.

### Example Chunking

**Original Text (1500 chars):**
```
Paragraph 1: Introduction to the topic...
[300 characters]

Paragraph 2: Detailed analysis of metrics...
[600 characters]

Paragraph 3: Conclusion and recommendations...
[600 characters]
```

**Resulting Chunks:**
- Chunk 0: Paragraph 1 + start of Paragraph 2 (1000 chars)
- Chunk 1: End of Paragraph 2 (200 char overlap) + Paragraph 3 (1000 chars)

## Vector Search

### Query Flow

```
User Query
    │
    ▼
Text → Embedding (via ChromaDB)
    │
    ▼
Vector Similarity Search (L2 distance)
    │
    ▼
Top N Results (chunks ranked by relevance)
    │
    ▼
Return with Metadata & Text
```

### Search Parameters

| Parameter  | Default | Description                        |
|------------|---------|----------------------------------- |
| n_results  | 5       | Number of chunks to return         |
| query_text | (user)  | Natural language search query      |

### Similarity Score

- Raw distance from ChromaDB (L2 distance)
- Lower distance = higher similarity
- Displayed as: `1 - distance` for intuitive scoring

## API Usage Examples

### Initialize Store

```python
from stock_training_db import PDFVectorStore

store = PDFVectorStore(
    db_path="./pdf_metadata.db",
    chroma_path="./chroma_db"
)
```

### Ingest PDF

```python
result = store.ingest_pdf("/path/to/document.pdf")
print(f"Ingested {result['num_chunks']} chunks from {result['filename']}")
```

### Search

```python
results = store.search("financial metrics analysis", n_results=5)
for result in results:
    print(f"File: {result['metadata']['filename']}")
    print(f"Page: {result['metadata']['page_number']}")
    print(f"Text: {result['text'][:200]}...")
    print(f"Score: {1 - result['distance']:.4f}\n")
```

### List PDFs

```python
pdfs = store.list_pdfs()
for pdf in pdfs:
    print(f"{pdf['pdf_id']}: {pdf['filename']} ({pdf['num_pages']} pages)")
```

### Delete PDF

```python
store.delete_pdf(pdf_id=1)
```

### Get Statistics

```python
stats = store.get_stats()
print(f"Total PDFs: {stats['num_pdfs']}")
print(f"Total Chunks: {stats['num_chunks']}")
print(f"Total Size: {stats['total_size_mb']:.2f} MB")
```

## CLI Interface (Option 71)

### Available Commands

| Command           | Description                              | Example                      |
|-------------------|------------------------------------------|------------------------------|
| `ingest <path>`   | Ingest a PDF file                        | `ingest ~/docs/report.pdf`   |
| `list`            | List all ingested PDFs                   | `list`                       |
| `search <query>`  | Search PDFs using semantic similarity    | `search revenue analysis`    |
| `delete <id>`     | Delete a PDF by ID                       | `delete 5`                   |
| `stats`           | Show collection statistics               | `stats`                      |
| `help`            | Show available commands                  | `help`                       |
| `menu` / `back`   | Return to main menu                      | `menu`                       |
| `quit` / `exit`   | Exit the application                     | `quit`                       |

## Performance Considerations

### Storage

- **SQLite**: ~1 KB per chunk (text + metadata)
- **ChromaDB**: ~384 bytes per chunk (vector embeddings)
- **Total**: ~1.4 KB per chunk

For a 100-page PDF with ~200 chunks:
- SQLite: ~200 KB
- ChromaDB: ~77 KB
- **Total: ~277 KB**

### Search Performance

- Embedding generation: ~50ms per query
- Vector search: ~10ms for 1000 chunks
- **Total latency**: ~60ms for typical queries

### Ingestion Performance

- Text extraction: ~100 pages/second
- Chunking: ~1000 chunks/second
- Embedding generation: ~100 chunks/second
- **Bottleneck**: Embedding generation

Estimated ingestion time:
- 10-page PDF: ~2 seconds
- 100-page PDF: ~20 seconds
- 1000-page PDF: ~3 minutes

## Error Handling

### Duplicate Detection

PDFs are tracked by `filepath`. Attempting to ingest the same file twice will:
1. Check if filepath exists in database
2. Return `{"status": "skipped", "reason": "already_exists"}`
3. No duplicate data is created

### Missing Dependencies

If ChromaDB, pypdf, or langchain-text-splitters are not installed:
1. `PDF_SUPPORT_AVAILABLE = False`
2. Attempting to use PDF features shows installation instructions
3. User prompted to install: `pip install chromadb pypdf langchain-text-splitters`

## Data Persistence

### Files Created

```
project_root/
├── pdf_metadata.db          # SQLite database
├── chroma_db/               # ChromaDB persistent storage
│   ├── chroma.sqlite3       # ChromaDB metadata
│   └── [embedding files]    # Vector storage
```

### Backup Recommendations

1. **SQLite**: Copy `pdf_metadata.db`
2. **ChromaDB**: Copy entire `chroma_db/` directory
3. Both must be backed up together to maintain consistency

### Migration

To move the system to another machine:
1. Copy both `pdf_metadata.db` and `chroma_db/` directory
2. Ensure same directory structure
3. Install dependencies
4. System will work immediately

## Security Considerations

1. **Path Traversal**: Uses `Path().resolve()` to normalize paths
2. **SQL Injection**: Uses parameterized queries exclusively
3. **File Validation**: Checks file existence before ingestion
4. **No Remote Access**: Both databases are local-only

## Future Enhancements

- [ ] Custom embedding models (OpenAI, Cohere)
- [ ] Multi-modal support (images, tables from PDFs)
- [ ] Batch ingestion for multiple PDFs
- [ ] Advanced search filters (by date, author, etc.)
- [ ] Full-text search (SQLite FTS5) + vector search hybrid
- [ ] Web interface for PDF management
- [ ] Export search results to markdown/PDF
