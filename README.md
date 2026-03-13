# Gemini OCR PoC (Structured JSON for RAG)

This PoC uses `gemini-2.5-flash-lite` to:

1. OCR a document/image file
2. Extract full text + per-page text + semantic blocks
3. Return strict structured JSON with RAG-ready chunks
4. Show OCR quality, processing time, token usage, and estimated cost in a local UI
5. Process PDF pages in parallel for faster throughput

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
```window
In Command Prompt (cmd):
.\.venv\Scripts\activate

In PowerShell:
.\.venv\Scripts\Activate.ps1
```
Set your API key:

```bash
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

## Run CLI

```bash
python gemini_ocr_poc.py \
  --input /path/to/document.pdf \
  --output ./out/ocr.json
```

Optional flags:

- `--model gemini-2.5-flash-lite`
- `--chunk-size 1200`
- `--max-workers 4` (PDF pages processed in parallel)
- `--pdf-dpi 180` (lower is faster, higher can improve OCR accuracy)
- `--no-quality-boost` (disable aggressive post-processing; enabled by default)
- `--api-key <key>` (instead of env var)

## Run Web UI

```bash
streamlit run streamlit_app.py
```

Then open the local URL shown by Streamlit, upload a PDF/image, click **Run OCR**, and review:

- Quality score + verdict
- Processing time, token usage, estimated paid-tier cost
- Quality boost status + boilerplate lines removed
- Full extracted text
- Per-page text
- Structured blocks (heading/paragraph/table/key_value/etc.)
- RAG chunks
- Downloadable JSON

## Output shape

```json
{
  "source_file": "document.pdf",
  "mime_type": "application/pdf",
  "extraction_mode": "pdf_parallel",
  "language": "en",
  "full_text": "...",
  "pages": [
    { "page_number": 1, "text": "..." }
  ],
  "blocks": [
    {
      "block_id": "p1_b1",
      "page_number": 1,
      "order": 1,
      "block_type": "heading",
      "title": "Invoice",
      "text": "ACME Corp",
      "key_values": []
    }
  ],
  "chunks": [
    { "chunk_id": "c1", "page_numbers": [1], "text": "..." }
  ],
  "usage": {
    "requests": 3,
    "input_tokens": 12345,
    "output_tokens": 2345,
    "total_tokens": 14690,
    "elapsed_seconds": 7.21,
    "estimated_cost_usd_paid_tier": 0.00217
  }
}
```

Use `chunks[*].text` for embedding, and keep `page_numbers` for citations in your RAG pipeline.

Estimated cost is calculated from your provided Gemini 2.5 Flash-Lite paid tier:

- Input: `$0.10 / 1M tokens`
- Output: `$0.40 / 1M tokens`
