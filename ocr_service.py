from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import mimetypes
import os
import random
import re
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import fitz
from PIL import Image
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

if TYPE_CHECKING:
    from google.genai import Client


class KeyValue(BaseModel):
    key: str = ""
    value: str = ""


class DocumentBlock(BaseModel):
    block_id: str = Field(description="Stable id for this block")
    page_number: int = Field(description="1-based page number")
    order: int = Field(description="Reading order index within page")
    block_type: str = Field(description="heading|paragraph|list_item|table|key_value|header|footer|other")
    title: str = ""
    text: str = ""
    key_values: list[KeyValue] = Field(default_factory=list)


class PageText(BaseModel):
    page_number: int = Field(description="1-based page number")
    text: str = Field(description="OCR text for this page")


class RAGChunk(BaseModel):
    chunk_id: str = Field(description="Stable chunk identifier")
    page_numbers: list[int] = Field(description="Pages covered by this chunk")
    text: str = Field(description="Chunk text for embedding")


class OCRResult(BaseModel):
    source_file: str = Field(description="Original file name")
    mime_type: str = Field(description="Input MIME type")
    extraction_mode: str = Field(description="single_file|pdf_parallel|pdf_batch")
    language: str = Field(description="Primary detected language, ISO code if possible")
    full_text: str = Field(description="All extracted text in reading order")
    pages: list[PageText] = Field(description="Per-page extracted text")
    blocks: list[DocumentBlock] = Field(description="Document blocks in reading order")
    chunks: list[RAGChunk] = Field(description="RAG-ready chunks")
    usage: dict = Field(description="Token, timing and estimated cost metrics")
    cached: bool = Field(default=False, description="Whether result was retrieved from cache")


class PageOCRResponse(BaseModel):
    page_number: int
    language: str = ""
    text: str = ""
    blocks: list[DocumentBlock] = Field(default_factory=list)


class BatchOCRResponse(BaseModel):
    """Response for batch OCR of multiple pages."""
    pages: list[PageOCRResponse] = Field(default_factory=list)


# =============================================================================
# BOOKLET SCANNER MODELS
# =============================================================================

class BookletEntry(BaseModel):
    """Single booklet entry with roll number and barcode."""
    roll_number: str = Field(description="The handwritten roll number from the boxes")
    barcode_value: str = Field(description="The barcode value extracted from the barcode sticker")
    roll_confidence: float = Field(default=0.0, description="Heuristic confidence for roll number (0..1)")
    barcode_confidence: float = Field(default=0.0, description="Heuristic confidence for barcode (0..1)")
    roll_source: str = Field(default="gemini", description="gemini|filename_digits")
    barcode_source: str = Field(default="gemini_sticker", description="gemini_sticker|missing_no_sticker")
    barcode_is_sticker: bool = Field(default=False, description="True only if barcode is on an attached sticker/label")
    barcode_position_hint: str = Field(default="unknown", description="right|left|center|unknown")
    processing_seconds: float = Field(default=0.0, description="Processing time for this PDF in seconds")
    input_tokens: int = Field(default=0, description="Prompt tokens used for this PDF")
    output_tokens: int = Field(default=0, description="Output tokens used for this PDF")
    estimated_cost_usd: float = Field(default=0.0, description="Estimated Gemini paid-tier cost for this PDF")
    review_required: bool = Field(default=False, description="Whether this entry should be manually reviewed")
    review_note: str = Field(default="", description="Short manual review suggestion")
    source_pdf: str = Field(default="", description="Source PDF file name")


class BookletScanResponse(BaseModel):
    """Response for booklet scanner extraction."""
    entries: list[BookletEntry] = Field(default_factory=list)
    total_pages: int = 0
    successful: int = 0
    failed: int = 0
    elapsed_seconds: float = 0.0
    avg_seconds_per_file: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    estimated_total_cost_usd: float = 0.0
    errors: list[str] = Field(default_factory=list)


class StickerBarcodeDecision(BaseModel):
    """Decision model for sticker barcode verification/recovery."""
    barcode_value: str = ""
    barcode_is_sticker: bool = False
    barcode_position_hint: str = "unknown"
    barcode_orientation: str = "unknown"
    confidence: float = 0.0
    sticker_boundary_visible: bool = False
    is_margin_printed_barcode: bool = False
    reason: str = ""


@dataclass
class PageRunResult:
    page: PageOCRResponse
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float


# Default settings optimized for speed
DEFAULT_MAX_WORKERS = 8  # Increased from 4
DEFAULT_PDF_DPI = 120    # Decreased from 180
DEFAULT_BATCH_SIZE = 3   # Pages per batch API call

INPUT_USD_PER_MILLION = 0.10
OUTPUT_USD_PER_MILLION = 0.40
_OCR_NETWORK_MAX_RETRIES = 3
_OCR_NETWORK_BASE_DELAY_SECONDS = 0.75
_OCR_NETWORK_MAX_DELAY_SECONDS = 6.0

# Simple in-memory cache
_OCR_CACHE: dict[str, OCRResult] = {}
_CACHE_MAX_SIZE = 100


def detect_mime_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.as_posix())
    return mime or "application/octet-stream"


def _build_page_prompt(page_number: int) -> str:
    return (
        "You are an OCR engine for ONE page only.\n"
        f"Extract text from page {page_number} in natural reading order.\n"
        "Output rules:\n"
        "1) Return all text exactly; do not summarize.\n"
        "2) Build blocks with useful semantic types: heading, paragraph, list_item, table, key_value, header, footer, other.\n"
        "3) For key-value regions (forms/invoices), fill key_values pairs.\n"
        "4) Keep table rows readable in block text.\n"
        "5) Return valid JSON only matching schema."
    )


def _build_batch_prompt(page_numbers: list[int]) -> str:
    pages_str = ", ".join(str(p) for p in page_numbers)
    return (
        "You are an OCR engine for multiple pages.\n"
        f"Extract text from pages {pages_str} in natural reading order.\n"
        "Output rules:\n"
        "1) Return all text exactly; do not summarize.\n"
        "2) Build blocks with useful semantic types: heading, paragraph, list_item, table, key_value, header, footer, other.\n"
        "3) For key-value regions (forms/invoices), fill key_values pairs.\n"
        "4) Keep table rows readable in block text.\n"
        "5) Return valid JSON only matching schema.\n"
        "6) Process each page independently and return a 'pages' array with each page's results."
    )


def _block_text(block: DocumentBlock) -> str:
    parts: list[str] = []
    if block.title.strip():
        parts.append(block.title.strip())
    if block.text.strip():
        parts.append(block.text.strip())
    for kv in block.key_values:
        k = kv.key.strip()
        v = kv.value.strip()
        if k or v:
            parts.append(f"{k}: {v}".strip())
    return "\n".join(parts).strip()


def _page_text_from_blocks(blocks: list[DocumentBlock]) -> str:
    ordered = sorted(blocks, key=lambda b: b.order)
    parts = [_block_text(b) for b in ordered]
    return "\n\n".join([p for p in parts if p.strip()]).strip()


def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _boilerplate_blacklist(blocks: list[DocumentBlock], pages_count: int) -> set[str]:
    counter: Counter[str] = Counter()
    for block in blocks:
        text = _block_text(block)
        for line in text.splitlines():
            norm = _normalize_line(line)
            if norm:
                counter[norm] += 1

    min_repeat = max(2, int(pages_count * 0.4))
    blacklist: set[str] = set()
    for line, count in counter.items():
        words = len(re.findall(r"\b\w+\b", line))
        if count >= min_repeat and len(line) <= 100 and words <= 12:
            blacklist.add(line)
    return blacklist


def _remove_boilerplate_lines(text: str, blacklist: set[str]) -> tuple[str, int]:
    removed = 0
    kept: list[str] = []
    for line in text.splitlines():
        norm = _normalize_line(line)
        if norm and norm in blacklist:
            removed += 1
            continue
        kept.append(line.rstrip())
    cleaned = "\n".join(kept).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned, removed


def _postprocess_blocks(blocks: list[DocumentBlock], pages_count: int) -> tuple[list[DocumentBlock], int]:
    if not blocks:
        return [], 0

    blacklist = _boilerplate_blacklist(blocks, pages_count=max(1, pages_count))
    cleaned_blocks: list[DocumentBlock] = []
    removed_lines = 0

    for block in sorted(blocks, key=lambda b: (b.page_number, b.order)):
        if block.block_type in {"header", "footer"}:
            continue

        title, rm1 = _remove_boilerplate_lines(block.title or "", blacklist)
        text, rm2 = _remove_boilerplate_lines(block.text or "", blacklist)
        removed_lines += rm1 + rm2

        new_kv: list[KeyValue] = []
        for kv in block.key_values:
            k, rmk = _remove_boilerplate_lines(kv.key or "", blacklist)
            v, rmv = _remove_boilerplate_lines(kv.value or "", blacklist)
            removed_lines += rmk + rmv
            if k or v:
                new_kv.append(KeyValue(key=k, value=v))

        title = _normalize_line(title)
        text = re.sub(r"[ \t]+", " ", text).strip()
        if not title and not text and not new_kv:
            continue

        cleaned_blocks.append(
            block.model_copy(update={"title": title, "text": text, "key_values": new_kv})
        )

    mergeable = {"paragraph", "other", "list_item"}
    merged: list[DocumentBlock] = []
    for block in cleaned_blocks:
        if not merged:
            merged.append(block)
            continue

        prev = merged[-1]
        prev_text = _block_text(prev)
        curr_text = _block_text(block)
        should_merge = (
            prev.page_number == block.page_number
            and prev.block_type == block.block_type
            and prev.block_type in mergeable
            and len(prev_text) < 900
            and len(curr_text) < 500
        )
        if should_merge:
            merged[-1] = prev.model_copy(
                update={"text": f"{prev.text}\n{block.text}".strip(), "key_values": prev.key_values + block.key_values}
            )
        else:
            merged.append(block)

    for idx, block in enumerate(merged, start=1):
        merged[idx - 1] = block.model_copy(
            update={"order": idx, "block_id": block.block_id or f"b{idx:04d}"}
        )

    return merged, removed_lines


def _chunks_from_blocks(blocks: list[DocumentBlock], chunk_size: int, overlap: int = 120) -> list[RAGChunk]:
    chunks: list[RAGChunk] = []
    if not blocks:
        return chunks

    current_text = ""
    current_pages: set[int] = set()

    def flush_chunk() -> None:
        nonlocal current_text, current_pages
        text = current_text.strip()
        if not text:
            current_text = ""
            current_pages = set()
            return
        chunks.append(
            RAGChunk(
                chunk_id=f"c{len(chunks) + 1:04d}",
                page_numbers=sorted(current_pages),
                text=text,
            )
        )
        if overlap > 0 and len(text) > overlap:
            current_text = text[-overlap:]
        else:
            current_text = ""
        current_pages = set()

    for block in blocks:
        text = _block_text(block)
        if not text:
            continue
        piece = f"[p{block.page_number}:{block.block_type}]\n{text}\n"

        if len(current_text) + len(piece) > chunk_size and current_text:
            flush_chunk()

        current_text = f"{current_text}\n{piece}".strip()
        current_pages.add(block.page_number)

    flush_chunk()
    return chunks


# =============================================================================
# CACHING LAYER (Phase 3)
# =============================================================================

def _compute_cache_key(
    input_path: Path,
    model: str,
    chunk_size: int,
    pdf_dpi: int,
    quality_boost: bool,
    batch_size: int,
) -> str:
    """Compute a unique cache key based on file content and settings."""
    # Use file modification time + size for quick hash
    stat = input_path.stat()
    quick_info = f"{input_path.name}:{stat.st_size}:{stat.st_mtime}"
    
    # Combine with settings
    settings_str = f"{model}:{chunk_size}:{pdf_dpi}:{quality_boost}:{batch_size}"
    
    # Create hash
    combined = f"{quick_info}:{settings_str}"
    return hashlib.md5(combined.encode()).hexdigest()


def _get_cached_result(cache_key: str) -> OCRResult | None:
    """Retrieve cached result if available."""
    return _OCR_CACHE.get(cache_key)


def _set_cached_result(cache_key: str, result: OCRResult) -> None:
    """Store result in cache with LRU-like eviction."""
    global _OCR_CACHE
    
    # Simple LRU: if cache is full, remove oldest entry
    if len(_OCR_CACHE) >= _CACHE_MAX_SIZE:
        # Remove first (oldest) entry
        oldest_key = next(iter(_OCR_CACHE))
        del _OCR_CACHE[oldest_key]
    
    _OCR_CACHE[cache_key] = result


# =============================================================================
# CLIENT REUSE (Phase 1)
# =============================================================================

@functools.lru_cache(maxsize=1)
def _get_shared_client(api_key: str) -> Client:
    """Get or create a shared Gemini client (cached per API key)."""
    return genai.Client(api_key=api_key)


def _extract_page_structured(
    *,
    image_path: Path,
    page_number: int,
    client: Client,  # CHANGED: Pass client instead of api_key
    model: str,
) -> PageRunResult:
    """Extract structured OCR data from a single page image using shared client."""
    started_at = time.perf_counter()
    mime_type = detect_mime_type(image_path)
    data = image_path.read_bytes()
    part = types.Part.from_bytes(data=data, mime_type=mime_type)

    response = None
    last_exc: Exception | None = None
    for attempt in range(1, _OCR_NETWORK_MAX_RETRIES + 2):
        try:
            # Use passed client instead of creating new one
            response = client.models.generate_content(
                model=model,
                contents=[_build_page_prompt(page_number), part],
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": PageOCRResponse.model_json_schema(),
                    "temperature": 0,
                },
            )
            break
        except Exception as exc:
            last_exc = exc
            if not _is_transient_network_error(exc) or attempt > _OCR_NETWORK_MAX_RETRIES:
                raise
            time.sleep(_compute_ocr_retry_delay(attempt))

    if response is None:
        raise RuntimeError(f"Page {page_number} OCR failed after retries: {last_exc or 'unknown error'}")

    raw_json = response.text or "{}"

    try:
        parsed = PageOCRResponse.model_validate_json(raw_json)
    except ValidationError as exc:
        raise ValueError(f"Invalid page JSON for page {page_number}: {exc}") from exc

    normalized_blocks: list[DocumentBlock] = []
    for idx, block in enumerate(parsed.blocks, start=1):
        normalized_blocks.append(
            block.model_copy(
                update={
                    "page_number": page_number,
                    "order": block.order if block.order > 0 else idx,
                    "block_id": block.block_id or f"p{page_number}_b{idx}",
                }
            )
        )

    normalized_page = parsed.model_copy(
        update={"page_number": page_number, "blocks": normalized_blocks}
    )
    usage = getattr(response, "usage_metadata", None)
    input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    elapsed_seconds = time.perf_counter() - started_at
    return PageRunResult(
        page=normalized_page,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        elapsed_seconds=elapsed_seconds,
    )


def _extract_batch_structured(
    *,
    image_paths: list[tuple[int, Path]],  # (page_number, path) pairs
    client: Client,
    model: str,
) -> list[PageRunResult]:
    """Extract OCR from multiple pages in a single API call (Phase 2)."""
    started_at = time.perf_counter()
    
    if not image_paths:
        return []
    
    # Build parts list with all images
    page_numbers = [pn for pn, _ in image_paths]
    parts: list[types.Part] = []
    
    for page_number, image_path in image_paths:
        mime_type = detect_mime_type(image_path)
        data = image_path.read_bytes()
        part = types.Part.from_bytes(data=data, mime_type=mime_type)
        parts.append(part)
    
    # Single API call for multiple pages
    response = client.models.generate_content(
        model=model,
        contents=[_build_batch_prompt(page_numbers)] + parts,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": BatchOCRResponse.model_json_schema(),
            "temperature": 0,
        },
    )
    raw_json = response.text or '{"pages": []}'
    
    try:
        parsed = BatchOCRResponse.model_validate_json(raw_json)
    except ValidationError as exc:
        # Fallback: process pages individually if batch fails
        return [
            _extract_page_structured(
                image_path=img_path,
                page_number=pn,
                client=client,
                model=model,
            )
            for pn, img_path in image_paths
        ]
    
    usage = getattr(response, "usage_metadata", None)
    total_input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    total_output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    elapsed_seconds = time.perf_counter() - started_at
    
    # Distribute tokens evenly across pages for tracking
    per_page_input = total_input_tokens // len(image_paths)
    per_page_output = total_output_tokens // len(image_paths)
    per_page_time = elapsed_seconds / len(image_paths)
    
    results: list[PageRunResult] = []
    for idx, page_response in enumerate(parsed.pages):
        page_number = page_numbers[idx] if idx < len(page_numbers) else idx + 1
        
        normalized_blocks: list[DocumentBlock] = []
        for block_idx, block in enumerate(page_response.blocks, start=1):
            normalized_blocks.append(
                block.model_copy(
                    update={
                        "page_number": page_number,
                        "order": block.order if block.order > 0 else block_idx,
                        "block_id": block.block_id or f"p{page_number}_b{block_idx}",
                    }
                )
            )
        
        normalized_page = page_response.model_copy(
            update={"page_number": page_number, "blocks": normalized_blocks}
        )
        
        results.append(PageRunResult(
            page=normalized_page,
            input_tokens=per_page_input,
            output_tokens=per_page_output,
            elapsed_seconds=per_page_time,
        ))
    
    return results


# =============================================================================
# PARALLEL PDF RENDERING (Phase 2)
# =============================================================================

def _render_single_page(args: tuple[str, int, int]) -> tuple[int, Path]:
    """Render a single PDF page to image. Used by ProcessPoolExecutor."""
    pdf_path_str, page_idx, dpi = args
    pdf_path = Path(pdf_path_str)
    
    doc = fitz.open(pdf_path.as_posix())
    try:
        zoom = max(1.0, dpi / 72.0)
        matrix = fitz.Matrix(zoom, zoom)
        page = doc.load_page(page_idx)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        
        fd, tmp_name = tempfile.mkstemp(suffix=f"_p{page_idx + 1}.png")
        os.close(fd)
        pix.save(tmp_name)
        return (page_idx + 1, Path(tmp_name))
    finally:
        doc.close()


def _pdf_to_page_images_parallel(pdf_path: Path, dpi: int, max_workers: int = 4) -> list[tuple[int, Path]]:
    """Convert PDF pages to images using parallel processing."""
    # First, get page count
    doc = fitz.open(pdf_path.as_posix())
    page_count = doc.page_count
    doc.close()
    
    if page_count == 0:
        return []
    
    # Prepare arguments for parallel processing
    args_list = [
        (str(pdf_path.resolve()), page_idx, dpi)
        for page_idx in range(page_count)
    ]
    
    # Use ProcessPoolExecutor for CPU-bound image rendering
    workers = min(max_workers, page_count, os.cpu_count() or 4)
    
    # For small page counts, sequential is often faster due to process spawn overhead
    if page_count <= 3:
        return _pdf_to_page_images_sequential(pdf_path, dpi)
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_render_single_page, args_list))
    
    return sorted(results, key=lambda x: x[0])


def _pdf_to_page_images_sequential(pdf_path: Path, dpi: int) -> list[tuple[int, Path]]:
    """Convert PDF pages to images sequentially (fallback for small PDFs)."""
    result: list[tuple[int, Path]] = []
    doc = fitz.open(pdf_path.as_posix())
    zoom = max(1.0, dpi / 72.0)
    matrix = fitz.Matrix(zoom, zoom)

    try:
        for page_idx in range(doc.page_count):
            page = doc.load_page(page_idx)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            fd, tmp_name = tempfile.mkstemp(suffix=f"_p{page_idx + 1}.png")
            os.close(fd)
            pix.save(tmp_name)
            result.append((page_idx + 1, Path(tmp_name)))
    finally:
        doc.close()

    return result


def _pdf_to_page_images(pdf_path: Path, dpi: int, max_workers: int = 4) -> list[tuple[int, Path]]:
    """Convert PDF to page images, using parallel processing for larger PDFs."""
    # Use parallel for PDFs with more than 3 pages
    doc = fitz.open(pdf_path.as_posix())
    page_count = doc.page_count
    doc.close()
    
    if page_count > 3:
        return _pdf_to_page_images_parallel(pdf_path, dpi, max_workers)
    else:
        return _pdf_to_page_images_sequential(pdf_path, dpi)


# =============================================================================
# ASYNC API CALLS (Phase 3)
# =============================================================================

async def _extract_page_async(
    *,
    image_path: Path,
    page_number: int,
    client: Client,
    model: str,
) -> PageRunResult:
    """Async version of page extraction."""
    started_at = time.perf_counter()
    mime_type = detect_mime_type(image_path)
    data = image_path.read_bytes()
    part = types.Part.from_bytes(data=data, mime_type=mime_type)

    response = None
    last_exc: Exception | None = None
    for attempt in range(1, _OCR_NETWORK_MAX_RETRIES + 2):
        try:
            # Use async API
            response = await client.aio.models.generate_content(
                model=model,
                contents=[_build_page_prompt(page_number), part],
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": PageOCRResponse.model_json_schema(),
                    "temperature": 0,
                },
            )
            break
        except Exception as exc:
            last_exc = exc
            if not _is_transient_network_error(exc) or attempt > _OCR_NETWORK_MAX_RETRIES:
                raise
            await asyncio.sleep(_compute_ocr_retry_delay(attempt))

    if response is None:
        raise RuntimeError(f"Page {page_number} OCR failed after retries: {last_exc or 'unknown error'}")

    raw_json = response.text or "{}"

    try:
        parsed = PageOCRResponse.model_validate_json(raw_json)
    except ValidationError as exc:
        raise ValueError(f"Invalid page JSON for page {page_number}: {exc}") from exc

    normalized_blocks: list[DocumentBlock] = []
    for idx, block in enumerate(parsed.blocks, start=1):
        normalized_blocks.append(
            block.model_copy(
                update={
                    "page_number": page_number,
                    "order": block.order if block.order > 0 else idx,
                    "block_id": block.block_id or f"p{page_number}_b{idx}",
                }
            )
        )

    normalized_page = parsed.model_copy(
        update={"page_number": page_number, "blocks": normalized_blocks}
    )
    usage = getattr(response, "usage_metadata", None)
    input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    elapsed_seconds = time.perf_counter() - started_at
    return PageRunResult(
        page=normalized_page,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        elapsed_seconds=elapsed_seconds,
    )


async def _extract_pages_concurrent_async(
    *,
    image_paths: list[tuple[int, Path]],
    client: Client,
    model: str,
    max_concurrent: int = 10,
) -> list[PageRunResult]:
    """Extract multiple pages concurrently using async."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def _limited_extract(args: tuple[int, Path]) -> PageRunResult:
        async with semaphore:
            return await _extract_page_async(
                image_path=args[1],
                page_number=args[0],
                client=client,
                model=model,
            )
    
    tasks = [_limited_extract((pn, path)) for pn, path in image_paths]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any exceptions
    final_results: list[PageRunResult] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            raise ValueError(f"Page {image_paths[i][0]} OCR failed: {result}")
        final_results.append(result)
    
    return final_results


def _run_async_extraction(
    image_paths: list[tuple[int, Path]],
    client: Client,
    model: str,
    max_concurrent: int = 10,
) -> list[PageRunResult]:
    """Run async extraction in event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    
    if loop and loop.is_running():
        # If already in async context, create task
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(
                asyncio.run,
                _extract_pages_concurrent_async(
                    image_paths=image_paths,
                    client=client,
                    model=model,
                    max_concurrent=max_concurrent,
                )
            )
            return future.result()
    else:
        return asyncio.run(
            _extract_pages_concurrent_async(
                image_paths=image_paths,
                client=client,
                model=model,
                max_concurrent=max_concurrent,
            )
        )


# =============================================================================
# MAIN EXTRACTION FUNCTION
# =============================================================================

def extract_ocr_from_path(
    *,
    input_path: Path,
    api_key: str,
    model: str = "gemini-2.5-flash-lite",
    chunk_size: int = 1200,
    max_workers: int = DEFAULT_MAX_WORKERS,  # Changed default
    pdf_dpi: int = DEFAULT_PDF_DPI,  # Changed default
    quality_boost: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,  # New parameter
    use_cache: bool = True,  # New parameter
    use_async: bool = True,  # New parameter
) -> OCRResult:
    """
    Extract OCR from a file with maximum performance optimizations.
    
    Args:
        input_path: Path to input file (PDF or image)
        api_key: Gemini API key
        model: Gemini model name
        chunk_size: Target characters per RAG chunk
        max_workers: Parallel workers for processing
        pdf_dpi: DPI for PDF rendering (lower = faster)
        quality_boost: Enable aggressive post-processing
        batch_size: Pages per batch API call (0 = no batching)
        use_cache: Enable result caching
        use_async: Use async API calls for better concurrency
    
    Returns:
        OCRResult with extracted text and metadata
    """
    started_at = time.perf_counter()
    
    # Check cache first
    if use_cache:
        cache_key = _compute_cache_key(
            input_path=input_path,
            model=model,
            chunk_size=chunk_size,
            pdf_dpi=pdf_dpi,
            quality_boost=quality_boost,
            batch_size=batch_size,
        )
        cached = _get_cached_result(cache_key)
        if cached is not None:
            # Return cached result with updated timing
            return cached.model_copy(
                update={
                    "cached": True,
                    "usage": {
                        **cached.usage,
                        "cached": True,
                        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
                    }
                }
            )
    
    mime_type = detect_mime_type(input_path)
    is_pdf = mime_type == "application/pdf" or input_path.suffix.lower() == ".pdf"

    # Get shared client (Phase 1 optimization)
    client = _get_shared_client(api_key)
    
    page_runs: list[PageRunResult] = []
    page_images: list[tuple[int, Path]] = []
    extraction_mode = "single_file"

    try:
        if is_pdf:
            extraction_mode = "pdf_parallel"
            
            # Phase 2: Parallel PDF rendering
            page_images = _pdf_to_page_images(input_path, dpi=pdf_dpi, max_workers=max_workers)
            if not page_images:
                raise ValueError("No pages found in PDF.")

            # Decide on processing strategy
            if batch_size > 1 and len(page_images) > 1:
                # Phase 2: Batch API calls
                extraction_mode = "pdf_batch"
                batches: list[list[tuple[int, Path]]] = []
                current_batch: list[tuple[int, Path]] = []
                
                for page_item in page_images:
                    current_batch.append(page_item)
                    if len(current_batch) >= batch_size:
                        batches.append(current_batch)
                        current_batch = []
                if current_batch:
                    batches.append(current_batch)
                
                # Process batches with thread pool for parallel batch calls
                workers = max(1, min(max_workers, len(batches)))
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(
                            _extract_batch_structured,
                            image_paths=batch,
                            client=client,
                            model=model,
                        ): i
                        for i, batch in enumerate(batches)
                    }
                    for future in as_completed(futures):
                        batch_idx = futures[future]
                        try:
                            batch_results = future.result()
                            page_runs.extend(batch_results)
                        except Exception as exc:
                            raise ValueError(f"Batch {batch_idx} OCR failed: {exc}") from exc
            
            elif use_async and len(page_images) > 3:
                # Phase 3: Async concurrent processing
                page_runs = _run_async_extraction(
                    image_paths=page_images,
                    client=client,
                    model=model,
                    max_concurrent=max_workers * 2,
                )
            else:
                # Phase 1: Thread pool with shared client
                workers = max(1, min(max_workers, len(page_images)))
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(
                            _extract_page_structured,
                            image_path=img_path,
                            page_number=page_number,
                            client=client,  # Pass shared client
                            model=model,
                        ): page_number
                        for page_number, img_path in page_images
                    }
                    for future in as_completed(futures):
                        page_number = futures[future]
                        try:
                            page_runs.append(future.result())
                        except Exception as exc:
                            raise ValueError(f"Page {page_number} OCR failed: {exc}") from exc

            page_runs.sort(key=lambda item: item.page.page_number)
        else:
            # Single image
            page_runs = [
                _extract_page_structured(
                    image_path=input_path,
                    page_number=1,
                    client=client,
                    model=model,
                )
            ]

        pages: list[PageText] = []
        blocks: list[DocumentBlock] = []
        languages: list[str] = []

        for run in page_runs:
            page = run.page
            text = (page.text or "").strip()
            if not text and page.blocks:
                text = _page_text_from_blocks(page.blocks)
            pages.append(PageText(page_number=page.page_number, text=text))
            languages.append((page.language or "").strip())
            ordered_blocks = sorted(page.blocks, key=lambda b: b.order)
            blocks.extend(ordered_blocks)

        blocks.sort(key=lambda b: (b.page_number, b.order))
        removed_boilerplate_lines = 0
        if quality_boost:
            blocks, removed_boilerplate_lines = _postprocess_blocks(blocks, pages_count=len(pages))

        text_by_page: dict[int, list[DocumentBlock]] = {}
        for block in blocks:
            text_by_page.setdefault(block.page_number, []).append(block)

        normalized_pages: list[PageText] = []
        for page in pages:
            block_text = _page_text_from_blocks(text_by_page.get(page.page_number, []))
            normalized_pages.append(
                PageText(page_number=page.page_number, text=block_text or page.text)
            )
        pages = normalized_pages

        full_text = "\n\n".join([p.text for p in pages if p.text.strip()]).strip()
        chunks = _chunks_from_blocks(blocks=blocks, chunk_size=chunk_size)

        if not chunks and full_text:
            chunks = [
                RAGChunk(chunk_id="c0001", page_numbers=[1], text=full_text[:chunk_size])
            ]

        lang_counter = Counter([x for x in languages if x])
        language = lang_counter.most_common(1)[0][0] if lang_counter else ""
        total_input_tokens = sum(item.input_tokens for item in page_runs)
        total_output_tokens = sum(item.output_tokens for item in page_runs)
        elapsed_seconds = round(time.perf_counter() - started_at, 3)
        estimated_cost_usd = round(
            (total_input_tokens / 1_000_000) * INPUT_USD_PER_MILLION
            + (total_output_tokens / 1_000_000) * OUTPUT_USD_PER_MILLION,
            6,
        )
        usage = {
            "requests": len(page_runs),
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "elapsed_seconds": elapsed_seconds,
            "estimated_cost_usd_paid_tier": estimated_cost_usd,
            "pricing_basis": {
                "input_usd_per_million_tokens": INPUT_USD_PER_MILLION,
                "output_usd_per_million_tokens": OUTPUT_USD_PER_MILLION,
            },
            "quality_boost_enabled": quality_boost,
            "boilerplate_lines_removed": removed_boilerplate_lines,
            "cached": False,
            "optimizations": {
                "client_reuse": True,
                "parallel_workers": max_workers,
                "pdf_dpi": pdf_dpi,
                "batch_size": batch_size if is_pdf else 0,
                "async_enabled": use_async and is_pdf and len(page_images) > 3,
            }
        }

        result = OCRResult(
            source_file=input_path.name,
            mime_type=mime_type,
            extraction_mode=extraction_mode,
            language=language,
            full_text=full_text,
            pages=pages,
            blocks=blocks,
            chunks=chunks,
            usage=usage,
            cached=False,
        )
        
        # Cache the result
        if use_cache:
            _set_cached_result(cache_key, result)
        
        return result
        
    finally:
        for _, img_path in page_images:
            img_path.unlink(missing_ok=True)


def quality_report(result: OCRResult) -> dict:
    full_text = (result.full_text or "").strip()
    pages = result.pages or []
    blocks = result.blocks or []
    chunks = result.chunks or []

    if not full_text and blocks:
        full_text = "\n\n".join(
            [
                _block_text(b)
                for b in sorted(blocks, key=lambda x: (x.page_number, x.order))
                if _block_text(b)
            ]
        ).strip()

    page_texts = [p.text or "" for p in pages]
    if blocks and page_texts and all(not t.strip() for t in page_texts):
        text_by_page: dict[int, list[DocumentBlock]] = {}
        for b in blocks:
            text_by_page.setdefault(b.page_number, []).append(b)
        page_texts = [
            _page_text_from_blocks(text_by_page.get(p.page_number, []))
            for p in pages
        ]
    chunk_texts = [c.text or "" for c in chunks]

    total_chars = len(full_text)
    total_words = len(re.findall(r"\b\w+\b", full_text))
    empty_pages = sum(1 for t in page_texts if not t.strip())
    avg_chars_per_page = int(sum(len(t) for t in page_texts) / len(page_texts)) if page_texts else 0
    avg_words_per_chunk = (
        int(sum(len(re.findall(r"\b\w+\b", t)) for t in chunk_texts) / len(chunk_texts))
        if chunk_texts
        else 0
    )

    suspicious_chars = len(re.findall(r"[^\x09\x0A\x0D\x20-\x7E]", full_text))
    suspicious_ratio = (suspicious_chars / total_chars) if total_chars else 0.0

    lines = [
        _normalize_line(ln)
        for ln in full_text.splitlines()
        if _normalize_line(ln) and len(_normalize_line(ln)) >= 3
    ]
    duplicate_lines = len(lines) - len(set(lines)) if lines else 0
    duplicate_line_ratio = (duplicate_lines / len(lines)) if lines else 0.0

    block_text_chars = sum(len(_block_text(b)) for b in blocks)
    block_coverage_ratio = (block_text_chars / total_chars) if total_chars else 0.0
    table_blocks = sum(1 for b in blocks if b.block_type == "table")
    kv_blocks = sum(1 for b in blocks if b.block_type == "key_value")

    score = 100
    if total_chars < 100:
        score -= 30
    elif total_chars < 400:
        score -= 10
    if total_words < 30:
        score -= 20
    elif total_words < 120:
        score -= 8
    score -= min(empty_pages * 12, 36)
    score -= min(int(max(0.0, suspicious_ratio - 0.02) * 200), 15)
    score -= min(int(max(0.0, duplicate_line_ratio - 0.15) * 120), 15)
    if avg_words_per_chunk < 20 and chunks:
        score -= 8
    if blocks and block_coverage_ratio < 0.6:
        score -= 8
    score = max(0, min(100, score))

    if score >= 85:
        verdict = "Good"
    elif score >= 65:
        verdict = "Fair"
    else:
        verdict = "Needs review"

    flags: list[str] = []
    if total_chars < 100:
        flags.append("Very little text extracted.")
    if empty_pages:
        flags.append(f"{empty_pages} page(s) appear empty after OCR.")
    if suspicious_ratio > 0.08:
        flags.append("High ratio of unusual characters; possible OCR noise.")
    if duplicate_line_ratio > 0.2:
        flags.append("Many repeated lines detected; possible layout/parsing issue.")
    if not chunks:
        flags.append("No chunks generated for RAG.")
    if blocks and block_coverage_ratio < 0.6:
        flags.append("Block structure coverage is low; inspect block segmentation.")

    return {
        "score": score,
        "verdict": verdict,
        "flags": flags,
        "stats": {
            "total_chars": total_chars,
            "total_words": total_words,
            "pages": len(pages),
            "blocks": len(blocks),
            "chunks": len(chunks),
            "table_blocks": table_blocks,
            "key_value_blocks": kv_blocks,
            "empty_pages": empty_pages,
            "avg_chars_per_page": avg_chars_per_page,
            "avg_words_per_chunk": avg_words_per_chunk,
            "suspicious_char_ratio": round(suspicious_ratio, 4),
            "duplicate_line_ratio": round(duplicate_line_ratio, 4),
            "block_coverage_ratio": round(block_coverage_ratio, 4),
            "extraction_mode": result.extraction_mode,
            "quality_boost_enabled": bool(result.usage.get("quality_boost_enabled")),
            "boilerplate_lines_removed": int(result.usage.get("boilerplate_lines_removed", 0)),
            "cached": result.cached,
        },
    }


# =============================================================================
# CACHE MANAGEMENT
# =============================================================================

def clear_cache() -> int:
    """Clear the OCR result cache. Returns number of entries cleared."""
    global _OCR_CACHE
    count = len(_OCR_CACHE)
    _OCR_CACHE.clear()
    return count


def get_cache_size() -> int:
    """Get current number of cached results."""
    return len(_OCR_CACHE)


# =============================================================================
# BOOKLET SCANNER - Targeted extraction for roll numbers and barcodes
# =============================================================================

_BOOKLET_ROLL_LENGTH_HINTS = {8, 12, 13}
_BOOKLET_RATE_LIMIT_MAX_RETRIES = 6
_BOOKLET_RATE_LIMIT_BASE_DELAY_SECONDS = 1.0
_BOOKLET_RATE_LIMIT_MAX_DELAY_SECONDS = 20.0
_BOOKLET_STICKER_MIN_CONFIDENCE = 0.60
_BOOKLET_STICKER_RECOVERY_DPI = 180


def _build_booklet_prompt() -> str:
    return (
        "You are a specialized OCR for exam booklet first pages.\n"
        "Your task is to extract ONLY two things:\n"
        "1. ROLL NUMBER: The handwritten number in the boxes at the top of the page (e.g., '110220200150')\n"
        "2. BARCODE VALUE: The numeric or alphanumeric value from the attached barcode STICKER label.\n"
        "IMPORTANT barcode rule:\n"
        "- Accept barcode only if it is clearly on a physically pasted sticker/label with distinct sticker boundary.\n"
        "- Sticker can be at any position on the page (left/right/center).\n"
        "- IGNORE barcodes printed as part of the form, including vertical margin barcodes.\n"
        "- If barcode is printed directly on the page (not sticker) or uncertain, do NOT extract it.\n"
        "- If no sticker barcode is present, set barcode_value to empty string and barcode_is_sticker=false.\n"
        "Ignore all other text on the page.\n"
        "Output as JSON with fields: roll_number, barcode_value, barcode_is_sticker, barcode_position_hint.\n"
        "barcode_position_hint must be one of: right, left, center, unknown.\n"
        "If you cannot find either value, leave it as empty string.\n"
    )


def _build_barcode_verify_prompt(candidate_barcode: str) -> str:
    return (
        "You verify whether a barcode belongs to a physically pasted sticker on an exam booklet page.\n"
        f"Candidate barcode value: {candidate_barcode}\n"
        "Rules:\n"
        "1) Accept only if barcode is clearly on a pasted sticker/label (visible sticker boundary or attached patch).\n"
        "2) Reject printed form/margin barcodes (including vertical margin barcode).\n"
        "3) If barcode is vertical orientation, treat it as likely printed/margin unless sticker evidence is very clear.\n"
        "3) If rejected, return empty barcode_value and barcode_is_sticker=false.\n"
        "Return JSON: barcode_value, barcode_is_sticker, barcode_position_hint (right|left|center|unknown), barcode_orientation (horizontal|vertical|unknown), confidence (0..1), sticker_boundary_visible (bool), is_margin_printed_barcode (bool), reason."
    )


def _build_barcode_recovery_prompt() -> str:
    return (
        "Find barcode value ONLY from a physically pasted sticker/label on this exam booklet page.\n"
        "Rules:\n"
        "1) Search whole page for pasted sticker barcode (left/right/center).\n"
        "2) Ignore barcodes printed as part of the form (including vertical margin barcode).\n"
        "3) If barcode is vertical orientation, usually reject unless clear sticker boundary is visible.\n"
        "3) If no sticker barcode is found, set barcode_value='' and barcode_is_sticker=false.\n"
        "4) If uncertain, set low confidence and return no barcode.\n"
        "Return JSON: barcode_value, barcode_is_sticker, barcode_position_hint (right|left|center|unknown), barcode_orientation (horizontal|vertical|unknown), confidence (0..1), sticker_boundary_visible (bool), is_margin_printed_barcode (bool), reason."
    )


def _extract_digits(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit())


def _extract_alnum_upper(value: str) -> str:
    return "".join(ch for ch in value.upper() if ch.isalnum())


def _clamp_confidence(value: float) -> float:
    return round(max(0.0, min(value, 0.99)), 2)


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    rate_limit_markers = (
        "429",
        "rate limit",
        "resource exhausted",
        "too many requests",
        "quota",
    )
    return any(marker in text for marker in rate_limit_markers)


def _is_transient_network_error(exc: Exception) -> bool:
    text = str(exc).lower()
    transient_markers = (
        "winerror 10053",
        "winerror 10054",
        "connection aborted",
        "connection reset",
        "connection forcibly closed",
        "timed out",
        "timeout",
        "temporary failure",
        "temporarily unavailable",
        "service unavailable",
        "503",
        "transport error",
    )
    return any(marker in text for marker in transient_markers)


def _compute_ocr_retry_delay(attempt: int) -> float:
    base = _OCR_NETWORK_BASE_DELAY_SECONDS * (2 ** max(0, attempt - 1))
    jitter = random.uniform(0, 0.35)
    return min(_OCR_NETWORK_MAX_DELAY_SECONDS, base + jitter)


def _compute_retry_delay(attempt: int) -> float:
    base = _BOOKLET_RATE_LIMIT_BASE_DELAY_SECONDS * (2 ** max(0, attempt - 1))
    jitter = random.uniform(0, 0.5)
    return min(_BOOKLET_RATE_LIMIT_MAX_DELAY_SECONDS, base + jitter)


def _estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    input_cost = (input_tokens / 1_000_000) * INPUT_USD_PER_MILLION
    output_cost = (output_tokens / 1_000_000) * OUTPUT_USD_PER_MILLION
    return round(input_cost + output_cost, 8)


def _normalize_position_hint(value: str) -> str:
    norm = (value or "").strip().lower() or "unknown"
    if norm not in {"right", "left", "center", "unknown"}:
        return "unknown"
    return norm


def _normalize_orientation(value: str) -> str:
    norm = (value or "").strip().lower() or "unknown"
    if norm not in {"horizontal", "vertical", "unknown"}:
        return "unknown"
    return norm


def _render_pdf_first_page_image(pdf_path: Path, dpi: int) -> Path:
    doc = fitz.open(pdf_path.as_posix())
    zoom = max(1.0, dpi / 72.0)
    matrix = fitz.Matrix(zoom, zoom)
    try:
        if doc.page_count < 1:
            raise ValueError("PDF has no pages.")
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        fd, tmp_name = tempfile.mkstemp(suffix="_barcode_recovery.png")
        os.close(fd)
        pix.save(tmp_name)
        return Path(tmp_name)
    finally:
        doc.close()


def _crop_image_ratio(
    image_path: Path,
    *,
    left_ratio: float,
    top_ratio: float,
    right_ratio: float,
    bottom_ratio: float,
    suffix: str,
) -> Path | None:
    with Image.open(image_path) as img:
        width, height = img.size
        left = int(width * left_ratio)
        top = int(height * top_ratio)
        right = int(width * right_ratio)
        bottom = int(height * bottom_ratio)
        if right - left < 80 or bottom - top < 80:
            return None
        cropped = img.crop((left, top, right, bottom))
        fd, tmp_name = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        cropped.save(tmp_name)
        return Path(tmp_name)


def _build_recovery_images(image_path: Path) -> list[tuple[str, Path, bool]]:
    images: list[tuple[str, Path, bool]] = [("full", image_path, False)]
    regions = [
        ("right_half", (0.5, 0.0, 1.0, 1.0)),
        ("left_half", (0.0, 0.0, 0.5, 1.0)),
        ("bottom_half", (0.0, 0.45, 1.0, 1.0)),
        ("right_bottom", (0.45, 0.4, 1.0, 1.0)),
    ]
    for name, (l, t, r, b) in regions:
        cropped = _crop_image_ratio(
            image_path,
            left_ratio=l,
            top_ratio=t,
            right_ratio=r,
            bottom_ratio=b,
            suffix=f"_{name}.png",
        )
        if cropped is not None:
            images.append((name, cropped, True))
    return images


def _generate_json_with_retry(
    *,
    image_path: Path,
    prompt: str,
    schema: dict,
    client: Client,
    model: str,
    context_label: str,
    max_retries: int,
) -> tuple[str | None, str | None, int, int]:
    mime_type = detect_mime_type(image_path)
    data = image_path.read_bytes()
    part = types.Part.from_bytes(data=data, mime_type=mime_type)

    response = None
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 2):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[prompt, part],
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": schema,
                    "temperature": 0,
                },
            )
            usage = getattr(response, "usage_metadata", None)
            input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
            output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
            return (response.text or "{}"), None, input_tokens, output_tokens
        except Exception as exc:
            last_exc = exc
            if not _is_rate_limit_error(exc) or attempt > max_retries:
                return None, f"Gemini request failed for {context_label}: {exc}", 0, 0
            time.sleep(_compute_retry_delay(attempt))

    return None, f"Gemini request failed for {context_label}: {last_exc or 'unknown error'}", 0, 0


def _parse_sticker_decision(raw_json: str) -> StickerBarcodeDecision:
    parsed = StickerBarcodeDecision.model_validate_json(raw_json)
    cleaned_value = _extract_alnum_upper(parsed.barcode_value or "")
    cleaned_position = _normalize_position_hint(parsed.barcode_position_hint)
    cleaned_orientation = _normalize_orientation(parsed.barcode_orientation)
    cleaned_conf = max(0.0, min(float(parsed.confidence or 0.0), 1.0))
    is_sticker = bool(parsed.barcode_is_sticker)
    boundary_visible = bool(parsed.sticker_boundary_visible)
    is_margin_printed = bool(parsed.is_margin_printed_barcode)

    # Strict acceptance gate:
    # must be sticker, must show sticker boundary, and must not be printed margin barcode.
    if (
        (not is_sticker)
        or (not boundary_visible)
        or is_margin_printed
        or cleaned_orientation == "vertical"
    ):
        cleaned_value = ""
        is_sticker = False
        cleaned_conf = 0.0
    return parsed.model_copy(
        update={
            "barcode_value": cleaned_value,
            "barcode_is_sticker": is_sticker,
            "barcode_position_hint": cleaned_position,
            "barcode_orientation": cleaned_orientation,
            "confidence": cleaned_conf,
            "sticker_boundary_visible": boundary_visible,
            "is_margin_printed_barcode": is_margin_printed,
        }
    )


def _run_sticker_decision(
    *,
    image_path: Path,
    prompt: str,
    client: Client,
    model: str,
    context_label: str,
    max_retries: int,
) -> tuple[StickerBarcodeDecision | None, str | None, int, int]:
    raw_json, error, in_tok, out_tok = _generate_json_with_retry(
        image_path=image_path,
        prompt=prompt,
        schema=StickerBarcodeDecision.model_json_schema(),
        client=client,
        model=model,
        context_label=context_label,
        max_retries=max_retries,
    )
    if error:
        return None, error, in_tok, out_tok
    try:
        return _parse_sticker_decision(raw_json or "{}"), None, in_tok, out_tok
    except ValidationError as exc:
        return None, f"Invalid sticker decision JSON for {context_label}: {exc}", in_tok, out_tok


def _refine_sticker_barcode(
    *,
    image_path: Path,
    pdf_path: Path,
    pdf_dpi: int,
    entry: BookletEntry,
    client: Client,
    model: str,
    max_retries: int,
    sticker_min_confidence: float,
) -> tuple[BookletEntry, int, int]:
    extra_input_tokens = 0
    extra_output_tokens = 0

    barcode_value = _extract_alnum_upper(entry.barcode_value.strip())
    barcode_is_sticker = bool(entry.barcode_is_sticker)
    barcode_position_hint = _normalize_position_hint(entry.barcode_position_hint)
    best_recovery: StickerBarcodeDecision | None = None

    # Pass 1: verify candidate if present.
    if barcode_value:
        verify_decision, verify_error, in_tok, out_tok = _run_sticker_decision(
            image_path=image_path,
            prompt=_build_barcode_verify_prompt(barcode_value),
            client=client,
            model=model,
            context_label=f"barcode verification {pdf_path.name}",
            max_retries=max_retries,
        )
        extra_input_tokens += in_tok
        extra_output_tokens += out_tok
        if verify_decision is not None:
            if verify_decision.barcode_is_sticker and verify_decision.confidence >= sticker_min_confidence:
                barcode_value = verify_decision.barcode_value
                barcode_is_sticker = True
                barcode_position_hint = verify_decision.barcode_position_hint
            else:
                barcode_value = ""
                barcode_is_sticker = False
                barcode_position_hint = verify_decision.barcode_position_hint

    # Pass 2: thorough recovery across full page + cropped regions.
    if not barcode_value:
        recovery_images = _build_recovery_images(image_path)
        try:
            for region_name, region_image, cleanup in recovery_images:
                recover_decision, recover_error, in_tok, out_tok = _run_sticker_decision(
                    image_path=region_image,
                    prompt=_build_barcode_recovery_prompt(),
                    client=client,
                    model=model,
                    context_label=f"barcode recovery {region_name} {pdf_path.name}",
                    max_retries=max_retries,
                )
                extra_input_tokens += in_tok
                extra_output_tokens += out_tok

                if recover_decision is None or not recover_decision.barcode_is_sticker:
                    continue
                if (best_recovery is None) or (recover_decision.confidence > best_recovery.confidence):
                    best_recovery = recover_decision
                if recover_decision.confidence >= max(sticker_min_confidence + 0.15, 0.85):
                    break
        finally:
            for _name, path, cleanup in recovery_images:
                if cleanup:
                    path.unlink(missing_ok=True)

        if best_recovery is not None and best_recovery.confidence >= sticker_min_confidence:
            barcode_value = best_recovery.barcode_value
            barcode_is_sticker = True
            barcode_position_hint = best_recovery.barcode_position_hint

    # Pass 3: high-DPI recovery for difficult pages.
    if not barcode_value and pdf_dpi < _BOOKLET_STICKER_RECOVERY_DPI:
        hi_dpi_image: Path | None = None
        try:
            hi_dpi_image = _render_pdf_first_page_image(pdf_path, dpi=_BOOKLET_STICKER_RECOVERY_DPI)
            hd_images = _build_recovery_images(hi_dpi_image)
            try:
                for region_name, region_image, cleanup in hd_images:
                    recover_hd_decision, _err, in_tok, out_tok = _run_sticker_decision(
                        image_path=region_image,
                        prompt=_build_barcode_recovery_prompt(),
                        client=client,
                        model=model,
                        context_label=f"barcode high-dpi recovery {region_name} {pdf_path.name}",
                        max_retries=max_retries,
                    )
                    extra_input_tokens += in_tok
                    extra_output_tokens += out_tok
                    if recover_hd_decision is None or not recover_hd_decision.barcode_is_sticker:
                        continue
                    if recover_hd_decision.confidence >= sticker_min_confidence:
                        barcode_value = recover_hd_decision.barcode_value
                        barcode_is_sticker = True
                        barcode_position_hint = recover_hd_decision.barcode_position_hint
                        break
            finally:
                for _name, path, cleanup in hd_images:
                    if cleanup:
                        path.unlink(missing_ok=True)
        finally:
            if hi_dpi_image is not None:
                hi_dpi_image.unlink(missing_ok=True)

    updated_entry = entry.model_copy(
        update={
            "barcode_value": barcode_value,
            "barcode_is_sticker": barcode_is_sticker,
            "barcode_position_hint": barcode_position_hint,
        }
    )
    return updated_entry, extra_input_tokens, extra_output_tokens


def _score_roll_confidence(value: str, source: str) -> float:
    if not value:
        return 0.0
    score = 0.55
    if value.isdigit():
        score += 0.15
    if len(value) in _BOOKLET_ROLL_LENGTH_HINTS:
        score += 0.2
    elif 7 <= len(value) <= 14:
        score += 0.08
    if source != "gemini":
        score -= 0.3
    return _clamp_confidence(score)


def _score_barcode_confidence(value: str, source: str, barcode_is_sticker: bool) -> float:
    if not value:
        return 0.0
    if not barcode_is_sticker:
        return 0.0
    score = 0.5
    if 8 <= len(value) <= 20:
        score += 0.2
    if value.isalnum():
        score += 0.1
    if value.isdigit():
        score += 0.1
    if source != "gemini_sticker":
        score -= 0.32
    return _clamp_confidence(score)


def _apply_booklet_fallbacks(entry: BookletEntry, pdf_path: Path) -> BookletEntry:
    roll_number = _extract_digits(entry.roll_number.strip())
    barcode_value = _extract_alnum_upper(entry.barcode_value.strip())
    roll_source = "gemini"
    barcode_source = "gemini_sticker"

    stem_digits = _extract_digits(pdf_path.stem)

    if not roll_number and stem_digits:
        roll_number = stem_digits
        roll_source = "filename_digits"

    barcode_is_sticker = bool(entry.barcode_is_sticker)
    barcode_position_hint = entry.barcode_position_hint.strip().lower() or "unknown"
    if barcode_position_hint not in {"right", "left", "center", "unknown"}:
        barcode_position_hint = "unknown"

    review_required = False
    review_notes: list[str] = []

    # Reject non-sticker (or uncertain) barcode values.
    if not barcode_is_sticker:
        barcode_value = ""
        barcode_source = "missing_no_sticker"
        review_required = True
        review_notes.append("No barcode sticker detected; please review this booklet manually.")

    if not roll_number:
        review_required = True
        review_notes.append("Roll number missing; verify handwritten roll number manually.")

    review_note = " ".join(review_notes).strip()

    return BookletEntry(
        roll_number=roll_number,
        barcode_value=barcode_value,
        roll_confidence=_score_roll_confidence(roll_number, roll_source),
        barcode_confidence=_score_barcode_confidence(barcode_value, barcode_source, barcode_is_sticker),
        roll_source=roll_source,
        barcode_source=barcode_source,
        barcode_is_sticker=barcode_is_sticker,
        barcode_position_hint=barcode_position_hint,
        review_required=review_required,
        review_note=review_note,
        source_pdf=pdf_path.name,
    )


def analyze_booklet_folder_patterns(folder_path: Path) -> dict:
    """
    Analyze booklet PDFs without API calls to understand extraction patterns.
    Useful for deciding fallback strategy before running OCR.
    """
    pdf_files = sorted(Path(folder_path).glob("*.pdf"))
    if not pdf_files:
        return {
            "total_pdfs": 0,
            "single_page_pdfs": 0,
            "filename_digit_length_distribution": {},
            "numeric_filename_ratio": 0.0,
            "files_with_text_layer": 0,
            "files_without_text_layer": 0,
            "file_details": [],
        }

    single_page_pdfs = 0
    files_with_text_layer = 0
    numeric_filename_count = 0
    filename_lengths: Counter[int] = Counter()
    file_details: list[dict] = []

    for pdf_path in pdf_files:
        stem = pdf_path.stem
        stem_digits = _extract_digits(stem)
        if stem_digits and stem_digits == stem:
            numeric_filename_count += 1
        filename_lengths[len(stem_digits or stem)] += 1

        page_count = 0
        first_page_text_chars = 0
        first_page_has_text = False
        error = ""

        try:
            doc = fitz.open(pdf_path)
            page_count = doc.page_count
            if page_count == 1:
                single_page_pdfs += 1
            if page_count > 0:
                first_page_text = (doc[0].get_text("text") or "").strip()
                first_page_text_chars = len(first_page_text)
                first_page_has_text = bool(first_page_text_chars)
                if first_page_has_text:
                    files_with_text_layer += 1
        except Exception as exc:
            error = str(exc)

        file_details.append(
            {
                "pdf": pdf_path.name,
                "filename_digits": stem_digits,
                "filename_digits_len": len(stem_digits),
                "pages": page_count,
                "first_page_text_chars": first_page_text_chars,
                "first_page_has_text_layer": first_page_has_text,
                "error": error,
            }
        )

    total_pdfs = len(pdf_files)
    return {
        "total_pdfs": total_pdfs,
        "single_page_pdfs": single_page_pdfs,
        "filename_digit_length_distribution": dict(sorted(filename_lengths.items())),
        "numeric_filename_ratio": round(numeric_filename_count / total_pdfs, 4),
        "files_with_text_layer": files_with_text_layer,
        "files_without_text_layer": total_pdfs - files_with_text_layer,
        "file_details": file_details,
    }


def _extract_single_booklet_page(
    *,
    image_path: Path,
    page_number: int,
    client: Client,
    model: str,
    max_retries: int = _BOOKLET_RATE_LIMIT_MAX_RETRIES,
) -> tuple[BookletEntry, str | None, float, int, int]:
    """Extract roll number and barcode from a single booklet page."""
    started_at = time.perf_counter()
    raw_json, error, input_tokens, output_tokens = _generate_json_with_retry(
        image_path=image_path,
        prompt=_build_booklet_prompt(),
        schema=BookletEntry.model_json_schema(),
        client=client,
        model=model,
        context_label=f"booklet page {page_number}",
        max_retries=max_retries,
    )
    if error:
        return BookletEntry(roll_number="", barcode_value=""), error, time.perf_counter() - started_at, 0, 0

    try:
        parsed = BookletEntry.model_validate_json(raw_json or "{}")
    except ValidationError as exc:
        error_msg = f"Invalid JSON for page {page_number}: {exc}"
        return BookletEntry(roll_number="", barcode_value=""), error_msg, time.perf_counter() - started_at, 0, 0

    elapsed = time.perf_counter() - started_at
    return parsed, None, elapsed, input_tokens, output_tokens


def _process_single_booklet_pdf(
    *,
    pdf_path: Path,
    client: Client,
    model: str,
    pdf_dpi: int,
    max_workers: int,
    max_retries: int,
    sticker_min_confidence: float,
) -> tuple[BookletEntry | None, str | None]:
    file_started_at = time.perf_counter()
    page_images: list[tuple[int, Path]] = []
    try:
        page_images = _pdf_to_page_images(pdf_path, dpi=pdf_dpi, max_workers=max_workers)
        if not page_images:
            return None, f"{pdf_path.name}: No pages found"

        page_number, image_path = page_images[0]
        entry, error, _ocr_elapsed, input_tokens, output_tokens = _extract_single_booklet_page(
            image_path=image_path,
            page_number=page_number,
            client=client,
            model=model,
            max_retries=max_retries,
        )
        if error:
            return None, f"{pdf_path.name}: {error}"

        refined_entry, extra_in_tokens, extra_out_tokens = _refine_sticker_barcode(
            image_path=image_path,
            pdf_path=pdf_path,
            pdf_dpi=pdf_dpi,
            entry=entry,
            client=client,
            model=model,
            max_retries=max_retries,
            sticker_min_confidence=sticker_min_confidence,
        )
        input_tokens += extra_in_tokens
        output_tokens += extra_out_tokens

        normalized_entry = _apply_booklet_fallbacks(refined_entry, pdf_path)
        file_elapsed = round(time.perf_counter() - file_started_at, 3)
        normalized_entry = normalized_entry.model_copy(
            update={
                "processing_seconds": file_elapsed,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "estimated_cost_usd": _estimate_cost_usd(input_tokens, output_tokens),
            }
        )
        return normalized_entry, None
    except Exception as exc:
        return None, f"{pdf_path.name}: {exc}"
    finally:
        for _, path in page_images:
            path.unlink(missing_ok=True)


def extract_booklet_from_folder(
    *,
    folder_path: Path,
    api_key: str,
    model: str = "gemini-2.5-flash-lite",
    pdf_dpi: int = DEFAULT_PDF_DPI,
    max_workers: int = 4,
    rate_limit_retries: int = _BOOKLET_RATE_LIMIT_MAX_RETRIES,
    sticker_min_confidence: float = _BOOKLET_STICKER_MIN_CONFIDENCE,
) -> BookletScanResponse:
    """
    Extract roll numbers and barcodes from all PDFs in a folder.

    Args:
        folder_path: Path to folder containing PDF files
        api_key: Gemini API key
        model: Gemini model name
        pdf_dpi: DPI for PDF rendering
        max_workers: Parallel workers
        rate_limit_retries: Number of retries with backoff on 429/rate-limit errors
        sticker_min_confidence: Confidence threshold (0..1) to accept detected sticker barcode

    Returns:
        BookletScanResponse with all extracted entries
    """
    started_at = time.perf_counter()

    # Find all PDF files in folder
    pdf_files = sorted(Path(folder_path).glob("*.pdf"))
    if not pdf_files:
        return BookletScanResponse(
            errors=[f"No PDF files found in {folder_path}"],
        )

    client = _get_shared_client(api_key)
    entries: list[BookletEntry] = []
    errors: list[str] = []
    successful = 0
    failed = 0

    # Process PDFs in parallel to maximize throughput.
    workers = max(1, min(max_workers, len(pdf_files)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _process_single_booklet_pdf,
                pdf_path=pdf_path,
                client=client,
                model=model,
                pdf_dpi=pdf_dpi,
                max_workers=max_workers,
                max_retries=rate_limit_retries,
                sticker_min_confidence=max(0.0, min(sticker_min_confidence, 1.0)),
            ): pdf_path.name
            for pdf_path in pdf_files
        }
        for future in as_completed(futures):
            try:
                entry, error = future.result()
                if error:
                    errors.append(error)
                    failed += 1
                elif entry is not None:
                    entries.append(entry)
                    successful += 1
                else:
                    errors.append(f"{futures[future]}: unknown processing failure")
                    failed += 1
            except Exception as exc:
                errors.append(f"{futures[future]}: {exc}")
                failed += 1

    entries.sort(key=lambda e: e.source_pdf)

    elapsed_seconds = round(time.perf_counter() - started_at, 3)
    avg_seconds_per_file = round(elapsed_seconds / len(pdf_files), 3) if pdf_files else 0.0
    total_input_tokens = sum(e.input_tokens for e in entries)
    total_output_tokens = sum(e.output_tokens for e in entries)
    estimated_total_cost_usd = round(sum(e.estimated_cost_usd for e in entries), 8)

    return BookletScanResponse(
        entries=entries,
        total_pages=len(pdf_files),
        successful=successful,
        failed=failed,
        elapsed_seconds=elapsed_seconds,
        avg_seconds_per_file=avg_seconds_per_file,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        estimated_total_cost_usd=estimated_total_cost_usd,
        errors=errors,
    )
