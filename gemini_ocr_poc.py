#!/usr/bin/env python3
"""CLI PoC: OCR + text extraction using Gemini 2.5 Flash-Lite with performance optimizations."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ocr_service import (
    extract_ocr_from_path,
    quality_report,
    clear_cache,
    get_cache_size,
    DEFAULT_MAX_WORKERS,
    DEFAULT_PDF_DPI,
    DEFAULT_BATCH_SIZE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gemini OCR PoC -> structured JSON (with performance optimizations)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Performance Tips:
  --max-workers 8-12    Increase parallelism for multi-page PDFs
  --pdf-dpi 100-150     Lower DPI for faster processing (default: 120)
  --batch-size 3-5      Batch multiple pages per API call
  --no-cache            Disable caching for one-time processing
  --no-async            Disable async for debugging

Examples:
  # Fast processing (lower quality)
  %(prog)s --input doc.pdf --output result.json --pdf-dpi 100 --quality-boost false

  # High quality processing
  %(prog)s --input doc.pdf --output result.json --pdf-dpi 200 --max-workers 4

  # Maximum speed
  %(prog)s --input doc.pdf --output result.json --max-workers 12 --batch-size 5 --pdf-dpi 100
        """,
    )
    parser.add_argument("--input", required=True, help="Path to image/PDF file")
    parser.add_argument("--output", required=True, help="Path to write JSON output")
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash-lite",
        help="Gemini model name (default: gemini-2.5-flash-lite)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        help="Approx target characters per RAG chunk (default: 1200)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Parallel workers for PDF page OCR (default: {DEFAULT_MAX_WORKERS})",
    )
    parser.add_argument(
        "--pdf-dpi",
        type=int,
        default=DEFAULT_PDF_DPI,
        help=f"PDF render DPI before OCR (default: {DEFAULT_PDF_DPI}, lower is faster)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Pages per batch API call, 0 to disable (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Gemini API key (optional). If omitted, uses GEMINI_API_KEY env var.",
    )
    parser.add_argument(
        "--no-quality-boost",
        action="store_true",
        help="Disable aggressive quality post-processing (enabled by default).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable result caching.",
    )
    parser.add_argument(
        "--no-async",
        action="store_true",
        help="Disable async API calls (use thread pool only).",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the OCR result cache before processing.",
    )
    parser.add_argument(
        "--cache-info",
        action="store_true",
        help="Show cache size and exit.",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        help=(
            "Words or phrases to search for in the extracted text. "
            "Provide them as space-separated arguments; wrap multi-word phrases in quotes."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Handle cache info
    if args.cache_info:
        print(f"Cache size: {get_cache_size()} entries")
        return 0

    # Handle clear cache
    if args.clear_cache:
        cleared = clear_cache()
        print(f"Cleared {cleared} cached results")

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_path.exists() or not input_path.is_file():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print(
            "Missing API key. Set GEMINI_API_KEY or pass --api-key.",
            file=sys.stderr,
        )
        return 1

    try:
        parsed = extract_ocr_from_path(
            input_path=input_path,
            api_key=api_key,
            model=args.model,
            chunk_size=args.chunk_size,
            max_workers=args.max_workers,
            pdf_dpi=args.pdf_dpi,
            quality_boost=not args.no_quality_boost,
            batch_size=args.batch_size,
            use_cache=not args.no_cache,
            use_async=not args.no_async,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    normalized = parsed.model_dump()
    report = quality_report(parsed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote OCR JSON: {output_path}")
    print(f"Pages: {len(normalized['pages'])}, Chunks: {len(normalized['chunks'])}")
    print(f"Quality score: {report['score']}/100 ({report['verdict']})")
    
    usage = normalized.get("usage", {})
    if usage:
        cached_str = " (CACHED)" if usage.get("cached") else ""
        print(
            f"Usage{cached_str}: "
            f"{usage.get('input_tokens', 0)} in / "
            f"{usage.get('output_tokens', 0)} out tokens, "
            f"{usage.get('elapsed_seconds', 0)} sec, "
            f"~${usage.get('estimated_cost_usd_paid_tier', 0)} paid-tier estimate"
        )
        
        # Show optimization info
        opts = usage.get("optimizations", {})
        if opts:
            print(
                f"Optimizations: workers={opts.get('parallel_workers')}, "
                f"dpi={opts.get('pdf_dpi')}, batch={opts.get('batch_size')}, "
                f"async={opts.get('async_enabled')}"
            )
    
    # Optional keyword / phrase search across the entire document
    if getattr(args, "keywords", None):
        print("\nKeyword match results:")
        full_text = (normalized.get("full_text") or "").lower()
        pages = [
            (p.get("page_number"), (p.get("text") or "").lower())
            for p in normalized.get("pages", [])
        ]
        for kw in args.keywords:
            kw_lower = kw.lower()
            pages_found = [num for num, text in pages if kw_lower in text]
            found = bool(pages_found) or (kw_lower in full_text)
            if found:
                if pages_found:
                    print(f'  "{kw}": FOUND on pages {pages_found}')
                else:
                    print(f'  "{kw}": FOUND (pages not determined)')
            else:
                print(f'  "{kw}": NOT FOUND')
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
