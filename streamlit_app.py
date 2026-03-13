from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args, **kwargs):
        return False

import ocr_service

extract_ocr_from_path = ocr_service.extract_ocr_from_path
quality_report = ocr_service.quality_report
clear_cache = ocr_service.clear_cache
get_cache_size = ocr_service.get_cache_size
DEFAULT_MAX_WORKERS = ocr_service.DEFAULT_MAX_WORKERS
DEFAULT_PDF_DPI = ocr_service.DEFAULT_PDF_DPI
DEFAULT_BATCH_SIZE = ocr_service.DEFAULT_BATCH_SIZE
extract_booklet_from_folder = getattr(ocr_service, "extract_booklet_from_folder", None)
analyze_booklet_folder_patterns = getattr(ocr_service, "analyze_booklet_folder_patterns", None)

load_dotenv()

st.set_page_config(page_title="Gemini OCR PoC", page_icon="📄", layout="wide")
st.title("🚀 OCR PoC - Optimized")
st.caption("Upload a document,Now with parallel processing, batching, async API calls, and caching for maximum speed.")

# Sidebar settings
with st.sidebar:
    # st.header("⚙️ Settings")
    # api_key = st.text_input(
    #     "Gemini API Key",
    #     type="password",
    #     value=os.getenv("GEMINI_API_KEY", ""),
    #     help="Stored only in this session.",
    # )
    # model = st.text_input("Model", value="gemini-2.5-flash-lite")
    # chunk_size = st.number_input("Chunk size (chars)", min_value=300, max_value=4000, value=1200, step=100)
    api_key = os.getenv("GEMINI_API_KEY", "")
    model = "gemini-2.5-flash-lite"
    chunk_size = 1200

    st.subheader("⚡ Performance Settings")
    
    # Speed preset buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🐢 Quality", help="High quality, slower processing"):
            st.session_state.max_workers = 4
            st.session_state.pdf_dpi = 200
            st.session_state.batch_size = 0
            st.session_state.quality_boost = True
    with col2:
        if st.button("⚖️ Balanced", help="Balanced speed and quality"):
            st.session_state.max_workers = DEFAULT_MAX_WORKERS
            st.session_state.pdf_dpi = DEFAULT_PDF_DPI
            st.session_state.batch_size = DEFAULT_BATCH_SIZE
            st.session_state.quality_boost = True
    with col3:
        if st.button("🚀 Speed", help="Maximum speed, lower quality"):
            st.session_state.max_workers = 12
            st.session_state.pdf_dpi = 100
            st.session_state.batch_size = 5
            st.session_state.quality_boost = False
    
    # Get values from session state or use defaults
    max_workers = st.slider(
        "Parallel workers", 
        min_value=1, 
        max_value=16, 
        value=st.session_state.get("max_workers", DEFAULT_MAX_WORKERS), 
        step=1,
        help="More workers = faster for multi-page PDFs"
    )
    st.session_state.max_workers = max_workers
    
    pdf_dpi = st.slider(
        "PDF render DPI", 
        min_value=72, 
        max_value=300, 
        value=st.session_state.get("pdf_dpi", DEFAULT_PDF_DPI), 
        step=10,
        help="Lower DPI = faster processing, smaller images"
    )
    st.session_state.pdf_dpi = pdf_dpi
    
    batch_size = st.slider(
        "Batch size (pages/call)", 
        min_value=0, 
        max_value=8, 
        value=st.session_state.get("batch_size", DEFAULT_BATCH_SIZE), 
        step=1,
        help="Pages per API call. 0 = no batching. Higher = fewer API calls."
    )
    st.session_state.batch_size = batch_size
    
    quality_boost = st.checkbox(
        "Aggressive quality boost", 
        value=st.session_state.get("quality_boost", True)
    )
    st.session_state.quality_boost = quality_boost
    
    use_cache = st.checkbox("Enable caching", value=True, help="Cache results for repeated files")
    use_async = st.checkbox("Enable async API", value=True, help="Use async for better concurrency")
    
    st.subheader("📦 Cache")
    cache_col1, cache_col2 = st.columns(2)
    with cache_col1:
        if st.button("Clear Cache"):
            cleared = clear_cache()
            st.success(f"Cleared {cleared} cached results")
    with cache_col2:
        st.metric("Cached", f"{get_cache_size()} entries")

# Top-level tabs: regular OCR and booklet folder scanner are independent.
main_tab_ocr, main_tab_booklet = st.tabs(["📄 OCR Extractor", "📋 Booklet Scanner"])

with main_tab_ocr:
    uploaded = st.file_uploader(
        "Upload PDF or image",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "webp"],
        key="ocr_upload",
    )

    # Let user enter keywords at the same time as PDF upload
    st.markdown("**Optional:** Enter words/phrases to check inside this PDF.")
    keyword_input = st.text_input(
        "Keywords / phrases (comma-separated)",
        value=st.session_state.get("keyword_input", ""),
        help="Example: invoice number, total amount, due date",
        key="keyword_input",
    )

    if uploaded:
        with st.expander("📊 Current Settings Summary", expanded=False):
            settings_col1, settings_col2, settings_col3, settings_col4 = st.columns(4)
            settings_col1.metric("Workers", max_workers)
            settings_col2.metric("DPI", pdf_dpi)
            settings_col3.metric("Batch Size", batch_size if batch_size > 0 else "Off")
            settings_col4.metric("Quality Boost", "On" if quality_boost else "Off")

        if st.button("▶️ Run OCR", type="primary", key="run_ocr"):
            if not api_key:
                st.error("Please set GEMINI_API_KEY in .env (or environment).")
            else:
                suffix = Path(uploaded.name).suffix or ".bin"
                with st.spinner("Processing document with optimized OCR..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded.getvalue())
                        tmp_path = Path(tmp.name)

                    try:
                        ocr_result = extract_ocr_from_path(
                            input_path=tmp_path,
                            api_key=api_key,
                            model=model,
                            chunk_size=int(chunk_size),
                            max_workers=int(max_workers),
                            pdf_dpi=int(pdf_dpi),
                            quality_boost=bool(quality_boost),
                            batch_size=int(batch_size),
                            use_cache=bool(use_cache),
                            use_async=bool(use_async),
                        )
                        st.session_state.ocr_result = ocr_result
                        st.session_state.ocr_report = quality_report(ocr_result)
                        st.session_state.ocr_uploaded_name = uploaded.name

                        # If user provided keywords, run match immediately after OCR
                        raw_keywords = st.session_state.get("keyword_input", "")
                        if raw_keywords:
                            keywords = [k.strip() for k in raw_keywords.split(",") if k.strip()]
                            if keywords:
                                pages_lower = [
                                    (p.page_number, (p.text or "").lower())
                                    for p in ocr_result.pages
                                ]
                                full_text_lower = (ocr_result.full_text or "").lower()

                                keyword_results = []
                                for kw in keywords:
                                    kw_lower = kw.lower()
                                    pages_found = [
                                        num for num, text in pages_lower
                                        if kw_lower in text
                                    ]
                                    found = bool(pages_found) or (kw_lower in full_text_lower)
                                    keyword_results.append({
                                        "Keyword": kw,
                                        "Found": found,
                                        "Pages": ", ".join(str(n) for n in pages_found) if pages_found else "",
                                    })

                                st.session_state.keyword_results = keyword_results
                            else:
                                st.session_state.keyword_results = []
                    except Exception as exc:
                        st.exception(exc)
                    finally:
                        tmp_path.unlink(missing_ok=True)
    else:
        st.info("📤 Upload a file to start OCR.")

    ocr_result = st.session_state.get("ocr_result")
    ocr_report = st.session_state.get("ocr_report")
    ocr_uploaded_name = st.session_state.get("ocr_uploaded_name", "document")

    if ocr_result and ocr_report:
        if ocr_result.cached:
            st.success("✅ OCR complete (from cache)")
        else:
            st.success("✅ OCR complete")

        stats = ocr_report["stats"]
        usage = ocr_result.usage or {}

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Quality", f"{ocr_report['score']}/100")
        c2.metric("Verdict", ocr_report["verdict"])
        c3.metric("Pages", str(stats["pages"]))
        c4.metric("Blocks", str(stats["blocks"]))
        c5.metric("Chunks", str(stats["chunks"]))

        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("Time (s)", f"{usage.get('elapsed_seconds', 0):.2f}")
        p2.metric("Input tokens", str(usage.get("input_tokens", 0)))
        p3.metric("Output tokens", str(usage.get("output_tokens", 0)))
        p4.metric("Est. cost (USD)", f"{usage.get('estimated_cost_usd_paid_tier', 0):.6f}")

        opts = usage.get("optimizations", {})
        mode = ocr_result.extraction_mode
        if mode == "pdf_batch":
            mode_str = f"📦 Batched ({opts.get('batch_size', 0)} pages/call)"
        elif opts.get("async_enabled"):
            mode_str = "⚡ Async parallel"
        else:
            mode_str = f"🔄 Thread pool ({opts.get('parallel_workers', 0)} workers)"
        p5.metric("Mode", mode_str)

        # If we have keyword match results from the run, show a compact summary here
        keyword_results = st.session_state.get("keyword_results")
        if keyword_results is not None:
            total = len(keyword_results)
            found_count = sum(1 for r in keyword_results if r.get("Found"))
            if total > 0:
                st.subheader("🔍 Keyword Match Summary")
                st.write(f"Found {found_count} of {total} keyword(s) in this PDF.")
                st.dataframe(pd.DataFrame(keyword_results), use_container_width=True)

        with st.expander("🔍 Quality checks", expanded=True):
            check_col1, check_col2 = st.columns(2)
            with check_col1:
                st.write({
                    "extraction_mode": stats["extraction_mode"],
                    "total_chars": stats["total_chars"],
                    "total_words": stats["total_words"],
                    "empty_pages": stats["empty_pages"],
                    "avg_chars_per_page": stats["avg_chars_per_page"],
                })
            with check_col2:
                st.write({
                    "avg_words_per_chunk": stats["avg_words_per_chunk"],
                    "suspicious_char_ratio": stats["suspicious_char_ratio"],
                    "duplicate_line_ratio": stats["duplicate_line_ratio"],
                    "block_coverage_ratio": stats["block_coverage_ratio"],
                    "table_blocks": stats["table_blocks"],
                    "key_value_blocks": stats["key_value_blocks"],
                })

            if ocr_report["flags"]:
                for flag in ocr_report["flags"]:
                    st.warning(flag)
            else:
                st.info("✅ No major OCR quality warnings detected.")

        with st.expander("⚡ Optimization Details", expanded=False):
            if opts:
                st.json({
                    "client_reuse": opts.get("client_reuse", False),
                    "parallel_workers": opts.get("parallel_workers", 0),
                    "pdf_dpi": opts.get("pdf_dpi", 0),
                    "batch_size": opts.get("batch_size", 0),
                    "async_enabled": opts.get("async_enabled", False),
                    "cached": stats.get("cached", False),
                })

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            ["📄 Full Text", "📑 Pages", "🧱 Blocks", "🔗 RAG Chunks", "📋 JSON", "🔍 Keyword Search"]
        )

        with tab1:
            st.text_area("Extracted full text", value=ocr_result.full_text, height=420)

        with tab2:
            for page in ocr_result.pages:
                with st.expander(f"Page {page.page_number}", expanded=False):
                    st.text(page.text)

        with tab3:
            for block in ocr_result.blocks:
                title = f"p{block.page_number} | {block.block_type} | {block.block_id}"
                with st.expander(title, expanded=False):
                    if block.title:
                        st.markdown(f"**Title:** {block.title}")
                    if block.text:
                        st.text(block.text)
                    if block.key_values:
                        st.write([kv.model_dump() for kv in block.key_values])

        with tab4:
            for chunk in ocr_result.chunks:
                title = f"{chunk.chunk_id} | pages: {chunk.page_numbers}"
                with st.expander(title, expanded=False):
                    st.text(chunk.text)

        with tab5:
            payload = ocr_result.model_dump()
            st.json(payload)
            st.download_button(
                label="📥 Download JSON",
                data=json.dumps(payload, ensure_ascii=False, indent=2),
                file_name=f"{Path(ocr_uploaded_name).stem}.ocr.json",
                mime="application/json",
            )

        with tab6:
            st.markdown("Enter one or more words/phrases to search across the entire PDF.")
            raw_keywords = st.text_input(
                "Keywords / phrases (comma-separated)",
                value=st.session_state.get("keyword_input", ""),
                help="Example: invoice number, total amount, due date",
                key="keyword_input_tab",
            )

            if raw_keywords:
                # Parse comma-separated values, strip whitespace, drop empties
                keywords = [k.strip() for k in raw_keywords.split(",") if k.strip()]
                if not keywords:
                    st.info("Add at least one non-empty keyword to search.")
                else:
                    pages_lower = [
                        (p.page_number, (p.text or "").lower())
                        for p in ocr_result.pages
                    ]
                    full_text_lower = (ocr_result.full_text or "").lower()

                    results = []
                    for kw in keywords:
                        kw_lower = kw.lower()
                        pages_found = [
                            num for num, text in pages_lower
                            if kw_lower in text
                        ]
                        found = bool(pages_found) or (kw_lower in full_text_lower)
                        results.append({
                            "Keyword": kw,
                            "Found": found,
                            "Pages": ", ".join(str(n) for n in pages_found) if pages_found else "",
                        })

                    st.subheader("Match Results")
                    st.dataframe(pd.DataFrame(results), use_container_width=True)

with main_tab_booklet:
    st.caption("Scan a folder of booklet first-page PDFs to extract roll number and barcode sticker value.")

    if extract_booklet_from_folder is None:
        st.error(
            "Booklet scanner is unavailable: `extract_booklet_from_folder` was not found in `ocr_service.py`."
        )
    else:
        folder_path = st.text_input(
            "PDF Folder Path",
            value="Booklet 1st Pages Pdf",
            help="Path to the folder containing booklet PDFs.",
            key="booklet_folder_path",
        )

        col1, col2 = st.columns(2)
        with col1:
            booklet_dpi = st.selectbox(
                "PDF DPI",
                options=[72, 100, 150, 200],
                index=2,
                help="Higher DPI can improve extraction but is slower.",
                key="booklet_dpi",
            )
        with col2:
            booklet_workers = st.slider(
                "Parallel workers",
                min_value=1,
                max_value=8,
                value=4,
                help="Number of PDFs to process in parallel.",
                key="booklet_workers",
            )

        barcode_conf_threshold = st.slider(
            "Sticker confidence threshold",
            min_value=0.30,
            max_value=0.95,
            value=0.60,
            step=0.05,
            help="Higher value reduces false positives but may miss hard-to-read sticker barcodes.",
            key="barcode_conf_threshold",
        )

        action_col1, action_col2 = st.columns(2)
        with action_col1:
            if st.button("🧪 Analyze Folder Pattern", key="analyze_booklet_pattern"):
                folder = Path(folder_path)
                if not folder.exists():
                    st.error(f"Folder not found: {folder_path}")
                elif analyze_booklet_folder_patterns is None:
                    st.error("Pattern analyzer is unavailable in `ocr_service.py`.")
                else:
                    try:
                        st.session_state.booklet_pattern_report = analyze_booklet_folder_patterns(folder)
                    except Exception as exc:
                        st.exception(exc)

        with action_col2:
            scan_clicked = st.button("🔍 Scan Booklet Folder", type="primary", key="scan_booklets")

        pattern_report = st.session_state.get("booklet_pattern_report")
        if pattern_report:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total PDFs", pattern_report["total_pdfs"])
            m2.metric("Single-page PDFs", pattern_report["single_page_pdfs"])
            m3.metric("Files w/o text layer", pattern_report["files_without_text_layer"])
            m4.metric("Numeric filename ratio", f"{pattern_report['numeric_filename_ratio']:.2%}")

            with st.expander("📐 Filename length distribution", expanded=False):
                st.json(pattern_report["filename_digit_length_distribution"])

            with st.expander("📄 Per-file pattern details", expanded=False):
                st.dataframe(pd.DataFrame(pattern_report["file_details"]), use_container_width=True)

        if scan_clicked:
            if not api_key:
                st.error("Please set GEMINI_API_KEY in .env (or environment).")
            else:
                folder = Path(folder_path)
                if not folder.exists():
                    st.error(f"Folder not found: {folder_path}")
                else:
                    with st.spinner("Scanning booklet folder..."):
                        try:
                            booklet_result = extract_booklet_from_folder(
                                folder_path=folder,
                                api_key=api_key,
                                model=model,
                                pdf_dpi=int(booklet_dpi),
                                max_workers=int(booklet_workers),
                                sticker_min_confidence=float(barcode_conf_threshold),
                            )
                            st.session_state.booklet_result = booklet_result
                        except Exception as exc:
                            st.exception(exc)

        booklet_result = st.session_state.get("booklet_result")
        if booklet_result:
            st.success(f"✅ Scan complete! {booklet_result.successful} successful, {booklet_result.failed} failed")
            t1, t2, t3, t4, t5 = st.columns(5)
            t1.metric("Folder scan time (s)", f"{getattr(booklet_result, 'elapsed_seconds', 0.0):.3f}")
            t2.metric("Avg time per PDF (s)", f"{getattr(booklet_result, 'avg_seconds_per_file', 0.0):.3f}")
            t3.metric("Total input tokens", str(getattr(booklet_result, "total_input_tokens", 0)))
            t4.metric("Total output tokens", str(getattr(booklet_result, "total_output_tokens", 0)))
            t5.metric("Total est. cost (USD)", f"{getattr(booklet_result, 'estimated_total_cost_usd', 0.0):.6f}")

            if booklet_result.entries:
                rows = []
                for entry in booklet_result.entries:
                    rows.append({
                        "Source PDF": getattr(entry, "source_pdf", ""),
                        "Time (s)": getattr(entry, "processing_seconds", 0.0),
                        "Input Tokens": getattr(entry, "input_tokens", 0),
                        "Output Tokens": getattr(entry, "output_tokens", 0),
                        # "Est. Cost (USD)": getattr(entry, "estimated_cost_usd", 0.0),
                        "Roll Number": entry.roll_number,
                        "Roll Confidence": getattr(entry, "roll_confidence", 0.0),
                        # "Roll Source": getattr(entry, "roll_source", "gemini"),
                        "Barcode": entry.barcode_value,
                        "Sticker?": getattr(entry, "barcode_is_sticker", False),
                        "Sticker Position": getattr(entry, "barcode_position_hint", "unknown"),
                        "Barcode Confidence": getattr(entry, "barcode_confidence", 0.0),
                        "Barcode Source": getattr(entry, "barcode_source", "gemini_sticker"),
                        "Review Required": getattr(entry, "review_required", False),
                        "Review Note": getattr(entry, "review_note", ""),
                    })

                booklet_df = pd.DataFrame(rows)
                st.dataframe(booklet_df, use_container_width=True)

                if "Review Required" in booklet_df.columns:
                    review_df = booklet_df[booklet_df["Review Required"] == True]
                    if not review_df.empty:
                        st.warning(f"⚠️ {len(review_df)} file(s) need manual review.")
                        st.dataframe(
                            review_df[["Source PDF", "Roll Number", "Barcode", "Review Note"]],
                            use_container_width=True,
                        )

                csv = booklet_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Booklet CSV",
                    data=csv,
                    file_name="booklet_scan_results.csv",
                    mime="text/csv",
                )

            if booklet_result.errors:
                with st.expander("⚠️ Errors", expanded=False):
                    for err in booklet_result.errors:
                        st.write(err)
