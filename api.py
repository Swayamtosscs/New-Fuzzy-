from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # safety if dotenv not installed
    def load_dotenv(*args, **kwargs):
        return False

import ocr_service

extract_ocr_from_path = ocr_service.extract_ocr_from_path
DEFAULT_MAX_WORKERS = ocr_service.DEFAULT_MAX_WORKERS
DEFAULT_PDF_DPI = ocr_service.DEFAULT_PDF_DPI
DEFAULT_BATCH_SIZE = ocr_service.DEFAULT_BATCH_SIZE


# Load environment variables from a .env file if present
load_dotenv()

app = FastAPI(
    title="Gemini OCR API",
    description="Simple HTTP API wrapper around the Gemini OCR PoC, with keyword matching.",
    version="1.0.0",
)


@app.post("/ocr/keywords")
async def ocr_with_keywords(
    file: UploadFile = File(..., description="PDF or image file"),
    keywords: Optional[str] = Form(
        default="",
        description="Comma-separated list of words/phrases to search for in the document.",
    ),
    model: str = Form("gemini-2.5-flash-lite"),
    chunk_size: int = Form(1200),
) -> JSONResponse:
    """
    Run OCR on the uploaded file and optionally search for given keywords
    across the entire extracted text.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not set on the server.",
        )

    # Save upload to a temporary file
    suffix = Path(file.filename or "upload.bin").suffix or ".bin"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)

        ocr_result = extract_ocr_from_path(
            input_path=tmp_path,
            api_key=api_key,
            model=model,
            chunk_size=int(chunk_size),
            max_workers=int(DEFAULT_MAX_WORKERS),
            pdf_dpi=int(DEFAULT_PDF_DPI),
            quality_boost=True,
            batch_size=int(DEFAULT_BATCH_SIZE),
            use_cache=True,
            use_async=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            tmp_path.unlink(missing_ok=True)  # type: ignore[name-defined]
        except Exception:
            pass

    payload = ocr_result.model_dump()

    # Keyword / phrase search
    raw_keywords = keywords or ""
    keyword_list: List[str] = [
        k.strip() for k in raw_keywords.split(",") if k.strip()
    ]

    keyword_results = []
    if keyword_list:
        pages_lower = [
            (p.page_number, (p.text or "").lower())
            for p in ocr_result.pages
        ]
        full_text_lower = (ocr_result.full_text or "").lower()

        for kw in keyword_list:
            kw_lower = kw.lower()
            pages_found = [
                num for num, text in pages_lower
                if kw_lower in text
            ]
            found = bool(pages_found) or (kw_lower in full_text_lower)
            keyword_results.append(
                {
                    "keyword": kw,
                    "found": found,
                    "pages": pages_found,
                }
            )

    response_body = {
        "ocr": payload,
        "keywords": {
            "input": keyword_list,
            "results": keyword_results,
        },
    }

    return JSONResponse(response_body)


