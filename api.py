from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
from difflib import SequenceMatcher

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Always load .env next to this file (uvicorn cwd is often not the project folder).
_PROJECT_DIR = Path(__file__).resolve().parent
_DOTENV_PATH = _PROJECT_DIR / ".env"

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # safety if dotenv not installed
    def load_dotenv(dotenv_path=None, *args, **kwargs):
        path = Path(dotenv_path) if dotenv_path else _DOTENV_PATH
        try:
            # utf-8-sig strips UTF-8 BOM so keys are not "\ufeffGEMINI_API_KEY"
            with open(path, encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ[k.strip()] = v.strip().strip("'").strip('"')
        except Exception:
            pass

# Load .env before other local imports (and use utf-8-sig / override so BOM and
# empty inherited env vars do not block GEMINI_API_KEY).
load_dotenv(_DOTENV_PATH, override=True, encoding="utf-8-sig")


def _sync_gemini_key_from_dotenv_file() -> None:
    """If GEMINI_API_KEY is still missing, parse it directly from .env (BOM-safe).

    Covers cases where python-dotenv does not run as expected (e.g. some
    workers, packaged layouts, or empty inherited GEMINI_API_KEY quirks).
    """
    if (os.getenv("GEMINI_API_KEY") or "").strip():
        return
    if not _DOTENV_PATH.is_file():
        return
    try:
        text = _DOTENV_PATH.read_text(encoding="utf-8-sig")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, val = line.partition("=")
        if name.strip() != "GEMINI_API_KEY":
            continue
        val = val.strip().strip("'\"")
        if val:
            os.environ["GEMINI_API_KEY"] = val
        break


def require_gemini_api_key() -> str:
    """Return a non-empty GEMINI_API_KEY, re-syncing from .env on each call."""
    _sync_gemini_key_from_dotenv_file()
    key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not key:
        raise HTTPException(
            status_code=500,
            detail=(
                "GEMINI_API_KEY is not set on the server. "
                f"Set GEMINI_API_KEY in {_DOTENV_PATH} or in the environment, then restart uvicorn."
            ),
        )
    return key


_sync_gemini_key_from_dotenv_file()

import ocr_service

extract_ocr_from_path = ocr_service.extract_ocr_from_path
DEFAULT_MAX_WORKERS = ocr_service.DEFAULT_MAX_WORKERS
DEFAULT_PDF_DPI = ocr_service.DEFAULT_PDF_DPI
DEFAULT_BATCH_SIZE = ocr_service.DEFAULT_BATCH_SIZE

app = FastAPI(
    title="Gemini OCR API",
    description="Simple HTTP API wrapper around the Gemini OCR PoC, with keyword matching and simple field verification.",
    version="1.1.0",
)

# Allow cross-origin requests from the frontend (development convenience)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    api_key = require_gemini_api_key()

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


def _normalize_spaces(text: str) -> str:
    """Remove whitespace and lowercase for robust matching."""
    return "".join(ch for ch in text.lower() if not ch.isspace())


def _normalize_alnum_upper(text: str) -> str:
    """
    Keep only alphanumeric characters and uppercase them.
    Used for strict ID/code comparison (PAN, GSTIN, CIN, UDYAM numbers).
    """
    return "".join(ch for ch in text.upper() if ch.isalnum())


def _normalize_strict_text(text: str) -> str:
    """
    Strict text normalisation:
    - lowercase
    - collapse multiple spaces to single
    - DO NOT remove punctuation or dots

    This ensures that even a missing dot/character will make the match fail.
    """
    return " ".join(text.lower().split())


def _all_words_present(expected: str, text: str) -> float:
    """
    Check if all significant words from `expected` are present in `text`
    (case-insensitive, order/line-breaks ignored).

    Returns a ratio [0..1] = matched_words / total_words.
    1.0 means every word from expected was found at least once in text.
    """
    import re

    expected_words = [w for w in re.findall(r"[a-zA-Z0-9]+", expected.lower()) if w]
    if not expected_words:
        return 0.0
    # IMPORTANT: use token/word matching, not substring matching.
    # Substring matching can give false positives, e.g. expected word "AN"
    # matching inside "BANK". We need exact word presence.
    text_words = set(re.findall(r"[a-zA-Z0-9]+", (text or "").lower()))
    matched = sum(1 for w in expected_words if w in text_words)
    return matched / len(expected_words)


def _levenshtein_distance(a: str, b: str) -> int:
    """Simple Levenshtein distance for short strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    previous_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current_row = [i]
        for j, cb in enumerate(b, start=1):
            insertions = previous_row[j] + 1
            deletions = current_row[j - 1] + 1
            substitutions = previous_row[j - 1] + (ca != cb)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def _best_fuzzy_substring_match(
    needle: str,
    haystack: str,
    *,
    max_distance: int = 1,
) -> Tuple[bool, float]:
    """
    Search for `needle` inside `haystack` allowing a small edit distance.

    Returns (matched, similarity_ratio[0..1]).
    """
    needle = needle.strip()
    haystack = haystack.strip()
    if not needle or not haystack:
        return False, 0.0

    n = len(needle)
    if n == 0:
        return False, 0.0

    best_ratio = 0.0
    matched = False
    if len(haystack) < n:
        dist = _levenshtein_distance(needle, haystack)
        ratio = 1.0 - dist / max(len(needle), len(haystack))
        return (dist <= max_distance), ratio

    for i in range(0, len(haystack) - n + 1):
        window = haystack[i : i + n]
        dist = _levenshtein_distance(needle, window)
        if dist <= max_distance:
            matched = True
        ratio = 1.0 - dist / max(len(needle), len(window))
        if ratio > best_ratio:
            best_ratio = ratio
    return matched, best_ratio


def _fuzzy_name_match(expected: str, text: str, *, threshold: float = 0.75) -> float:
    """
    Fuzzy match for names: normalize whitespace and compare similarity.

    Returns similarity ratio [0..1].
    """
    if not expected or not text:
        return 0.0
    expected_norm = " ".join(expected.strip().lower().split())
    text_norm = " ".join(text.strip().lower().split())
    # Direct ratio over whole text
    ratio_full = SequenceMatcher(None, expected_norm, text_norm).ratio()
    if ratio_full >= threshold:
        return ratio_full

    # Sliding window over tokens for better locality
    tokens = text_norm.split()
    target_len = len(expected_norm.split())
    if target_len == 0 or len(tokens) < target_len:
        return ratio_full

    best = ratio_full
    for i in range(0, len(tokens) - target_len + 1):
        window = " ".join(tokens[i : i + target_len])
        r = SequenceMatcher(None, expected_norm, window).ratio()
        if r > best:
            best = r
    return best


@app.post("/ocr/bank-verify")
async def ocr_bank_verify(
    file: UploadFile = File(..., description="Bank proof document (PDF or image)"),
    account_number: str = Form(...),
    ifsc_code: str = Form(...),
    bank_holder_name: Optional[str] = Form(None),
    bank_name: Optional[str] = Form(None),
    branch: Optional[str] = Form(None),
    model: str = Form("gemini-2.5-flash-lite"),
    chunk_size: int = Form(1200),
) -> JSONResponse:
    """
    Run OCR on the uploaded bank document and verify that the extracted text
    contains the provided account details.

    A field is considered matched if its value (after basic normalisation) is
    found anywhere in the OCR'd text.
    """
    api_key = require_gemini_api_key()

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

    full_text = ocr_result.full_text or ""
    full_text_norm = _normalize_spaces(full_text)
    full_text_alnum_upper = _normalize_alnum_upper(full_text)

    def match_account(num: str) -> bool:
        # Strict digits-only account number match (ignore spaces)
        target = "".join(ch for ch in str(num) if ch.isdigit())
        haystack = "".join(ch for ch in (full_text or "") if ch.isdigit())
        return bool(target) and target in haystack

    def match_ifsc(code: str) -> bool:
        # IFSC: 11 chars => 4 letters + '0' + 6 digits
        # OCR commonly confuses: O<->0, I/L<->1, S<->5, B<->8, Z<->2 (mostly in digit part)
        # Make IFSC matching robust while remaining strict on structure.
        import re

        raw_target = _normalize_alnum_upper(code)
        if not raw_target:
            return False

        def _ocr_safe_ifsc(value: str) -> str:
            v = _normalize_alnum_upper(value)
            if len(v) != 11:
                return ""
            # Normalize the mandatory 5th character to "0" when OCR outputs "O"
            v = v[:4] + ("0" if v[4] in {"0", "O"} else v[4]) + v[5:]

            # Digit-part normalization for OCR confusions (positions 6..11)
            digit_map = {
                "O": "0",
                "I": "1",
                "L": "1",
                "Z": "2",
                "S": "5",
                "B": "8",
            }
            suffix = "".join(digit_map.get(ch, ch) for ch in v[5:])
            return v[:5] + suffix

        target = _ocr_safe_ifsc(raw_target)
        if not target:
            return False

        # Build OCR-safe haystack (for continuous text)
        hay = _ocr_safe_ifsc(full_text_alnum_upper[:11])  # placeholder to reuse function shape
        # Instead of slicing, normalize the whole OCR text with the same digit-map rules,
        # then look for target as a substring.
        digit_map_global = {
            "O": "0",
            "I": "1",
            "L": "1",
            "Z": "2",
            "S": "5",
            "B": "8",
        }
        ocr_alnum = full_text_alnum_upper
        if not ocr_alnum:
            return False
        # Normalize any 'O' that appears right after 4 letters (pattern: [A-Z]{4}O)
        ocr_alnum = re.sub(r"([A-Z]{4})O", r"\g<1>0", ocr_alnum)
        ocr_alnum = "".join(digit_map_global.get(ch, ch) for ch in ocr_alnum)
        if target in ocr_alnum:
            return True

        # Also check token candidates of length 11 (more precise)
        tokens = re.findall(r"[A-Z0-9]{11}", ocr_alnum)
        return any(tok == target for tok in tokens)

    def match_simple(value: Optional[str]) -> bool:
        if not value:
            return False
        return value.strip().lower() in full_text.lower()

    # Individual field matches
    account_match = match_account(account_number)
    ifsc_match = match_ifsc(ifsc_code)

    # For passbook verification: these must match strictly at word level.
    # If any single word is missing, treat as mismatch.
    holder_ratio = _all_words_present(bank_holder_name or "", full_text) if bank_holder_name else 0.0
    holder_match = bool(bank_holder_name) and holder_ratio >= 1.0

    bank_ratio = _all_words_present(bank_name or "", full_text) if bank_name else 0.0
    bank_name_match = bool(bank_name) and bank_ratio >= 1.0

    branch_ratio = _all_words_present(branch or "", full_text) if branch else 0.0
    branch_match = bool(branch) and branch_ratio >= 1.0

    field_matches = {
        "accountNumber": {
            "value": account_number,
            "matched": account_match,
        },
        "ifscCode": {
            "value": ifsc_code,
            "matched": ifsc_match,
        },
        "bankHolderName": {
            "value": bank_holder_name,
            "matched": holder_match,
            "wordMatchRatio": round(holder_ratio, 3) if bank_holder_name else None,
            "requireAllWords": True if bank_holder_name else None,
        },
        "bankName": {
            "value": bank_name,
            "matched": bank_name_match,
            "wordMatchRatio": round(bank_ratio, 3) if bank_name else None,
            "requireAllWords": True if bank_name else None,
        },
        "bankBranch": {
            "value": branch,
            "matched": branch_match,
            "wordMatchRatio": round(branch_ratio, 3) if branch else None,
            "requireAllWords": True if branch else None,
        },
    }

    # Passbook rule (as requested):
    # verified ONLY if ALL fields match:
    # - account number
    # - IFSC
    # - bank holder name
    # - bank name
    # - branch
    verified = account_match and ifsc_match and holder_match and bank_name_match and branch_match

    response_body = {
        "verified": verified,
        "fields": field_matches,
        "ocrPreviewChars": len(full_text),
    }

    return JSONResponse(response_body)


@app.post("/ocr/credit-card-verify")
async def ocr_credit_card_verify(
    file: UploadFile = File(..., description="Credit card statement (PDF or image)"),
    card_number: str = Form(...),
    holder_name: str = Form(...),
    bank_name: Optional[str] = Form(None),
    branch: Optional[str] = Form(None),
    model: str = Form("gemini-2.5-flash-lite"),
    chunk_size: int = Form(1200),
) -> JSONResponse:
    """
    Run OCR on the uploaded credit card statement and verify that the extracted text
    contains the provided card details (card number, bank name, optional branch/name).
    """
    api_key = require_gemini_api_key()

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

    full_text = ocr_result.full_text or ""
    full_text_norm = _normalize_spaces(full_text)

    def match_card(num: str) -> bool:
        value_norm = _normalize_spaces(num)
        return bool(value_norm) and value_norm in full_text_norm

    def match_simple(value: Optional[str]) -> bool:
        if not value:
            return False
        return value.strip().lower() in full_text.lower()

    card_match = match_card(card_number)
    bank_match = match_simple(bank_name)
    branch_match = match_simple(branch)
    # Holder name must be STRICT at word level:
    # - If any one word from expected holder_name is missing in OCR text,
    #   we consider it NOT matched.
    # - This matches the requirement: "ek word bhi missing hai to match na ho".
    holder_word_ratio = _all_words_present(holder_name, full_text)
    holder_match = (holder_word_ratio >= 1.0)

    field_matches = {
        "creditCardNumber": {"value": card_number, "matched": card_match},
        "bankName": {
            "value": bank_name,
            "matched": bank_match if bank_name else None,
        },
        "bankBranch": {
            "value": branch,
            "matched": branch_match if branch else None,
        },
        "cardHolderName": {
            "value": holder_name,
            "matched": holder_match if holder_name else False,
            "wordMatchRatio": round(holder_word_ratio, 3) if holder_name else None,
            "requireAllWords": True if holder_name else None,
        },
    }

    # Required for verification:
    # - card number must match
    # - holder name must be provided AND must match (strict all-words)
    #
    # Bank name and branch are informational-only for now (do not block),
    # but are still surfaced in the response.
    verified = card_match and holder_match

    response_body = {
        "verified": verified,
        "fields": field_matches,
        "ocrPreviewChars": len(full_text),
    }

    return JSONResponse(response_body)


@app.post("/ocr/pan-verify")
async def ocr_pan_verify(
    file: UploadFile = File(..., description="PAN card document (PDF or image)"),
    pan_number: str = Form(...),
    pan_name: str = Form(...),
    model: str = Form("gemini-2.5-flash-lite"),
    chunk_size: int = Form(1200),
) -> JSONResponse:
    """
    Run OCR on the uploaded PAN card document and verify that the extracted text
    contains both the PAN number and PAN name.
    
    PAN is verified only if both PAN number and PAN name match in the document.
    """
    api_key = require_gemini_api_key()

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

    full_text = ocr_result.full_text or ""
    full_text_norm = _normalize_spaces(full_text)

    def match_pan_number(num: str) -> bool:
        """
        STRICT PAN number match.
        - Normalise to alnum+upper (spaces and dashes ignored)
        - Require exact substring match, no edit distance allowed.
        """
        target = _normalize_alnum_upper(num)
        haystack = _normalize_alnum_upper(full_text)
        return bool(target) and target in haystack

    def match_pan_name(name: str) -> bool:
        """
        STRICT PAN name match.
        - Lowercase + collapse spaces
        - Punctuation and dots must be identical to count as match.
        """
        if not name:
            return False
        target = _normalize_strict_text(name)
        text_norm = _normalize_strict_text(full_text)
        return bool(target) and target in text_norm

    # Individual field matches (strict)
    pan_number_ok = match_pan_number(pan_number)
    pan_name_ok = match_pan_name(pan_name)

    field_matches = {
        "panNumber": {
            "value": pan_number,
            "matched": pan_number_ok,
        },
        "panName": {
            "value": pan_name,
            "matched": pan_name_ok,
        },
    }

    # Overall verification: both PAN number and PAN name must match
    verified = pan_number_ok and pan_name_ok

    response_body = {
        "verified": verified,
        "fields": field_matches,
        "ocrPreviewChars": len(full_text),
    }

    return JSONResponse(response_body)


@app.post("/ocr/udyam-verify")
async def ocr_udyam_verify(
    file: UploadFile = File(..., description="UDYAM / MSME registration certificate"),
    udyam_number: str = Form(...),
    enterprise_name: str = Form(...),
    enterprise_type: Optional[str] = Form(None),
    model: str = Form("gemini-2.5-flash-lite"),
    chunk_size: int = Form(1200),
) -> JSONResponse:
    """
    Verify UDYAM (MSME) certificate details using fuzzy OCR matching.
    """
    api_key = require_gemini_api_key()

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

    full_text = ocr_result.full_text or ""

    # UDYAM number – STRICT alnum match (no edit distance)
    udyam_target = _normalize_alnum_upper(udyam_number)
    udyam_haystack = _normalize_alnum_upper(full_text)
    udyam_ok = bool(udyam_target) and udyam_target in udyam_haystack

    # Enterprise name – STRICT name match
    name_target = _normalize_strict_text(enterprise_name)
    text_norm = _normalize_strict_text(full_text)
    name_ok = bool(name_target) and name_target in text_norm

    # Enterprise type – simple substring, optional
    if enterprise_type:
        ent_type_target = _normalize_strict_text(enterprise_type)
        ent_type_ok = bool(ent_type_target) and ent_type_target in text_norm
    else:
        ent_type_ok = None

    fields = {
        "udyamNumber": {
            "value": udyam_number,
            "matched": udyam_ok,
        },
        "enterpriseName": {
            "value": enterprise_name,
            "matched": name_ok,
        },
        "enterpriseType": {
            "value": enterprise_type,
            "matched": ent_type_ok,
        },
    }

    verified = udyam_ok and name_ok and (ent_type_ok is not False)

    return JSONResponse(
        {
            "verified": verified,
            "fields": fields,
            "ocrPreviewChars": len(full_text),
        }
    )


@app.post("/ocr/gstin-verify")
async def ocr_gstin_verify(
    file: UploadFile = File(..., description="GST registration certificate or GST screen"),
    gstin: str = Form(...),
    legal_name: str = Form(...),
    trade_name: Optional[str] = Form(None),
    taxpayer_type: Optional[str] = Form(None),
    status_text: Optional[str] = Form(None),
    model: str = Form("gemini-2.5-flash-lite"),
    chunk_size: int = Form(1200),
) -> JSONResponse:
    """
    Verify GSTIN details (GST number + names + optional fields) with fuzzy matching.
    """
    api_key = require_gemini_api_key()

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

    full_text = ocr_result.full_text or ""

    # GSTIN: strict alnum+upper exact match
    gstin_target = _normalize_alnum_upper(gstin)
    gstin_haystack = _normalize_alnum_upper(full_text)
    gstin_ok = bool(gstin_target) and gstin_target in gstin_haystack

    # Names: fuzzy match with slightly relaxed threshold so genuine matches
    # with small OCR glitches (broken characters, minor punctuation) pass,
    # but still high enough to block clearly different names.
    text_norm = " ".join(full_text.lower().split())

    # Word-level check: every word from expected name must appear somewhere
    # in the OCR text (order/line-breaks ignored). This is stricter on content
    # but tolerant to layout differences.
    legal_ratio = _all_words_present(legal_name, full_text)
    legal_ok = legal_ratio >= 1.0

    if trade_name:
        trade_ratio = _all_words_present(trade_name, full_text)
        trade_ok = trade_ratio >= 1.0
    else:
        trade_ratio = 0.0
        trade_ok = None

    def _optional_flag_strict(value: Optional[str]) -> Optional[bool]:
        if not value:
            return None
        target = _normalize_strict_text(value)
        return bool(target) and target in text_norm

    taxpayer_ok = _optional_flag_strict(taxpayer_type)

    # Status: only check if document clearly shows "Active" (case-insensitive).
    # This field does NOT affect overall `verified`, it's informational only.
    if status_text:
        status_ok = "active" in text_norm
    else:
        status_ok = None

    fields = {
        "gstin": {
            "value": gstin,
            "matched": gstin_ok,
        },
        "legalName": {
            "value": legal_name,
            "matched": legal_ok,
            "similarity": round(legal_ratio, 3),
        },
        "tradeName": {
            "value": trade_name,
            "matched": trade_ok,
            "similarity": round(trade_ratio, 3) if trade_name else None,
        },
        "taxpayerType": {
            "value": taxpayer_type,
            "matched": taxpayer_ok,
        },
        "status": {
            "value": status_text,
            "matched": status_ok,
        },
    }

    # Require GSTIN and legal name to pass; trade name is optional if provided.
    # Status does NOT block verification.
    verified = gstin_ok and legal_ok and (trade_ok is not False)

    return JSONResponse(
        {
            "verified": verified,
            "fields": fields,
            "ocrPreviewChars": len(full_text),
        }
    )


@app.post("/ocr/cin-verify")
async def ocr_cin_verify(
    file: UploadFile = File(..., description="CIN / Company master data (MCA)"),
    cin: str = Form(...),
    company_name: str = Form(...),
    state_name: Optional[str] = Form(None),
    status_text: Optional[str] = Form(None),
    model: str = Form("gemini-2.5-flash-lite"),
    chunk_size: int = Form(1200),
) -> JSONResponse:
    """
    Verify CIN (Company Identification Number) details using fuzzy OCR matching.
    """
    api_key = require_gemini_api_key()

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

    full_text = ocr_result.full_text or ""

    cin_target = _normalize_alnum_upper(cin)
    cin_haystack = _normalize_alnum_upper(full_text)
    cin_ok = bool(cin_target) and cin_target in cin_haystack

    text_norm = _normalize_strict_text(full_text)
    name_target = _normalize_strict_text(company_name)
    name_ok = bool(name_target) and name_target in text_norm

    if state_name:
        state_target = _normalize_strict_text(state_name)
        state_ok = bool(state_target) and state_target in text_norm
    else:
        state_ok = None

    if status_text:
        status_target = _normalize_strict_text(status_text)
        status_ok = bool(status_target) and status_target in text_norm
    else:
        status_ok = None

    fields = {
        "cin": {
            "value": cin,
            "matched": cin_ok,
        },
        "companyName": {
            "value": company_name,
            "matched": name_ok,
        },
        "state": {
            "value": state_name,
            "matched": state_ok,
        },
        "status": {
            "value": status_text,
            "matched": status_ok,
        },
    }

    verified = cin_ok and name_ok and (status_ok is not False)

    return JSONResponse(
        {
            "verified": verified,
            "fields": fields,
            "ocrPreviewChars": len(full_text),
        }
    )
