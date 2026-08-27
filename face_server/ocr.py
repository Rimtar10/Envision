"""
Text Reading (OCR) module for the Envision face_server.

Uses PaddleOCR for production recognition (Arabic + English), matching the
FYP proposal's tools-used slide. EasyOCR was used only for early testing and
validation and is intentionally NOT a runtime dependency here.

Models are lazy-loaded and cached at module level (same pattern as
`classifier.py` caching the trained SVM) since PaddleOCR model init is slow
(~seconds) and should only happen once per server process, not per request.
"""

import os
import tempfile

# ── Lazy-loaded PaddleOCR engines ────────────────────────────────────────────
# One engine per language. Arabic and English text often appear on the same
# sign/label, so a single request runs both engines and merges the results
# rather than trying to guess the language up front.
_engines = {}


def _get_engine(lang: str):
    """Lazily construct and cache a PaddleOCR engine for the given language."""
    if lang in _engines:
        return _engines[lang]

    from paddleocr import PaddleOCR

    print(f"[OCR] Loading PaddleOCR engine for lang='{lang}' (first use, may take a moment)...")
    engine = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
    _engines[lang] = engine
    print(f"[OCR] PaddleOCR engine ready for lang='{lang}'")
    return engine


# Minimum confidence to keep a detected text line. Lines below this are
# almost always noise (texture, edges misread as characters).
CONFIDENCE_THRESHOLD = 0.5


def _run_engine(engine, image_path: str, lang_tag: str):
    """Run a single PaddleOCR engine and normalize its output."""
    lines = []
    try:
        result = engine.ocr(image_path, cls=True)
    except Exception as e:
        print(f"[OCR] Engine error ({lang_tag}): {e}")
        return lines

    if not result or result[0] is None:
        return lines

    for detection in result[0]:
        bbox, (text, confidence) = detection
        if confidence < CONFIDENCE_THRESHOLD or not text.strip():
            continue
        # bbox is 4 (x, y) corner points; use the top-left corner's y for
        # reading-order sorting and the top-left x for left/right grouping.
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        lines.append({
            "text": text.strip(),
            "confidence": round(float(confidence), 3),
            "lang": lang_tag,
            "bbox": {
                "left": min(xs), "top": min(ys),
                "right": max(xs), "bottom": max(ys),
            },
        })
    return lines


def _merge_lines(en_lines, ar_lines):
    """
    Merge English and Arabic detections into one reading-order list.

    Some lines will be detected by both engines (e.g. numbers, logos). Since
    we don't have a language classifier in front of OCR, we keep whichever
    detection has higher confidence when two boxes overlap significantly,
    then sort everything top-to-bottom, left-to-right for natural reading
    order when spoken aloud.
    """
    def boxes_overlap(a, b, threshold=0.5):
        left = max(a["left"], b["left"])
        top = max(a["top"], b["top"])
        right = min(a["right"], b["right"])
        bottom = min(a["bottom"], b["bottom"])
        if right <= left or bottom <= top:
            return False
        intersection = (right - left) * (bottom - top)
        area_a = (a["right"] - a["left"]) * (a["bottom"] - a["top"])
        area_b = (b["right"] - b["left"]) * (b["bottom"] - b["top"])
        smaller = min(area_a, area_b) or 1
        return (intersection / smaller) > threshold

    merged = list(en_lines)
    for ar_line in ar_lines:
        duplicate = None
        for i, existing in enumerate(merged):
            if boxes_overlap(ar_line["bbox"], existing["bbox"]):
                duplicate = i
                break
        if duplicate is None:
            merged.append(ar_line)
        elif ar_line["confidence"] > merged[duplicate]["confidence"]:
            merged[duplicate] = ar_line

    merged.sort(key=lambda l: (round(l["bbox"]["top"] / 20), l["bbox"]["left"]))
    return merged


def extract_text_from_bytes(image_bytes: bytes) -> dict:
    """
    Run OCR (Arabic + English) on image bytes sent from the Flutter app.

    Returns:
        {
            "text": "<full text, reading order, newline per line>",
            "lines": [{"text", "confidence", "lang", "bbox"}, ...],
            "status": "success" | "no_text_detected" | "error"
        }
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir=".") as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        en_lines = _run_engine(_get_engine("en"), tmp_path, "en")
        ar_lines = _run_engine(_get_engine("ar"), tmp_path, "ar")
        merged = _merge_lines(en_lines, ar_lines)

        if not merged:
            return {"text": "", "lines": [], "status": "no_text_detected"}

        full_text = "\n".join(l["text"] for l in merged)
        return {"text": full_text, "lines": merged, "status": "success"}

    except Exception as e:
        print(f"[OCR] Error extracting text: {e}")
        return {"text": "", "lines": [], "status": "error", "error": str(e)}

    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def warm_up():
    """Optionally pre-load both engines at server startup to avoid a slow
    first request. Call this from server.py's __main__ block if desired."""
    _get_engine("en")
    _get_engine("ar")
