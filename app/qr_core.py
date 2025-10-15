
import re
from pathlib import Path
from typing import List, Dict, Tuple

import fitz  # PyMuPDF
import numpy as np

try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False

try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    PYZBAR_AVAILABLE = True
except Exception:
    PYZBAR_AVAILABLE = False

URL_REGEX = re.compile(r"(https?://[^\s\"'<>]+)", re.IGNORECASE)

def _np_from_pixmap(pix: fitz.Pixmap) -> np.ndarray:
    if pix.n >= 4:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    data = np.frombuffer(pix.samples, dtype=np.uint8)
    return data.reshape(pix.h, pix.w, 3)

def _extract_urls(texts: List[str]) -> List[str]:
    urls, seen, uniq = [], set(), []
    for t in texts:
        urls.extend(URL_REGEX.findall(t or ""))
    for u in urls:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq

def _opencv_decode_variants(img_rgb: np.ndarray, enable_preproc: bool) -> List[str]:
    if not OPENCV_AVAILABLE:
        return []
    # Use the globally imported cv2
    det = cv2.QRCodeDetector()
    texts: List[str] = []

    def decode_any(arr):
        # multi
        try:
            ok, decoded, _, _ = det.detectAndDecodeMulti(arr)
            if ok and decoded:
                return [s for s in decoded if s]
        except Exception:
            pass
        # single
        try:
            d, _, _ = det.detectAndDecode(arr)
            if d:
                return [d]
        except Exception:
            pass
        return []

    # raw BGR first
    bgr = img_rgb[:, :, ::-1]
    texts += decode_any(bgr)
    if texts:
        return texts

    if enable_preproc:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 33, 2)
        t = decode_any(thr)
        if t:
            return t

        # NEW: CLAHE + Otsu
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        eq = clahe.apply(gray)
        _, otsu = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        t = decode_any(otsu)
        if t:
            return t

        up = cv2.resize(bgr, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
        t = decode_any(up)
        if t:
            return t


    return texts


def _pyzbar_decode(img_rgb: np.ndarray) -> List[str]:
    if not PYZBAR_AVAILABLE:
        return []
    try:
        decoded = pyzbar_decode(img_rgb)
        return [d.data.decode("utf-8", errors="ignore") for d in decoded if d.data]
    except Exception:
        return []

def _urls_from_pdf_text(doc: fitz.Document) -> List[str]:
    urls, seen = [], set()
    for pno in range(len(doc)):
        try:
            text = doc[pno].get_text("text")
        except Exception:
            continue
        for u in URL_REGEX.findall(text or ""):
            if u not in seen:
                seen.add(u)
                urls.append(u)
    return urls

def extract_qr_urls_from_pdf_path(pdf_path: str, dpi: int = 250, preproc: bool = True,
                                  include_text_fallback: bool = True) -> Tuple[List[str], Dict[int, List[str]]]:
    doc = fitz.open(pdf_path)
    all_urls: List[str] = []
    per_page: Dict[int, List[str]] = {}
    try:
        for p in range(len(doc)):
            page = doc[p]
            zoom = dpi / 72.0
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            img_rgb = _np_from_pixmap(pix)

            texts = _opencv_decode_variants(img_rgb, enable_preproc=preproc)
            if not texts:
                texts = _pyzbar_decode(img_rgb)

            page_urls = _extract_urls(texts)
            per_page[p] = page_urls
            for u in page_urls:
                if u not in all_urls:
                    all_urls.append(u)

        if include_text_fallback and not all_urls:
            for u in _urls_from_pdf_text(doc):
                if u not in all_urls:
                    all_urls.append(u)
    finally:
        doc.close()
    return all_urls, per_page

def extract_qr_urls_from_pdf_bytes(pdf_bytes: bytes, dpi: int = 250, preproc: bool = True,
                                   include_text_fallback: bool = True) -> Tuple[List[str], Dict[int, List[str]]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    all_urls: List[str] = []
    per_page: Dict[int, List[str]] = {}
    try:
        for p in range(len(doc)):
            page = doc[p]
            zoom = dpi / 72.0
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            img_rgb = _np_from_pixmap(pix)

            texts = _opencv_decode_variants(img_rgb, enable_preproc=preproc)
            if not texts:
                texts = _pyzbar_decode(img_rgb)

            page_urls = _extract_urls(texts)
            per_page[p] = page_urls
            for u in page_urls:
                if u not in all_urls:
                    all_urls.append(u)

        if include_text_fallback and not all_urls:
            for u in _urls_from_pdf_text(doc):
                if u not in all_urls:
                    all_urls.append(u)
    finally:
        doc.close()
    return all_urls, per_page

def build_response_for_path(pdf_path: str, dpi: int = 250, preproc: bool = True,
                            include_text_fallback: bool = True) -> dict:
    path = Path(pdf_path)
    urls, per_page = extract_qr_urls_from_pdf_path(
        str(path), dpi=dpi, preproc=preproc, include_text_fallback=include_text_fallback
    )
    return {
        "File name": path.name,
        "URL": urls[0] if urls else None,
        "URLs": urls,
        "Count": len(urls),
        "Source": "qr" if urls else ("text" if include_text_fallback else "none"),
        "Details": {str(p): u for p, u in per_page.items()},
        "Status": "QR code available" if urls else "QR code not available",
    }

def build_response_for_bytes(pdf_bytes: bytes, original_filename: str = "upload.pdf",
                             dpi: int = 250, preproc: bool = True,
                             include_text_fallback: bool = True) -> dict:
    urls, per_page = extract_qr_urls_from_pdf_bytes(
        pdf_bytes, dpi=dpi, preproc=preproc, include_text_fallback=include_text_fallback
    )
    return {
        "File name": original_filename,
        "URL": urls[0] if urls else None,
        "URLs": urls,
        "Count": len(urls),
        "Source": "qr" if urls else ("text" if include_text_fallback else "none"),
        "Details": {str(p): u for p, u in per_page.items()},
        "Status": "QR code available" if urls else "QR code not available",
    }

