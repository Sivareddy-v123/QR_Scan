# qr_core.py — Fast & simplified QR URL extractor for PDFs
# -------------------------------------------------------
# Features:
#  • Fast preflight text scan (no rendering)
#  • Decode image XObjects directly (no page render)
#  • Cheap 300 DPI scan + ROI re-render @900 DPI if needed
#  • Minimal preprocess (CLAHE + Otsu)
#  • Early exit knobs: stop_after_first, max_pages
#  • Always returns: {"urls": [...]}

from __future__ import annotations

import io
import re
from typing import Iterable, List, Optional

try:
    import cv2  # type: ignore
    OPENCV_AVAILABLE = True
except Exception:
    cv2 = None  # type: ignore
    OPENCV_AVAILABLE = False

try:
    from pyzbar.pyzbar import decode as zbar_decode  # type: ignore
    PYZBAR_AVAILABLE = True
except Exception:
    zbar_decode = None  # type: ignore
    PYZBAR_AVAILABLE = False

try:
    import fitz  # PyMuPDF
except Exception as e:
    raise RuntimeError("PyMuPDF (fitz) is required. Install it via 'pip install pymupdf'.") from e

import numpy as np


# ------------------------- Config / Regex -------------------------

_URL_RE = re.compile(r'\bhttps?://[^\s<>"\'\)\]]+', flags=re.IGNORECASE)
_DIFC_BASE = "https://difc.my.salesforce-sites.com/DocumentAuthentication/QRView?ref="
_DIFC_CODE_RE = re.compile(r"(SR-\d{6}-[A-Za-z0-9]+-\d{8}-[A-Za-z0-9]+)")


# ------------------------- Small utils ----------------------------

def _unique_keep_order(seq: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _np_from_pixmap(pix: "fitz.Pixmap") -> np.ndarray:
    if pix.alpha:
        pix = fitz.Pixmap(pix, 0)
    data = np.frombuffer(pix.samples, dtype=np.uint8)
    arr = data.reshape(pix.height, pix.width, pix.n)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr


def _extract_urls_from_text(text: str) -> List[str]:
    return _unique_keep_order(m.group(0).strip() for m in _URL_RE.finditer(text or ""))


def _extract_difc_url_from_text(text: str) -> List[str]:
    m = _DIFC_CODE_RE.search(text or "")
    if not m:
        return []
    return [f"{_DIFC_BASE}{m.group(1)}"]


# ------------------------- Preflight scan -------------------------

def _preflight_text_scan(doc, scan_pages: int = 3) -> List[str]:
    """
    Fast path: scan first few pages for DIFC code or direct URLs.
    """
    n = min(scan_pages, len(doc))
    for p in range(n):
        text = doc[p].get_text("text") or ""
        difc = _extract_difc_url_from_text(text)
        if difc:
            return difc
        generic = _extract_urls_from_text(text)
        if generic:
            return generic
    return []


# ------------------------- XObject decode -------------------------

def _decode_page_images(page) -> List[str]:
    if not OPENCV_AVAILABLE:
        return []
    urls = []
    for info in page.get_images(full=True) or []:
        try:
            xref = info[0]
            pix = fitz.Pixmap(page.parent, xref)
            if pix.n not in (3, 4):
                pix = fitz.Pixmap(fitz.csRGB, pix)
            img = _np_from_pixmap(pix)
            urls.extend(_opencv_try_simple(img))
        except Exception:
            continue
    return _unique_keep_order(urls)


# ------------------------- OpenCV passes -------------------------

def _opencv_try_simple(img_rgb: np.ndarray) -> List[str]:
    if not OPENCV_AVAILABLE:
        return []
    bgr = img_rgb[:, :, ::-1]
    q = cv2.QRCodeDetector()
    texts: List[str] = []
    data, _, _ = q.detectAndDecode(bgr)
    if data:
        texts.append(data)
    ok, datas, _, _ = q.detectAndDecodeMulti(bgr)
    if ok and datas is not None:
        texts.extend(d for d in datas if d)
    return _unique_keep_order(_extract_urls_from_text("\n".join(texts)))


def _opencv_clahe_otsu(img_rgb: np.ndarray) -> List[str]:
    if not OPENCV_AVAILABLE:
        return []
    bgr = img_rgb[:, :, ::-1]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray)
    _, th = cv2.threshold(cl, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    q = cv2.QRCodeDetector()
    data, _, _ = q.detectAndDecode(cv2.cvtColor(th, cv2.COLOR_GRAY2BGR))
    results = [data] if data else []
    ok, datas, _, _ = q.detectAndDecodeMulti(cv2.cvtColor(th, cv2.COLOR_GRAY2BGR))
    if ok and datas is not None:
        results.extend(d for d in datas if d)
    return _unique_keep_order(_extract_urls_from_text("\n".join(results)))


def _opencv_detect_pts(img_rgb: np.ndarray):
    if not OPENCV_AVAILABLE:
        return None
    bgr = img_rgb[:, :, ::-1]
    q = cv2.QRCodeDetector()
    ok, pts = q.detect(bgr)
    return pts if ok else None


# ------------------------- ROI helper ----------------------------

def _clip_rect_from_pts(pts, w: int, h: int, pad: int = 12) -> fitz.Rect:
    xs = [p[0][0] for p in pts]
    ys = [p[0][1] for p in pts]
    x0 = max(0, min(xs) - pad)
    y0 = max(0, min(ys) - pad)
    x1 = min(w, max(xs) + pad)
    y1 = min(h, max(ys) + pad)
    return fitz.Rect(x0, y0, x1, y1)


# ------------------------- Ranking -------------------------------

def _score_url(u: str) -> int:
    score = 0
    if "difc" in u and "QRView" in u:
        score += 20
    if "eservices" in u or "gov" in u:
        score += 10
    if "https://" in u:
        score += 5
    if len(u) > 300:
        score -= 2
    return score


def _rank_urls(urls: List[str]) -> List[str]:
    indexed = list(enumerate(urls))
    indexed.sort(key=lambda t: (_score_url(t[1]), -t[0]), reverse=True)
    return [u for _, u in indexed]


# ------------------------- Main API -------------------------------

def extract_qr_urls_from_pdf_bytes(
    pdf_bytes: bytes,
    dpi: int = 300,
    preproc: bool = True,
    include_text_fallback: bool = True,
    max_dpi: int = 600,
    stop_after_first: bool = True,
    max_pages: int = 3,
) -> dict:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    # 1) Preflight text scan (super fast)
    if include_text_fallback:
        fast_urls = _preflight_text_scan(doc, scan_pages=min(3, max_pages))
        if fast_urls:
            return {"urls": _rank_urls(fast_urls)}

    all_urls: List[str] = []
    end_page = min(len(doc), max_pages)

    for p in range(end_page):
        page = doc[p]
        urls: List[str] = []

        # 2) Image XObjects
        urls.extend(_decode_page_images(page))
        if urls and stop_after_first:
            return {"urls": _rank_urls(urls)}

        # 3) Render @300 DPI
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = _np_from_pixmap(pix)

        urls.extend(_opencv_try_simple(img))
        if urls and stop_after_first:
            return {"urls": _rank_urls(urls)}

        # 4) If QR detected but undecodable, re-render ROI @900 DPI
        pts = _opencv_detect_pts(img)
        if pts is not None and not urls:
            rect_raster = _clip_rect_from_pts(pts, pix.width, pix.height, pad=16)
            rect_pdf = fitz.Rect(
                rect_raster.x0 / zoom, rect_raster.y0 / zoom,
                rect_raster.x1 / zoom, rect_raster.y1 / zoom
            )
            zoom2 = 900 / 72.0
            pix2 = page.get_pixmap(matrix=fitz.Matrix(zoom2, zoom2), clip=rect_pdf, alpha=False)
            img2 = _np_from_pixmap(pix2)
            urls.extend(_opencv_try_simple(img2))
            if not urls and preproc:
                urls.extend(_opencv_clahe_otsu(img2))
            if urls and stop_after_first:
                return {"urls": _rank_urls(urls)}

        # 5) Minimal preprocess if still empty
        if not urls and preproc:
            urls.extend(_opencv_clahe_otsu(img))
            if urls and stop_after_first:
                return {"urls": _rank_urls(urls)}

        # 6) Optional pyzbar
        if not urls and PYZBAR_AVAILABLE:
            try:
                zdata = zbar_decode(img)
                for r in zdata:
                    s = r.data.decode("utf-8", errors="ignore").strip()
                    if s:
                        urls.append(s)
            except Exception:
                pass
            if urls and stop_after_first:
                return {"urls": _rank_urls(_extract_urls_from_text("\n".join(urls)))}

        # 7) Escalate to 600 DPI (rare)
        if not urls and dpi < max_dpi:
            zoom3 = max_dpi / 72.0
            pix3 = page.get_pixmap(matrix=fitz.Matrix(zoom3, zoom3), alpha=False)
            img3 = _np_from_pixmap(pix3)
            urls.extend(_opencv_try_simple(img3))
            if not urls and preproc:
                urls.extend(_opencv_clahe_otsu(img3))
            if urls and stop_after_first:
                return {"urls": _rank_urls(urls)}

        # 8) Text fallback per page
        if include_text_fallback and not urls:
            text = page.get_text("text") or ""
            difc_urls = _extract_difc_url_from_text(text)
            if difc_urls:
                urls.extend(difc_urls)
            else:
                urls.extend(_extract_urls_from_text(text))

        all_urls.extend(urls)
        if stop_after_first and urls:
            break

    ranked = _rank_urls(_unique_keep_order(all_urls))
    return {"urls": ranked}


# ------------------------- Convenience wrappers ------------------

def extract_qr_urls_from_pdf_file(pdf_path: str, **kwargs) -> dict:
    with open(pdf_path, "rb") as f:
        return extract_qr_urls_from_pdf_bytes(f.read(), **kwargs)

def extract_qr_urls_from_pdf_stream(stream: io.BufferedIOBase, **kwargs) -> dict:
    return extract_qr_urls_from_pdf_bytes(stream.read(), **kwargs)
