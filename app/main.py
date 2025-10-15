# main.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path


from app.qr_core import (  # adjust to 'app.qr_core' if your package layout requires it
    extract_qr_urls_from_pdf_bytes,
    extract_qr_urls_from_pdf_file,
)

app = FastAPI(title="PDF QR URL Extractor", version="1.2.0")

class ExtractByPathRequest(BaseModel):
    pdf_path: str = Field(..., description="Absolute or relative path to a PDF.")
    dpi: Optional[int] = Field(300, ge=72, le=1200)
    preproc: Optional[bool] = True
    text_fallback: Optional[bool] = True

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/extract-by-path")
def extract_by_path(req: ExtractByPathRequest):
    p = Path(req.pdf_path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=400, detail=f"PDF not found: {p}")
    if p.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")
    try:
        # Uses the qr_core convenience wrapper
        result = extract_qr_urls_from_pdf_file(
            str(p),
            dpi=req.dpi or 300,
            preproc=bool(req.preproc),
            include_text_fallback=bool(req.text_fallback),
            max_dpi=600,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

@app.post("/extract-by-upload")
@app.post("/extract-by-upload")
async def extract_by_upload(
    file: UploadFile = File(...),
    dpi: int = Query(300, ge=72, le=1200),
    preproc: bool = Query(True),
    text_fallback: bool = Query(True),
):
    fname = (file.filename or "").lower()
    if not fname.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file.")

    try:
        pdf_bytes = await file.read()
        result = extract_qr_urls_from_pdf_bytes(
            pdf_bytes,
            dpi=dpi,
            preproc=preproc,
            include_text_fallback=text_fallback,
            max_dpi=600,
        )

        # return only the flattened list of URLs
        urls = result.get("urls", [])
        return {"urls": urls}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process uploaded PDF: {e}")

