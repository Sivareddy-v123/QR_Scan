
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from pydantic import BaseModel, Field
from typing import Optional
from app.qr_core import build_response_for_path, build_response_for_bytes
from pathlib import Path

app = FastAPI(title="PDF QR URL Extractor", version="1.2.0")

class ExtractByPathRequest(BaseModel):
    pdf_path: str = Field(..., description="Absolute or relative path to a PDF inside the service.")
    dpi: Optional[int] = Field(250, ge=72, le=600)
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
        return build_response_for_path(
            str(p),
            dpi=req.dpi or 250,
            preproc=bool(req.preproc),
            include_text_fallback=bool(req.text_fallback),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

@app.post("/extract-by-upload")
async def extract_by_upload(
    file: UploadFile = File(...),
    dpi: int = Query(250, ge=72, le=600),
    preproc: bool = Query(True),
    text_fallback: bool = Query(True),
):
    fname = file.filename or "upload.pdf"
    if not fname.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file.")
    try:
        pdf_bytes = await file.read()
        return build_response_for_bytes(
            pdf_bytes,
            original_filename=fname,
            dpi=dpi,
            preproc=preproc,
            include_text_fallback=text_fallback,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process uploaded PDF: {e}")
