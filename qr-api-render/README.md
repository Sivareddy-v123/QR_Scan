
# PDF QR URL Extractor — Render (no Docker)

FastAPI app that extracts URL(s) from QR codes in PDFs.

## Local run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Deploy to Render (Free) — No Docker
1. Push to GitHub.
2. Render → New → Blueprint → select repo.
3. Render reads render.yaml and deploys.
4. Open https://<your-app>.onrender.com/docs

## Endpoints
- GET /healthz
- POST /extract-by-upload?dpi=300&preproc=true&text_fallback=true
  (form-data → key: file, type: File)
- POST /extract-by-path (JSON)
  { "pdf_path": "/app/some.pdf", "dpi": 300, "preproc": true, "text_fallback": true }
