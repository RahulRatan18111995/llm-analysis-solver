# fake_iitm_server.py
"""
Local IITM-style quiz server for testing your solver locally.

Run:
    uv run uvicorn fake_iitm_server:app --host 0.0.0.0 --port 9000 --reload

Endpoints:
 - POST /send    : returns a solver_response JSON pointing to the first quiz URL
 - GET  /quiz/N  : quiz pages (HTML)
 - GET  /files/* : sample files (CSV, PDF, XLSX, JSON)
 - POST /submit/N: submit answers (accepts JSON payload with "answer")
"""

import io
import json
import os
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import pandas as pd
from openpyxl import Workbook
from PIL import Image, ImageDraw, ImageFont
import random
import datetime

ROOT = Path(__file__).parent.resolve()
FILES_DIR = ROOT / "files"
FILES_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Local IITM Test Suite (Realistic)")

# -----------------------------
# File generators (idempotent)
# -----------------------------
def generate_csv(path: Path):
    # Make a CSV with a header 'value' and some numbers
    df = pd.DataFrame({"value": [10, 20, 30, 40, 50]})
    df.to_csv(path, index=False)
    print("Generated CSV:", path)

def generate_pdf_with_table(path: Path):
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
    from reportlab.lib import colors

    data = [
        ["id", "value"],
        [1, 25],
        [2, 35],
        [3, 40],
        [4, 50],
    ]

    doc = SimpleDocTemplate(str(path), pagesize=A4)
    table = Table(data, colWidths=[50, 100])

    style = TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("TEXTCOLOR", (0,0), (-1,0), colors.black),
        ("GRID", (0,0), (-1,-1), 1, colors.black),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 12),
    ])

    table.setStyle(style)
    doc.build([table])
    print("Generated PDF (machine-readable table):", path)


def generate_xlsx(path: Path):
    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["name", "score"])
    ws.append(["Alice", 85])
    ws.append(["Bob", 92])
    ws.append(["Carol", 78])
    wb.save(path)
    print("Generated XLSX:", path)

def generate_json(path: Path):
    payload = {
        "question": "What is the concatenation of 'Hello' and 'World'?",
        "submit": f"http://localhost:9000/submit/5",
        "answer": "HelloWorld"
    }
    path.write_text(json.dumps(payload))
    print("Generated JSON:", path)

def generate_image(path: Path):
    # Create a simple PNG that contains text (not OCR-tested, but demonstrates image attachments)
    img = Image.new("RGB", (400, 120), color=(255,255,255))
    d = ImageDraw.Draw(img)
    try:
        fnt = ImageFont.load_default()
    except Exception:
        fnt = None
    d.text((10, 10), "This is a sample image (not used for OCR).", font=fnt, fill=(0,0,0))
    img.save(path)
    print("Generated image:", path)

def ensure_files_exist():
    # CSV (sum should be 150)
    generate_csv(FILES_DIR / "sample.csv")
    # PDF (values: 25+35+40+50 = 150) — our solver should be able to find the 'value' column
    generate_pdf_with_table(FILES_DIR / "sample.pdf")
    # XLSX
    generate_xlsx(FILES_DIR / "sample.xlsx")
    # JSON with direct answer and submit URL
    generate_json(FILES_DIR / "task.json")
    # Image
    generate_image(FILES_DIR / "sample.png")

ensure_files_exist()

# -----------------------------
# HTML helper
# -----------------------------
def html_page(body: str):
    return HTMLResponse(f"""<html><head><meta charset="utf-8"></head><body style="font-family: sans-serif; padding:20px;">{body}</body></html>""")

# -----------------------------
# /send entrypoint (IITM posts to your endpoint with "url" that points here)
# -----------------------------
@app.post("/send")
async def entrypoint(req: Request):
    now = datetime.datetime.utcnow().isoformat()
    return {"solver_response": {"message": "Request accepted", "started_at": now, "url": "http://localhost:9000/quiz/1"}}

# -----------------------------
# Quiz pages
# -----------------------------

# QUIZ 1: CSV sum
@app.get("/quiz/1")
async def quiz_1():
    html = """
      <h2>Quiz 1 — CSV sum</h2>
      <p>Download file. What is the sum of the "value" column?</p>
      <a href="/files/sample.csv">sample.csv</a>
      <pre>
{
  "question": "Download file. What is the sum of the 'value' column?",
  "submit": "http://localhost:9000/submit/1"
}
      </pre>
    """
    return html_page(html)

@app.post("/submit/1")
async def submit_1(payload: dict):
    # expected sum = 10+20+30+40+50 = 150
    try:
        ans = float(payload.get("answer", -1))
    except:
        return {"correct": False, "reason": "answer not numeric"}
    if abs(ans - 150.0) < 1e-6:
        return {"correct": True, "url": "http://localhost:9000/quiz/2"}
    return {"correct": False, "reason": "wrong sum"}

# QUIZ 2: simple text math
@app.get("/quiz/2")
async def quiz_2():
    html = """
      <h2>Quiz 2 — Math</h2>
      <p>What is 12 + 30?</p>
      <pre>{
  "question": "What is 12 + 30?",
  "submit": "http://localhost:9000/submit/2"
}</pre>
    """
    return html_page(html)

@app.post("/submit/2")
async def submit_2(payload: dict):
    # expected answer 42
    if str(payload.get("answer")).strip() in ("42", "42.0"):
        return {"correct": True, "url": "http://localhost:9000/quiz/3"}
    return {"correct": False}

# QUIZ 3: PDF extraction
@app.get("/quiz/3")
async def quiz_3():
    html = """
      <h2>Quiz 3 — PDF table</h2>
      <p>Download file. What is the sum of the "value" column on page 1?</p>
      <a href="/files/sample.pdf">sample.pdf</a>
      <pre>{
  "question": "Download file. What is the sum of the 'value' column on page 1?",
  "submit": "http://localhost:9000/submit/3"
}</pre>
    """
    return html_page(html)

@app.post("/submit/3")
async def submit3(payload: dict):
    try:
        ans = float(payload.get("answer", -1))
    except:
        return {"correct": False, "reason": "Not a number"}

    # Accept numbers close to 150
    if abs(ans - 150.0) < 1e-3:
        return {"correct": True, "url": "http://localhost:9000/quiz/4"}

    return {"correct": False, "reason": f"Expected 150, got {ans}"}

# QUIZ 4: XLSX reading
@app.get("/quiz/4")
async def quiz_4():
    html = """
      <h2>Quiz 4 — XLSX</h2>
      <p>Download file. What is Bob's score?</p>
      <a href="/files/sample.xlsx">sample.xlsx</a>
      <pre>{
  "question": "What is Bob's score in sample.xlsx?",
  "submit": "http://localhost:9000/submit/4"
}</pre>
    """
    return html_page(html)

@app.post("/submit/4")
async def submit_4(payload: dict):
    # Bob = 92
    if str(payload.get("answer")).strip() in ("92", "92.0"):
        return {"correct": True, "url": "http://localhost:9000/quiz/5"}
    return {"correct": False}

# QUIZ 5: direct JSON already contains answer (test extraction)
@app.get("/quiz/5")
async def quiz_5():
    html = """
      <h2>Quiz 5 — direct JSON</h2>
      <p>This page contains a JSON blob (task.json) with an 'answer' field. The solver should extract and submit it.</p>
      <a href="/files/task.json">task.json</a>
      <pre>{
  "question": "Direct JSON contains answer",
  "submit": "http://localhost:9000/submit/5"
}</pre>
    """
    return html_page(html)

@app.post("/submit/5")
async def submit_5(payload: dict):
    if payload.get("answer") == "HelloWorld":
        return {"correct": True, "url": None}
    return {"correct": False}

# -----------------------------
# files endpoints
# -----------------------------
@app.get("/files/{name}")
async def serve_file(name: str):
    p = FILES_DIR / name
    if not p.exists():
        return JSONResponse({"error": "file not found"}, status_code=404)
    # log path for debugging
    print("SERVING:", p)
    return FileResponse(str(p))

# health
@app.get("/health")
async def health():
    return {"ok": True}
