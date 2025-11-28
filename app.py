# app.py -- merged production-ready LLM-Quiz solver + data endpoints
import os
import re
import json
import time
import base64
import shutil
import tempfile
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from urllib.parse import urljoin, urlparse
from io import BytesIO

import httpx
import pandas as pd
import pdfplumber

from fastapi import FastAPI, Request, BackgroundTasks, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from bs4 import BeautifulSoup

# Optional: Playwright (DOM execution). Controlled by PLAYWRIGHT_DISABLE env var.
try:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    PLAYWRIGHT_OK = True
except Exception:
    sync_playwright = None
    PWTimeout = Exception
    PLAYWRIGHT_OK = False

# Optional: OCR / transcription
try:
    import pytesseract
    from PIL import Image
    TESSERACT_OK = True
except Exception:
    pytesseract = None
    Image = None
    TESSERACT_OK = False

try:
    import whisper
    WHISPER_OK = True
except Exception:
    whisper = None
    WHISPER_OK = False

# Plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except Exception:
    plt = None
    MATPLOTLIB_OK = False

# -------------------------
# Logging + config
# -------------------------
LOG = logging.getLogger("quizsolver")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

TOTAL_TIMEOUT_SECONDS = int(os.getenv("TOTAL_TIMEOUT_SECONDS", "180"))  # 3 minutes default
NAV_TIMEOUT_MS = int(os.getenv("NAV_TIMEOUT_MS", "30000"))
DOWNLOAD_TIMEOUT = int(os.getenv("DOWNLOAD_TIMEOUT", "60"))

SECRET_KEY = os.getenv("SECRET_KEY")
SECRET_MAP_JSON = os.getenv("SECRET_MAP")  # optional {"email":"secret",...}

AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
AIPIPE_URL = os.getenv("AIPIPE_URL", "https://aipipe.org/openrouter/v1/chat/completions")

# Explicit environment toggle: if set to "1" or "true" (case-insensitive), disable Playwright.
PLAYWRIGHT_DISABLE = os.getenv("PLAYWRIGHT_DISABLE", "").lower() in ("1", "true", "yes")

app = FastAPI(title="LLM Analysis Quiz Solver (merged app2)")

# -------------------------
# Models / Validation
# -------------------------
class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

class ScrapeRequest(BaseModel):
    email: str
    secret: str
    url: str
    javascript: Optional[bool] = True
    wait_for: Optional[str] = None
    timeout_ms: Optional[int] = None

class APIRequest(BaseModel):
    email: str
    secret: str
    url: str
    method: Optional[str] = "GET"
    headers: Optional[Dict[str,str]] = None
    params: Optional[Dict[str,Any]] = None
    json: Optional[Any] = None
    data: Optional[Dict[str,Any]] = None
    timeout: Optional[int] = 30

class AnalyzeRequest(BaseModel):
    email: str
    secret: str
    op: str  # 'filter', 'groupby', 'sort', 'describe'
    params: Dict[str,Any]

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Return a safe, serializable message. Avoid returning exc.errors() directly (may contain bytes).
    return JSONResponse(status_code=400, content={"detail": "Invalid JSON or missing required fields"})

# -------------------------
# Helpers
# -------------------------
def clean_url(url: str) -> str:
    if not url:
        return url
    url = str(url).strip()
    url = "".join(ch for ch in url if ch.isprintable())
    url = re.sub(r"\?\d+$", "", url)
    url = url.rstrip("?")
    parsed = urlparse(url)
    if not parsed.scheme:
        url = "http://" + url
    return url

def verify_secret(email: str, secret: str) -> bool:
    if SECRET_MAP_JSON:
        try:
            smap = json.loads(SECRET_MAP_JSON)
            if isinstance(smap, dict):
                return smap.get(email) == secret
        except Exception:
            # allow "email:secret,email2:secret2" convenience format
            try:
                pairs = re.split(r"[;,]\s*", SECRET_MAP_JSON.strip())
                for p in pairs:
                    if not p:
                        continue
                    if ":" in p:
                        e, s = p.split(":", 1)
                        if e.strip() == email and s.strip() == secret:
                            return True
            except Exception:
                pass
        return False
    if SECRET_KEY:
        return secret == SECRET_KEY
    # default: disallow if nothing configured
    return False

def extract_json_from_html(html: str) -> Optional[Dict[str, Any]]:
    soup = BeautifulSoup(html or "", "html.parser")
    # <pre> blocks
    for pre in soup.find_all("pre"):
        text = pre.get_text().strip()
        try:
            return json.loads(text)
        except Exception:
            pass
    # atob(base64)
    b64pat = re.compile(r"atob\((?:`|'|\")([A-Za-z0-9+/=\n\r]+)(?:`|'|\")\)")
    for script in soup.find_all("script"):
        m = b64pat.search(str(script))
        if m:
            try:
                decoded = base64.b64decode(m.group(1)).decode("utf-8", "ignore")
                inner = re.search(r"(\{[\s\S]*\})", decoded)
                if inner:
                    return json.loads(inner.group(1))
            except Exception:
                pass
    # loose JSON
    m = re.search(r"(\{(?:[^{}]|(?1))+\})", html or "", flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    return None

def find_submit_url(html: str) -> Optional[str]:
    if not html:
        return None
    m = re.search(r"Post your answer to\s*(https?://[^\s\"'<]+)", html, flags=re.I)
    if m:
        return m.group(1)
    m2 = re.search(r"(https?://[^\s\"'<>]*/submit[^\s\"'<>]*)", html, flags=re.I)
    if m2:
        return m2.group(1)
    m3 = re.search(r"(/submit[^\s\"'<>]*)", html)
    if m3:
        return m3.group(1)
    return None

def download_url_to_file(url: str, dest_dir: str, referer: Optional[str] = None) -> Optional[str]:
    try:
        headers = {"User-Agent": "QuizSolver/1.0"}
        if referer:
            headers["Referer"] = referer
        with httpx.Client(timeout=DOWNLOAD_TIMEOUT) as client:
            r = client.get(url, headers=headers, follow_redirects=True)
            r.raise_for_status()
            parsed = urlparse(url)
            fname = os.path.basename(parsed.path) or "downloaded"
            fname = re.sub(r"[^A-Za-z0-9._-]", "_", fname)
            base, ext = os.path.splitext(fname)
            path = os.path.join(dest_dir, fname)
            i = 1
            while os.path.exists(path):
                path = os.path.join(dest_dir, f"{base}_{i}{ext}")
                i += 1
            with open(path, "wb") as f:
                f.write(r.content)
            LOG.info("[download] saved: %s", path)
            return path
    except Exception as e:
        LOG.warning("[download error]: %s", e)
        return None

def sum_csv_column(path: str, col: str) -> Optional[float]:
    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_csv(path, sep="\t")
        except Exception:
            return None
    if col in df.columns:
        return float(pd.to_numeric(df[col], errors="coerce").sum())
    for c in df.columns:
        if c.lower() == col.lower():
            return float(pd.to_numeric(df[c], errors="coerce").sum())
    for c in df.columns:
        if c.lower() in ("value", "amount", "val", "total"):
            return float(pd.to_numeric(df[c], errors="coerce").sum())
    return None

def sum_pdf_column(path: str, col: str, page_num: Optional[int] = None) -> Optional[float]:
    try:
        total = 0.0
        found_any = False
        with pdfplumber.open(path) as pdf:
            pages = [pdf.pages[page_num - 1]] if page_num and page_num <= len(pdf.pages) else pdf.pages
            for p in pages:
                tables = p.extract_tables()
                if not tables:
                    continue
                for t in tables:
                    header_row = t[0] if t and len(t) > 0 else []
                    header = [str(h).strip().lower() for h in header_row]
                    if not header:
                        continue
                    try:
                        idx = header.index(col.lower())
                    except ValueError:
                        idx = None
                        for i, h in enumerate(header):
                            if h in ("value", "amount", "val", "total") or col.lower() in h:
                                idx = i
                                break
                    if idx is None:
                        continue
                    for row in t[1:]:
                        try:
                            v = row[idx]
                            if v is None:
                                continue
                            s = str(v).replace(",", "").strip()
                            total += float(s)
                            found_any = True
                        except Exception:
                            continue
        return total if found_any else None
    except Exception as e:
        LOG.warning("[pdf error]: %s", e)
        return None

# -------------------------
# AiPipe LLM fallback (optional)
# -------------------------
def solve_with_llm(question_text: str, email: str, secret: str, url: str) -> Optional[Dict[str, Any]]:
    if not AIPIPE_TOKEN:
        LOG.info("[LLM] No AiPipe token - skipping fallback")
        return None
    LOG.info("[LLM] Fallback activated")
    prompt = f"""
Extract ONLY the final correct answer to this quiz question.

Question:
{question_text}

Return ONLY strict JSON:
{{ "answer": <value> }}

<value> can be number/string/bool/base64/json.
No explanation.
"""
    try:
        with httpx.Client(timeout=30) as client:
            rsp = client.post(
                AIPIPE_URL,
                headers={"Authorization": f"Bearer {AIPIPE_TOKEN}", "Content-Type": "application/json"},
                json={
                    "model": "openai/gpt-4.1-nano",
                    "messages": [
                        {"role": "system", "content": "You output only JSON."},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            rsp.raise_for_status()
            body = rsp.json()
            content = body["choices"][0]["message"]["content"]
            ans = json.loads(content)
            if "answer" in ans:
                return {"email": email, "secret": secret, "url": url, "answer": ans["answer"]}
    except Exception as e:
        LOG.warning("[LLM error]: %s", e)
    return None

# -------------------------
# Rendering (Playwright optional) - respects PLAYWRIGHT_DISABLE env var
# -------------------------
def render_page(url: str, timeout_ms: int = NAV_TIMEOUT_MS, wait_for: Optional[str] = None, javascript: bool = True) -> Dict[str, Any]:
    url = clean_url(url)
    html = None
    meta = {"rendered_with": "httpx", "url": url}
    use_playwright = (PLAYWRIGHT_OK and not PLAYWRIGHT_DISABLE and javascript)
    if use_playwright:
        try:
            pw = sync_playwright().start()
            browser = pw.chromium.launch(headless=True)
            ctx = browser.new_context()
            page = ctx.new_page()
            try:
                page.goto(url, timeout=timeout_ms)
            except Exception:
                pass
            if wait_for:
                try:
                    page.wait_for_selector(wait_for, timeout=timeout_ms//2)
                except Exception:
                    pass
            try:
                page.wait_for_load_state("networkidle", timeout=2000)
            except Exception:
                pass
            html = page.content()
            page.close()
            ctx.close()
            browser.close()
            try:
                pw.stop()
            except Exception:
                pass
            meta["rendered_with"] = "playwright"
        except Exception as e:
            LOG.warning("Playwright render failed: %s", e)
            html = None
            meta["render_error"] = str(e)
    if not html:
        try:
            with httpx.Client(timeout=30) as client:
                r = client.get(url, follow_redirects=True)
                r.raise_for_status()
                html = r.text
            meta["rendered_with"] = "httpx"
        except Exception as e:
            LOG.warning("httpx fetch failed: %s", e)
            meta["fetch_error"] = str(e)
            return {"html": None, "meta": meta}
    return {"html": html, "meta": meta}

# -------------------------
# Core solver loop (background)
# -------------------------
def solve_quiz(start_url: str, email: str, secret: str, deadline: float) -> Dict[str, Any]:
    tmp = tempfile.mkdtemp(prefix="quizsolver_")
    result = {"steps": [], "finished": False}
    try:
        use_playwright = (PLAYWRIGHT_OK and not PLAYWRIGHT_DISABLE)
        if use_playwright:
            LOG.info("Playwright available: will try browser rendering (if browsers installed).")
        else:
            LOG.info("Playwright not available or disabled: will use httpx fallback.")

        url = clean_url(start_url)

        # Playwright browser reuse
        pw = None
        browser = None
        if use_playwright:
            try:
                pw = sync_playwright().start()
                browser = pw.chromium.launch(headless=True)
            except Exception as e:
                LOG.warning("Playwright start failed: %s (falling back to httpx)", e)
                try:
                    if pw:
                        pw.stop()
                except Exception:
                    pass
                use_playwright = False
                browser = None

        while time.time() < deadline:
            url = clean_url(url)
            LOG.info("[solver] Visiting: %s", url)

            # /send special-case (IITM demo server)
            parsed = urlparse(url)
            if parsed.path.endswith("/send"):
                LOG.info("[solver] Detected /send → sending POST")
                try:
                    with httpx.Client(timeout=15) as client:
                        r = client.post(url, json={"ping": True})
                        r.raise_for_status()
                        jr = r.json()
                    result["steps"].append({"url": url, "payload_found": True, "response": jr})
                    next_url = None
                    if isinstance(jr, dict):
                        next_url = (jr.get("solver_response") or {}).get("url") or jr.get("url")
                    if next_url:
                        url = clean_url(next_url)
                        continue
                    else:
                        result["error"] = "/send returned no next url"
                        return result
                except Exception as e:
                    result["error"] = f"/send handling failed: {e}"
                    return result

            # 1) render/fetch page
            html = None
            if use_playwright and browser:
                try:
                    ctx = browser.new_context()
                    page = ctx.new_page()
                    try:
                        page.goto(url, timeout=NAV_TIMEOUT_MS)
                    except PWTimeout:
                        LOG.debug("Playwright navigation timeout for %s", url)
                    try:
                        page.wait_for_load_state("networkidle", timeout=2000)
                    except Exception:
                        pass
                    html = page.content()
                    page.close()
                    ctx.close()
                except Exception as e:
                    LOG.warning("Playwright failed for %s: %s; falling back to httpx", url, e)
                    try:
                        if browser:
                            browser.close()
                    except Exception:
                        pass
                    use_playwright = False
                    browser = None

            if not html:
                try:
                    with httpx.Client(timeout=15) as client:
                        r = client.get(url, follow_redirects=True)
                        r.raise_for_status()
                        html = r.text
                except Exception as e:
                    LOG.warning("HTTP fetch failed for %s: %s", url, e)
                    result["error"] = f"fetch failed: {e}"
                    return result

            soup = BeautifulSoup(html or "", "html.parser")
            step = {"url": url}
            payload = extract_json_from_html(html)
            step["payload_found"] = bool(payload)

            # question detection
            question_text = None
            if payload and isinstance(payload, dict):
                question_text = payload.get("question") or payload.get("task") or payload.get("prompt") or payload.get("instructions")
            if not question_text:
                txt = soup.get_text(" ", strip=True)
                m = re.search(r"(Q\d+.*|What is.*|Download file.*|Sum of.*)", txt, flags=re.I)
                if m:
                    question_text = m.group(0)
            step["question_text"] = question_text

            # find submit url
            submit_url = None
            if payload and isinstance(payload, dict):
                submit_url = payload.get("submit") or payload.get("submit_url") or payload.get("url")
            if not submit_url:
                submit_url = find_submit_url(html or "")
            if submit_url and not submit_url.startswith("http"):
                base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
                submit_url = urljoin(base, submit_url)

            # find downloadable file
            file_link = None
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if not href.lower().startswith("http"):
                    href = urljoin(url, href)
                if any(href.lower().endswith(ext) for ext in (".csv", ".tsv", ".pdf", ".xlsx", ".xls", ".json", ".txt")):
                    file_link = href
                    break
            step["file_link"] = file_link

            # deterministic answers
            answer_payload = None
            if payload and isinstance(payload, dict) and "answer" in payload:
                answer_payload = {"email": email, "secret": secret, "url": url, "answer": payload["answer"]}

            if file_link and not answer_payload:
                fpath = download_url_to_file(file_link, tmp, referer=url)
                if fpath:
                    if fpath.lower().endswith((".csv", ".tsv", ".txt")):
                        col = "value"
                        m = re.search(r"the ['\"]?(.+?)['\"]? column", (question_text or ""), flags=re.I)
                        if m:
                            col = m.group(1)
                        s = sum_csv_column(fpath, col)
                        if s is not None:
                            answer_payload = {"email": email, "secret": secret, "url": url, "answer": float(s)}
                    elif fpath.lower().endswith(".pdf"):
                        col = "value"
                        m = re.search(r"the ['\"]?(.+?)['\"]? column", (question_text or ""), flags=re.I)
                        if m:
                            col = m.group(1)
                        pg = None
                        mm = re.search(r"page (\d+)", (question_text or ""), flags=re.I)
                        if mm:
                            pg = int(mm.group(1))
                        s = sum_pdf_column(fpath, col, page_num=pg)
                        if s is not None:
                            answer_payload = {"email": email, "secret": secret, "url": url, "answer": float(s)}
                    elif fpath.lower().endswith((".xlsx", ".xls")):
                        try:
                            df = pd.read_excel(fpath)
                            # Example heuristic: find Bob's score (first col contains names)
                            row = df[df.iloc[:, 0].astype(str).str.lower() == "bob"]
                            if not row.empty:
                                score = float(row.iloc[0, 1])
                                answer_payload = {"email": email, "secret": secret, "url": url, "answer": score}
                        except Exception as e:
                            LOG.warning("[xlsx error]: %s", e)
                    elif fpath.lower().endswith(".json"):
                        try:
                            with open(fpath, "r") as jf:
                                jdata = json.load(jf)
                            if "answer" in jdata:
                                answer_payload = {"email": email, "secret": secret, "url": url, "answer": jdata["answer"]}
                        except Exception as e:
                            LOG.warning("[json error]: %s", e)

            # LLM fallback
            if not answer_payload:
                LOG.info("[solver] Trying LLM fallback...")
                llm_ans = solve_with_llm(question_text or "", email, secret, url)
                if llm_ans and "answer" in llm_ans:
                    answer_payload = {"email": email, "secret": secret, "url": url, "answer": llm_ans["answer"]}

            # submit
            if submit_url and answer_payload:
                LOG.info("[solver] Submitting: %s", submit_url)
                try:
                    with httpx.Client(timeout=30) as client:
                        r = client.post(submit_url, json=answer_payload)
                        r.raise_for_status()
                        rsp_json = None
                        try:
                            rsp_json = r.json()
                        except Exception:
                            pass
                    step["submit_status"] = r.status_code
                    step["response"] = rsp_json
                    result["steps"].append(step)
                    next_url = None
                    if isinstance(rsp_json, dict):
                        next_url = rsp_json.get("url") or (rsp_json.get("solver_response") or {}).get("url")
                    if next_url:
                        url = clean_url(next_url)
                        continue
                    result["finished"] = True
                    return result
                except Exception as e:
                    step["submit_error"] = str(e)
                    result["steps"].append(step)
                    return result

            # record and exit if no answer/submit possible
            result["steps"].append(step)
            return result

        result["error"] = "timeout"
        return result

    except Exception as e:
        LOG.exception("solve_quiz top-level error")
        return {"finished": False, "error": str(e), "steps": result.get("steps", [])}
    finally:
        try:
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            pass
        try:
            if PLAYWRIGHT_OK and sync_playwright:
                try:
                    sync = sync_playwright()
                    sync.stop()
                except Exception:
                    pass
        except Exception:
            pass

# -------------------------
# Endpoint implementations (scrape, call_api, clean_file, process_file, analyze, chart, transcribe, vision_ocr)
# -------------------------
@app.post("/scrape")
async def scrape_endpoint(req: ScrapeRequest):
    if not verify_secret(req.email, req.secret):
        raise HTTPException(status_code=403, detail="Invalid secret")
    timeout_ms = req.timeout_ms or NAV_TIMEOUT_MS
    # prefer javascript unless disabled via env
    javascript = bool(req.javascript) and not PLAYWRIGHT_DISABLE
    data = render_page(req.url, timeout_ms=timeout_ms, wait_for=req.wait_for, javascript=javascript)
    html = data.get("html")
    meta = data.get("meta", {})
    if not html:
        return {"ok": False, "error": "failed to fetch", "meta": meta}
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else None
    text = soup.get_text(" ", strip=True)[:200000]
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href.lower().startswith("http"):
            href = urljoin(req.url, href)
        links.append(href)
    tables = []
    for t in soup.find_all("table"):
        try:
            df = pd.read_html(str(t))[0]
            tables.append({"columns": df.columns.tolist(), "rows": df.head(10).to_dict(orient="records")})
        except Exception:
            continue
    js = extract_json_from_html(html)
    return {"ok": True, "meta": meta, "title": title, "text_snippet": text[:5000], "links_count": len(links), "links_sample": links[:20], "tables": tables, "embedded_json": js}

@app.post("/call_api")
async def call_api(req: APIRequest):
    if not verify_secret(req.email, req.secret):
        raise HTTPException(status_code=403, detail="Invalid secret")
    method = (req.method or "GET").upper()
    headers = req.headers or {}
    timeout = req.timeout or 30
    try:
        with httpx.Client(timeout=timeout) as client:
            func = getattr(client, method.lower(), client.get)
            r = func(req.url, headers=headers, params=req.params, json=req.json, data=req.data, follow_redirects=True)
            r.raise_for_status()
            content_type = r.headers.get("Content-Type","")
            if "application/json" in content_type:
                return {"ok": True, "status": r.status_code, "json": r.json()}
            text = r.text
            return {"ok": True, "status": r.status_code, "text_snippet": text[:40000], "content_type": content_type}
    except Exception as e:
        LOG.warning("call_api error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clean_file")
async def clean_file_endpoint(email: str = Form(...), secret: str = Form(...), file: UploadFile = File(None), url: str = Form(None)):
    if not verify_secret(email, secret):
        raise HTTPException(status_code=403, detail="Invalid secret")
    tmp = tempfile.mkdtemp(prefix="clean_")
    try:
        path = None
        if file:
            p = os.path.join(tmp, file.filename)
            with open(p, "wb") as f:
                f.write(await file.read())
            path = p
        elif url:
            path = download_url_to_file(url, tmp)
        else:
            raise HTTPException(status_code=400, detail="Provide file or url")
        if not path or not os.path.exists(path):
            raise HTTPException(status_code=500, detail="Failed to download or save file")
        lower = path.lower()
        if lower.endswith(".pdf"):
            text_chunks = []
            tables = []
            try:
                with pdfplumber.open(path) as pdf:
                    for p in pdf.pages:
                        t = p.extract_text()
                        if t:
                            text_chunks.append(t.strip())
                        for tab in p.extract_tables() or []:
                            try:
                                df = pd.DataFrame(tab[1:], columns=tab[0])
                                tables.append({"columns": df.columns.tolist(), "rows": df.head(20).to_dict(orient="records")})
                            except Exception:
                                continue
            except Exception as e:
                LOG.warning("pdf cleaning error: %s", e)
            return {"ok": True, "file": os.path.basename(path), "text_snippet": ("\n".join(text_chunks)[:20000]), "tables_count": len(tables), "tables_sample": tables[:3]}
        elif lower.endswith((".csv", ".tsv", ".txt")):
            try:
                df = pd.read_csv(path)
            except Exception:
                try:
                    df = pd.read_csv(path, sep="\t")
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"CSV parse error: {e}")
            for c in df.select_dtypes(include=["object"]).columns:
                df[c] = df[c].astype(str).str.strip()
            return {"ok": True, "file": os.path.basename(path), "columns": df.columns.tolist(), "rows_sample": df.head(20).to_dict(orient="records")}
        elif lower.endswith((".xlsx", ".xls")):
            try:
                df = pd.read_excel(path)
                return {"ok": True, "file": os.path.basename(path), "columns": df.columns.tolist(), "rows_sample": df.head(20).to_dict(orient="records")}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Excel parse error: {e}")
        elif lower.endswith((".png", ".jpg", ".jpeg", ".tiff")):
            if TESSERACT_OK:
                img = Image.open(path)
                text = pytesseract.image_to_string(img)
                return {"ok": True, "file": os.path.basename(path), "ocr_text_snippet": text[:20000]}
            else:
                raise HTTPException(status_code=501, detail="OCR not available (pytesseract not installed)")
        else:
            with open(path, "rb") as f:
                raw = f.read(20000)
            try:
                text = raw.decode("utf-8", errors="ignore")
                return {"ok": True, "file": os.path.basename(path), "text_snippet": text}
            except Exception:
                return {"ok": True, "file": os.path.basename(path), "message": "binary file - no preview available"}
    finally:
        try:
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            pass

@app.post("/process_file")
async def process_file(email: str = Form(...), secret: str = Form(...), file: UploadFile = File(None), op: str = Form(...), params: str = Form("{}")):
    if not verify_secret(email, secret):
        raise HTTPException(status_code=403, detail="Invalid secret")
    try:
        params = json.loads(params)
    except Exception:
        params = {}
    if not file:
        raise HTTPException(status_code=400, detail="file required")
    tmp = tempfile.mkdtemp(prefix="process_")
    try:
        p = os.path.join(tmp, file.filename)
        with open(p, "wb") as f:
            f.write(await file.read())
        lower = p.lower()
        if lower.endswith((".csv", ".tsv", ".txt")):
            df = pd.read_csv(p)
        elif lower.endswith((".xlsx", ".xls")):
            df = pd.read_excel(p)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type for processing")
        if op == "sum_column":
            col = params.get("column")
            if not col:
                raise HTTPException(status_code=400, detail="column param required")
            s = float(pd.to_numeric(df[col], errors="coerce").sum())
            return {"ok": True, "operation": "sum_column", "column": col, "sum": s}
        elif op == "filter_rows":
            expr = params.get("expr")
            if not expr:
                raise HTTPException(status_code=400, detail="expr param required")
            try:
                df2 = df.query(expr)
                return {"ok": True, "rows": df2.head(100).to_dict(orient="records"), "count": len(df2)}
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"filter error: {e}")
        elif op == "describe":
            return {"ok": True, "describe": df.describe(include="all").to_dict()}
        elif op == "pivot":
            index = params.get("index")
            columns = params.get("columns")
            values = params.get("values")
            aggfunc = params.get("aggfunc", "sum")
            if not (index and columns and values):
                raise HTTPException(status_code=400, detail="index, columns, values required")
            try:
                table = pd.pivot_table(df, index=index, columns=columns, values=values, aggfunc=aggfunc)
                return {"ok": True, "pivot": table.head(200).to_json()}
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"pivot error: {e}")
        else:
            raise HTTPException(status_code=400, detail="unsupported op")
    finally:
        try:
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            pass

@app.post("/analyze")
async def analyze_endpoint(req: AnalyzeRequest, file: UploadFile = File(...)):
    if not verify_secret(req.email, req.secret):
        raise HTTPException(status_code=403, detail="Invalid secret")
    tmp = tempfile.mkdtemp(prefix="analyze_")
    try:
        p = os.path.join(tmp, file.filename)
        with open(p, "wb") as f:
            f.write(await file.read())
        lower = p.lower()
        if lower.endswith((".csv", ".tsv", ".txt")):
            df = pd.read_csv(p)
        elif lower.endswith((".xlsx", ".xls")):
            df = pd.read_excel(p)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        op = req.op.lower()
        params = req.params or {}
        if op == "filter":
            expr = params.get("expr")
            if not expr:
                raise HTTPException(status_code=400, detail="expr required")
            df2 = df.query(expr)
            return {"ok": True, "count": len(df2), "rows_sample": df2.head(100).to_dict(orient="records")}
        elif op == "groupby":
            cols = params.get("cols")
            agg = params.get("agg", {"*": "count"})
            if not cols:
                raise HTTPException(status_code=400, detail="cols required")
            gb = df.groupby(cols).agg(agg).reset_index()
            return {"ok": True, "result_sample": gb.head(200).to_dict(orient="records")}
        elif op == "sort":
            by = params.get("by")
            asc = params.get("ascending", True)
            if not by:
                raise HTTPException(status_code=400, detail="by required")
            df2 = df.sort_values(by=by, ascending=asc)
            return {"ok": True, "rows_sample": df2.head(200).to_dict(orient="records")}
        elif op == "describe":
            return {"ok": True, "describe": df.describe(include="all").to_dict()}
        else:
            raise HTTPException(status_code=400, detail="unsupported op")
    finally:
        try:
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            pass

@app.post("/chart")
async def chart_endpoint(email: str = Form(...), secret: str = Form(...), file: UploadFile = File(...), x: str = Form(...), y: str = Form(...), kind: str = Form("line"), width: int = Form(800), height: int = Form(600)):
    if not verify_secret(email, secret):
        raise HTTPException(status_code=403, detail="Invalid secret")
    if not MATPLOTLIB_OK:
        raise HTTPException(status_code=501, detail="matplotlib not available on server")
    tmp = tempfile.mkdtemp(prefix="chart_")
    try:
        p = os.path.join(tmp, file.filename)
        with open(p, "wb") as f:
            f.write(await file.read())
        lower = p.lower()
        if lower.endswith((".csv", ".tsv", ".txt")):
            df = pd.read_csv(p)
        elif lower.endswith((".xlsx", ".xls")):
            df = pd.read_excel(p)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        if x not in df.columns or y not in df.columns:
            raise HTTPException(status_code=400, detail=f"x or y column not found, available: {df.columns.tolist()}")
        buf = BytesIO()
        plt.figure(figsize=(width/100, height/100))
        if kind == "line":
            plt.plot(df[x], df[y])
        elif kind == "bar":
            plt.bar(df[x].astype(str), df[y])
            plt.xticks(rotation=45, ha="right")
        elif kind == "scatter":
            plt.scatter(df[x], df[y])
        elif kind == "hist":
            plt.hist(df[y].dropna())
        else:
            raise HTTPException(status_code=400, detail="unsupported chart kind")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.tight_layout()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    finally:
        try:
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            pass

@app.post("/transcribe")
async def transcribe_file(email: str = Form(...), secret: str = Form(...), file: UploadFile = File(...), model: str = Form("small")):
    if not verify_secret(email, secret):
        raise HTTPException(status_code=403, detail="Invalid secret")
    if not WHISPER_OK:
        raise HTTPException(status_code=501, detail="Whisper not installed/available on server")
    tmp = tempfile.mkdtemp(prefix="trans_")
    try:
        p = os.path.join(tmp, file.filename)
        with open(p, "wb") as f:
            f.write(await file.read())
        try:
            m = whisper.load_model(model)
            res = m.transcribe(p)
            return {"ok": True, "text": res.get("text"), "segments": res.get("segments")}
        except Exception as e:
            LOG.warning("whisper error: %s", e)
            raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            pass

@app.post("/vision_ocr")
async def vision_ocr(email: str = Form(...), secret: str = Form(...), file: UploadFile = File(...)):
    if not verify_secret(email, secret):
        raise HTTPException(status_code=403, detail="Invalid secret")
    if not TESSERACT_OK:
        raise HTTPException(status_code=501, detail="pytesseract not installed on server")
    tmp = tempfile.mkdtemp(prefix="ocr_")
    try:
        p = os.path.join(tmp, file.filename)
        with open(p, "wb") as f:
            f.write(await file.read())
        try:
            img = Image.open(p)
            text = pytesseract.image_to_string(img)
            return {"ok": True, "text": text}
        except Exception as e:
            LOG.warning("ocr error: %s", e)
            raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            pass

# -------------------------
# /receive_request (main quiz entry) runs solver in background task
# -------------------------
@app.post("/receive_request")
async def receive_request(payload: QuizRequest, background_tasks: BackgroundTasks):
    if not verify_secret(payload.email, payload.secret):
        raise HTTPException(status_code=403, detail="Invalid secret")
    deadline = time.time() + TOTAL_TIMEOUT_SECONDS

    def run_solver():
        try:
            res = solve_quiz(payload.url, payload.email, payload.secret, deadline)
            LOG.info("[RESULT]:\n%s", json.dumps(res, indent=2, default=str))
        except Exception as e:
            LOG.exception("Background solver crashed: %s", e)

    background_tasks.add_task(run_solver)
    return {"message": "Request accepted", "started_at": datetime.utcnow().isoformat()}

@app.get("/health")
async def health():
    return {
        "ok": True,
        "playwright_available": PLAYWRIGHT_OK and not PLAYWRIGHT_DISABLE,
        "playwright_installed": PLAYWRIGHT_OK,
        "playwright_disabled_env": PLAYWRIGHT_DISABLE,
        "tesseract": TESSERACT_OK,
        "whisper": WHISPER_OK,
        "matplotlib": MATPLOTLIB_OK,
    }
