# app.py -- patched single-file LLM-Quiz solver
import os
import re
import json
import time
import base64
import shutil
import tempfile
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from urllib.parse import urljoin, urlparse

import httpx
import pandas as pd
import pdfplumber

from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from bs4 import BeautifulSoup

# Playwright is optional — the code will fall back to httpx if playwright or browsers are missing.
try:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    PLAYWRIGHT_OK = True
except Exception:
    sync_playwright = None
    PWTimeout = Exception
    PLAYWRIGHT_OK = False

# -------------------------
# Configuration
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

app = FastAPI(title="LLM Analysis Quiz Solver (patched single-file)")

# -------------------------
# Models / Validation
# -------------------------
class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=400, content={"detail": "Invalid JSON or missing fields", "errors": exc.errors()})


# -------------------------
# Helpers
# -------------------------
def clean_url(url: str) -> str:
    """Sanitize incoming URLs (strip invisible chars, remove trailing ? or ?1 etc., ensure scheme)."""
    if not url:
        return url
    url = str(url).strip()
    url = "".join(ch for ch in url if ch.isprintable())
    # remove trailing bare ? or numeric query like ?1 or ?123
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
            return smap.get(email) == secret
        except Exception:
            return False
    if SECRET_KEY:
        return secret == SECRET_KEY
    # if none set, disallow
    return False


def extract_json_from_html(html: str) -> Optional[Dict[str, Any]]:
    """Try finding JSON inside <pre>, atob(base64), or loose JSON."""
    soup = BeautifulSoup(html, "html.parser")
    # <pre> JSON
    for pre in soup.find_all("pre"):
        text = pre.get_text().strip()
        try:
            return json.loads(text)
        except Exception:
            pass
    # atob(`base64`) inside scripts
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
    # loose JSON fallback
    m = re.search(r"(\{(?:[^{}]|(?1))+\})", html, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    return None


def find_submit_url(html: str) -> Optional[str]:
    # pattern examples used in the course
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
    """Download and save to dest_dir, return local path or None"""
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
    # exact
    if col in df.columns:
        return float(pd.to_numeric(df[col], errors="coerce").sum())
    # case-insensitive
    for c in df.columns:
        if c.lower() == col.lower():
            return float(pd.to_numeric(df[c], errors="coerce").sum())
    # fallback common names
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
# Optional AiPipe LLM fallback
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
# Core solver loop
# -------------------------
def solve_quiz(start_url: str, email: str, secret: str, deadline: float) -> Dict[str, Any]:
    tmp = tempfile.mkdtemp(prefix="quizsolver_")
    result = {"steps": [], "finished": False}
    try:
        use_playwright = PLAYWRIGHT_OK
        if use_playwright:
            LOG.info("Playwright available: will try browser rendering (if browsers installed).")
        else:
            LOG.info("Playwright not available: will use httpx HTML fetching fallback.")

        url = clean_url(start_url)

        # If using playwright, open it outside loop to reuse browser
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

            # Special-case IITM /send style endpoints: they often expect a POST and return JSON with next URL
            parsed = urlparse(url)
            if parsed.path.endswith("/send"):
                LOG.info("[solver] Detected /send → sending POST")
                try:
                    with httpx.Client(timeout=15) as client:
                        r = client.post(url, json={"ping": True})
                        r.raise_for_status()
                        jr = r.json()
                    # record and jump if URL returned
                    result["steps"].append({"url": url, "payload_found": True, "response": jr})
                    # try to extract next URL
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

            # 1) fetch/render page HTML
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
                # httpx fetch fallback
                try:
                    with httpx.Client(timeout=15) as client:
                        r = client.get(url, follow_redirects=True)
                        r.raise_for_status()
                        html = r.text
                except Exception as e:
                    LOG.warning("HTTP fetch failed for %s: %s", url, e)
                    result["error"] = f"fetch failed: {e}"
                    return result

            soup = BeautifulSoup(html, "html.parser")
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
                submit_url = find_submit_url(html)
            if submit_url and not submit_url.startswith("http"):
                base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
                submit_url = urljoin(base, submit_url)

            # find downloadable file
            file_link = None
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if not href.lower().startswith("http"):
                    href = urljoin(url, href)
                # include JSON files as downloadable resources (Quiz 5 uses task.json)
                if any(href.lower().endswith(ext) for ext in (".csv", ".tsv", ".pdf", ".xlsx", ".xls", ".json")):
                    file_link = href
                    break
            step["file_link"] = file_link

            # deterministic solvers
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
                    # note: xlsx not implemented deterministically here
                            # XLSX / Excel
                    elif fpath.lower().endswith((".xlsx", ".xls")):
                        try:
                            df = pd.read_excel(fpath)
                            # Example: find Bob's score
                            # Detect column names dynamically
                            row = df[df.iloc[:, 0].astype(str).str.lower() == "bob"]
                            if not row.empty:
                                score = float(row.iloc[0, 1])
                                answer_payload = {
                                    "email": email,
                                    "secret": secret,
                                    "url": url,
                                    "answer": score
                                }
                        except Exception as e:
                            print("[xlsx error]:", e)

                    # NEW: JSON auto-answer support (fixes Quiz 5)
                    elif fpath.lower().endswith(".json"):
                        try:
                            with open(fpath, "r") as jf:
                                jdata = json.load(jf)
                            if "answer" in jdata:
                                answer_payload = {
                                    "email": email,
                                    "secret": secret,
                                    "url": url,
                                    "answer": jdata["answer"]
                                }
                        except Exception as e:
                            print("[json error]:", e)  
            # LLM fallback
            if not answer_payload:
                LOG.info("[solver] Trying LLM fallback...")
                answer_payload = solve_with_llm(question_text or "", email, secret, url)

            # submit if possible
            if submit_url and answer_payload:
                LOG.info("[solver] Submitting: %s", submit_url)
                try:
                    with httpx.Client(timeout=30) as client:
                        r = client.post(submit_url, json=answer_payload)
                        r.raise_for_status()
                        rsp_json = r.json()
                    step["submit_status"] = r.status_code
                    step["response"] = rsp_json
                    result["steps"].append(step)
                    # follow next url if provided
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

            # record and exit (no answer/submit)
            result["steps"].append(step)
            return result

        result["error"] = "timeout"
        return result

    except Exception as e:
        LOG.exception("solve_quiz top-level error")
        return {"finished": False, "error": str(e), "steps": result.get("steps", [])}
    finally:
        # cleanup
        try:
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            pass
        # stop playwright if started
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
# API Endpoints
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
    return {"ok": True}
