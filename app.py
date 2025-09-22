import os
import sqlite3
import requests
import trafilatura
from flask import Flask, request, render_template, redirect, url_for, flash
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import time
import datetime

# ---------- Optional OpenAI (will be used only if available/has quota) ----------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ---------- Hugging Face transformers ----------
from transformers import pipeline

# ---------- Extractive fallback (Sumy) ----------
# Install: pip install sumy nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer as SumyTokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk

# Ensure punkt is available for sentence tokenization
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    try:
        nltk.download("punkt", quiet=True)
    except Exception:
        pass

# ---------- Boot ----------
load_dotenv()

# ---------- Config ----------
DB_FILE = os.environ.get("DB_FILE", "reports.db")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
HF_SUMMARY_MODEL = os.environ.get("HF_SUMMARY_MODEL", "sshleifer/distilbart-cnn-12-6")  # smaller & faster
MAX_SOURCES = int(os.environ.get("MAX_SOURCES", "3"))
EXTRACT_CHAR_LIMIT = int(os.environ.get("EXTRACT_CHAR_LIMIT", "2800"))  # keep modest for HF
HF_CHUNK_CHARS = int(os.environ.get("HF_CHUNK_CHARS", "1400"))          # chunk size for HF calls

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")

# Init clients
client = OpenAI() if (OpenAI and OPENAI_API_KEY) else None
hf_summarizer = pipeline("summarization", model=HF_SUMMARY_MODEL)


# ---------- DB ----------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            summary TEXT NOT NULL,
            links TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


init_db()


# ---------- Tool #1: Tavily ----------
def tavily_search(query, num_results=3):
    if not TAVILY_API_KEY:
        print("[Tavily] Missing TAVILY_API_KEY")
        return []
    try:
        r = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "max_results": int(num_results),
                "include_answer": False,
                "include_domains": [],
                "exclude_domains": [],
                "search_depth": "advanced",
            },
            timeout=20,
        )
        if r.status_code != 200:
            print("[Tavily] HTTP", r.status_code, r.text[:200])
            return []
        data = r.json()
        urls = [it.get("url") for it in (data.get("results") or []) if it.get("url")]
        return urls[:num_results]
    except Exception as e:
        print("[Tavily] error:", repr(e))
        return []


# ---------- Tool #2: Extractor ----------
def extract_content(url):
    try:
        resp = requests.get(
            url,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        ctype = resp.headers.get("content-type", "")

        # PDF
        if "application/pdf" in ctype or url.lower().endswith(".pdf"):
            tmp = "temp.pdf"
            with open(tmp, "wb") as f:
                f.write(resp.content)
            try:
                reader = PdfReader(tmp)
                texts = [(page.extract_text() or "") for page in reader.pages]
                return "\n".join(texts).strip() or None
            finally:
                try:
                    os.remove(tmp)
                except Exception:
                    pass

        # HTML
        text = trafilatura.extract(resp.text, include_comments=False, include_tables=False)
        if text:
            return text.strip()
    except Exception as e:
        print("[Extract] error:", repr(e))
    return None


# ---------- Summarizers ----------
def summarize_with_openai(query, docs):
    if not client:
        return None
    joined = "\n\n".join([f"Source: {d['url']}\n{(d['snippet'] or '')[:EXTRACT_CHAR_LIMIT]}" for d in docs])
    prompt = f"""
You are a precise research assistant. Create a short, structured report.

User Query: {query}

Corpus (snippets from sources):
{joined}

Please output only in this structure (Markdown):
# Report
**Query:** <query>

## Key Insights
- <3–6 bullets with crisp, factual points>

## Caveats & Limitations
- <1–3 bullets: data gaps, conflicts, recency limits, etc.>

## Sources
- <title or site> — <url>
- <title or site> — <url>
- <title or site> — <url>
"""
    try:
        resp = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content
    except Exception as e:
        print("[OpenAI] failed:", repr(e))
        return None


def _chunk_text(text, limit=1400):
    # Sentence-aware chunking if nltk is available; fallback to naive slicing
    try:
        from nltk.tokenize import sent_tokenize
        sents = sent_tokenize(text)
        chunks, cur = [], ""
        for s in sents:
            if len(cur) + len(s) + 1 <= limit:
                cur += (" " if cur else "") + s
            else:
                if cur:
                    chunks.append(cur)
                cur = s
        if cur:
            chunks.append(cur)
        return chunks or [text[:limit]]
    except Exception:
        return [text[i:i + limit] for i in range(0, len(text), limit)] or [text[:limit]]


def summarize_with_hf(query, docs):
    # Join & trim
    body = "\n\n".join([f"[{i+1}] {d['url']}\n{(d['snippet'] or '')[:EXTRACT_CHAR_LIMIT]}"
                        for i, d in enumerate(docs)])
    body = body[:EXTRACT_CHAR_LIMIT * len(docs)]
    chunks = _chunk_text(body, limit=HF_CHUNK_CHARS)

    summaries = []
    for ch in chunks:
        try:
            # length heuristics scale a bit with chunk size
            max_len = 180 if len(ch) > 800 else 120
            min_len = 50 if len(ch) > 800 else 30
            out = hf_summarizer(ch, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]
            summaries.append(out.strip())
        except Exception as e:
            print("[HF] chunk failed:", repr(e))
            continue

    if summaries:
        bullets = "\n".join([f"- {s}" for s in summaries])
    else:
        return None  # let caller fall back to extractive

    lines = [
        "# Report",
        f"**Query:** {query}",
        "",
        "## Key Insights",
        bullets,
        "",
        "## Sources",
    ]
    for d in docs:
        lines.append(f"- {d['url']}")
    return "\n".join(lines)


def summarize_extractive(query, docs, n_sentences=6):
    # Concatenate text and select key sentences with LSA (Sumy)
    text = "\n\n".join([(d["snippet"] or "") for d in docs])[:EXTRACT_CHAR_LIMIT * len(docs)]
    try:
        parser = PlaintextParser.from_string(text, SumyTokenizer("english"))
        summarizer = LsaSummarizer()
        sents = summarizer(parser.document, n_sentences)
        bullets = "\n".join([f"- {str(s)}" for s in sents]) or "(no salient sentences found)"
    except Exception as e:
        print("[Sumy] failed:", repr(e))
        bullets = "(fallback summarization unavailable)"

    lines = [
        "# Report",
        f"**Query:** {query}",
        "",
        "## Key Insights",
        bullets,
        "",
        "## Sources",
    ]
    for d in docs:
        lines.append(f"- {d['url']}")
    return "\n".join(lines)


def summarize(query, docs):
    # 1) Try OpenAI (if present), 2) try HF with chunking, 3) extractive Sumy
    return (
        summarize_with_openai(query, docs)
        or summarize_with_hf(query, docs)
        or summarize_extractive(query, docs)
    )


# ---------- Routes ----------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = (request.form.get("query") or "").strip()
        if not query:
            flash("Please enter a query.", "error")
            return redirect(url_for("index"))

        urls = tavily_search(query, num_results=MAX_SOURCES)
        if not urls:
            flash("⚠️ Search failed or returned no results. Try another query.", "warning")
            _save_report(query, "⚠️ No results found.", [])
            return redirect(url_for("index"))

        docs = []
        for u in urls:
            text = extract_content(u)
            docs.append({"url": u, "snippet": (text or "(no extractable text)")[:EXTRACT_CHAR_LIMIT]})
            time.sleep(0.4)  # polite delay

        summary = summarize(query, docs)
        _save_report(query, summary, [d["url"] for d in docs])
        flash("✅ Report created.", "success")
        return redirect(url_for("index"))

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, query, created_at FROM reports ORDER BY created_at DESC")
    reports = c.fetchall()
    conn.close()
    return render_template("index.html", reports=reports)


def _save_report(query, summary, links_list):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO reports (query, summary, links) VALUES (?, ?, ?)", (query, summary, "\n".join(links_list)))
    conn.commit()
    conn.close()


@app.route("/report/<int:report_id>")
def report(report_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT query, summary, links, created_at FROM reports WHERE id=?", (report_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        flash("Report not found.", "error")
        return redirect(url_for("index"))
    query, summary, links, created_at = row
    links_render = [l for l in (links or "").splitlines() if l.strip()]
    return render_template("report.html", query=query, summary=summary, links=links_render, created_at=created_at)


@app.route("/health")
def health():
    return {"status": "ok", "time": datetime.datetime.utcnow().isoformat()}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True)
