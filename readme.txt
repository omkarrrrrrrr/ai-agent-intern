# AI Agent Intern — Research Agent (Flask + Tavily + HF)

A tiny web app that:
1) searches the web (Tavily),
2) extracts page text (trafilatura / PyPDF2),
3) summarizes (OpenAI if available; otherwise Hugging Face; lastly Sumy extractive),
4) saves reports to SQLite, and
5) lets you view them in a simple UI.

> Two tools only (per assignment): **Search** + **Content Extractor**.
> Summarization is a post-processing step over extracted content.

---


Flow
- User enters a query → Tavily returns N URLs → we fetch each page (or PDF) and extract clean text → summarizer turns snippets into a short report → report is stored in SQLite and shown on the web.

Summarization fallback chain
1) OpenAI (if key present + quota)
2) Hugging Face `transformers` (default model: `sshleifer/distilbart-cnn-12-6`)
3) Sumy (LSA extractive) – guarantees you never see “summarization failed”


