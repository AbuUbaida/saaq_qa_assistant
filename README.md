**The Scraper (scripts/scrape_saaq.py)**
Since you have URLs, you need a dedicated script to fetch them.

Job: Reads data/urls.txt. Uses a library like BeautifulSoup or LangChain WebBaseLoader to visit each URL, extract the main content (ignoring navigation bars/footers), and save the text as .txt or .json files into data/raw_html/.

Why: SAAQ websites change. You want a snapshot of the text locally so your RAG pipeline is stable and you don't DDoS their website every time you re-index.

---

**The Ingester (scripts/ingest_data.py)**
This is where the magic happens before the user arrives.

ingest_data.py:

Job: Reads the SAAQ PDFs from `data/raw_pdfs/` and the SAAQ URLs from `data/urls.txt`.

Logic: Loads PDFs (PyPDFLoader) and scrapes URLs (FireCrawl), cleans + chunks text, generates embeddings (HuggingFace endpoint embeddings by default), then upserts vectors into Weaviate (`backend/app/db/vector_store.py`) using deterministic chunk IDs so re-runs don’t duplicate vectors.

Why separate? You run this once (or weekly). You don't want to re-ingest data every time a user asks a question.

[UPDATE]: The ingestion pipeline is implemented in `scripts/{pdf_collector,web_collector,text_processor,embedder}.py` and orchestrated by `scripts/ingest_data.py`.

Future-Proofing (French Support): When you chunk the data here, add a metadata field {"lang": "en"} to every vector. Even though you are only doing English now, this simple tag will save you from having to delete your entire database when you add French later. You will simply filter by lang="en" or lang="fr" in your retrieval query.
