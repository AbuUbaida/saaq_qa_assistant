**The Scraper (scripts/scrape_saaq.py)**
Since you have URLs, you need a dedicated script to fetch them.

Job: Reads data/urls.txt. Uses a library like BeautifulSoup or LangChain WebBaseLoader to visit each URL, extract the main content (ignoring navigation bars/footers), and save the text as .txt or .json files into data/raw_html/.

Why: SAAQ websites change. You want a snapshot of the text locally so your RAG pipeline is stable and you don't DDoS their website every time you re-index.

---

**The Ingester (scripts/ingest_data.py)**
This is where the magic happens before the user arrives.

ingest_data.py:

Job: Reads the SAAQ PDFs from data/raw_pdfs/.

Logic: Uses a library (like Unstructured or PyPDF) to extract text. Splits text into chunks (e.g., 500 characters with overlapping). Calls the Embedding Model (e.g., OpenAI text-embedding-3-small) to turn text into vectors. Pushes these vectors into your Vector Database.

Why separate? You run this once (or weekly). You don't want to re-ingest data every time a user asks a question.

[UPDATE]: This script now needs two "loaders": one for PDFs (e.g., PyPDFLoader) and one for text files (e.g., TextLoader).

Future-Proofing (French Support): When you chunk the data here, add a metadata field {"lang": "en"} to every vector. Even though you are only doing English now, this simple tag will save you from having to delete your entire database when you add French later. You will simply filter by lang="en" or lang="fr" in your retrieval query.
