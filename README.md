# SAAQ QA Assistant (Portfolio Project)

SAAQ QA Assistant is a simple **RAG (Retrieval-Augmented Generation)** project:

- **Backend**: FastAPI (`backend/`) exposes `POST /api/v1/chat`
- **Frontend**: Streamlit (`frontend/`) shows the answer + sources
- **Vector DB**: Weaviate (`weaviate` container)
- **Reverse proxy**: Nginx (`nginx` container)

This README focuses on **deploying to a VM using Docker Compose** (with Nginx).  
Prometheus/Grafana are intentionally **not** part of the deployment workflow for now.

---

## What you will deploy (services)

- `weaviate`: stores your vectors + document chunks (internal only)
- `backend`: FastAPI RAG API (internal only)
- `frontend`: Streamlit UI (internal only)
- `nginx`: single public entrypoint (port 80)

Nginx routes:
- `http://<VM-IP>/` → Streamlit UI
- `http://<VM-IP>/api/...` → FastAPI backend

---

## VM prerequisites

- A Linux VM (Ubuntu is fine)
- Docker + Docker Compose installed
- Firewall/security group:
  - open **80** (and **443** later if you add HTTPS)
  - optionally open **22** for SSH
  - keep **8080/8000/8501** closed publicly (Nginx is the entrypoint)

---

## Step 1: Clone the repo on the VM

```bash
git clone https://github.com/AbuUbaida/saaq_qa_assistant.git
cd saaq_qa_assistant
```

---

## Step 2: Create `.env` (required)

Create a `.env` file in the repo root:

```bash
cat > .env <<'EOF'
HF_API_KEY=<HUGGINGFACE_API_KEY>
FIRECRAWL_API_KEY=<FIRECRAWL_API_KEY>
EOF
```

Notes:
- `HF_API_KEY` is required for **query embeddings** + **LLM generation**.

---

## Step 3: Start the stack (with Nginx)

Build and run:

```bash
docker compose up -d --build weaviate backend frontend nginx
```

Check containers:

```bash
docker compose ps
```

At this point the UI should be reachable at:
- `http://<VM-IP>/`
  - Only Nginx is exposed. Backend/Frontend/Weaviate are internal to Docker.

---

## Step 4: One-time indexing (create collection + upload vectors)

The API will return a server error until the Weaviate collection exists.
To create it, you need to index embeddings into Weaviate.

### 4.1 Create processed chunks

```bash
python -m scripts.text_processor \
  --input data/raw/documents/pdf_documents/1_drivers_handbook.jsonl \
  --output-dir data/processed/pdf_documents
```

### 4.2 Create embeddings JSONL

```bash
python -m scripts.embedder \
  --input data/processed/pdf_documents/1_drivers_handbook.jsonl \
  --output-dir data/embeddings/pdf_documents
```

This creates a file like:
- `data/embeddings/pdf_documents/1_drivers_handbook[sentence-transformers_all-MiniLM-L6-v2].jsonl`

### 4.3 Index embeddings into Weaviate (HTML + PDF)

Run indexing twice: once for **HTML embeddings**, once for **PDF embeddings**.

```bash
# HTML embeddings
python -m scripts.index_embeddings --input-dir data/embeddings/html_documents

# PDF embeddings
python -m scripts.index_embeddings --input-dir data/embeddings/pdf_documents
```

---

## Step 5: Test the API (directly)

```bash
curl -X POST http://<VM-IP>/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is a probationary license in Quebec?",
    "search_method": "hybrid",
    "top_k": 5,
    "language": "en",
    "answer_style": "concise",
    "include_citations": true
  }'
```

Expected response shape:
- `answer`
- `sources` (citations)
- `retrieved_count`
- `latency`

---

## Operational notes (production-ish)

- **Logs**:

```bash
docker compose logs -f backend
docker compose logs -f frontend
docker compose logs -f nginx
docker compose logs -f weaviate
```

- **Restart after VM reboot**:

```bash
docker compose up -d
```

- **Backups / persistence**:
  - Weaviate data is stored in a Docker volume: `weaviate_data`

---

## Local development (optional)

If you want to run without Docker:
- run FastAPI on `:8000`
- run Streamlit on `:8501`
- run Weaviate separately (Docker recommended)
