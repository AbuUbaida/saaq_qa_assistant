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

- A Linux VM (Ubuntu 20.04+ recommended)
- Docker + Docker Compose installed
- Minimum resources:
  - **RAM**: 2 GB (4 GB recommended)
  - **CPU**: 2 vCPU (4 vCPU recommended)
  - **Disk**: 10 GB free space
- Firewall/security group:
  - open **80** (and **443** later if you add HTTPS)
  - optionally open **22** for SSH
  - keep **8080/8000/8501** closed publicly (Nginx is the entrypoint)

### Install Docker & Docker Compose (if not installed)

```bash
# Update package index
sudo apt-get update

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add your user to docker group (to run without sudo)
sudo usermod -aG docker $USER
# Log out and back in for this to take effect

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation
docker --version
docker compose version
```

---

## Step 1: Clone the repo on the VM

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd saaq_qa_assistant
```

---

## Step 2: Create `.env` (required)

Create a `.env` file in the repo root:

```bash
cat > .env <<'EOF'
HF_API_KEY=YOUR_HUGGINGFACE_API_KEY
EOF
```

Notes:

- `HF_API_KEY` is required for **query embeddings** + **LLM generation**.
- Get your API key from: [Hugging Face tokens](https://huggingface.co/settings/tokens)

---

## Step 3: Start the stack (with Nginx)

Build and run all services:

```bash
docker compose up -d --build weaviate backend frontend nginx
```

Check containers are running:

```bash
docker compose ps
```

All containers should show `Up` status. At this point the UI should be reachable at:

- `http://<VM-IP>/`
  - Only Nginx is exposed. Backend/Frontend/Weaviate are internal to Docker.

---

## Step 4: Prepare embeddings (auto-index on startup)

The backend now **auto-indexes** embeddings on startup. You do **not** need to run
manual indexing commands as long as the embeddings files exist in these folders:

- `data/embeddings/html_documents/*.jsonl`
- `data/embeddings/pdf_documents/*.jsonl`

If you don't have embeddings yet, follow the ingestion pipeline:

1. **Collect** → `scripts/web_collector.py` or `scripts/pdf_collector.py`
2. **Process** → `scripts/text_processor.py`
3. **Embed** → `scripts/embedder.py`

Auto-index runs only when the collection is empty. If you need to re-index,
delete the collection and restart the backend:

```bash
docker compose exec backend python -m scripts.delete_collection --collection SAAQDocuments --yes
docker compose restart backend
```

---

## Step 5: Test the API

### Via UI

Open your browser and navigate to: `http://<VM-IP>/`

Ask a question like: "What is a probationary license in Quebec?"

### Via curl (direct API test)

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

## Operational notes (production)

### View logs

```bash
# All services
docker compose logs -f

# Individual services
docker compose logs -f backend
docker compose logs -f frontend
docker compose logs -f nginx
docker compose logs -f weaviate
```

### Restart services

After VM reboot, services will auto-restart (thanks to `restart: unless-stopped` policy).

To manually restart:

```bash
docker compose restart
```

To restart a specific service:

```bash
docker compose restart backend
```

### Resource monitoring

Check container resource usage:

```bash
docker stats
```

### Backups / persistence

- Weaviate data is stored in a Docker volume: `weaviate_data`
- To backup Weaviate data:

  ```bash
  docker compose exec weaviate ls -la /var/lib/weaviate/backups
  ```

### Update application

```bash
# Pull latest code
git pull

# Rebuild and restart
docker compose up -d --build
```

---

## Troubleshooting

### Backend returns "could not find class SAAQDocuments in schema"

**Solution**: You haven't indexed data yet. Follow **Step 4** to index embeddings.

### Frontend shows "Request exceeded timeout"

**Solution**: The RAG pipeline is taking longer than 60 seconds. The timeout is set to 120 seconds in the frontend config. If queries consistently take longer, consider:

- Reducing `top_k` in requests
- Using a faster LLM model
- Optimizing your document chunks

### Containers won't start / OOM errors

**Solution**: Your VM may not have enough RAM. Check resource limits in `docker-compose.yml` and adjust based on your VM size.

### Can't access the UI

**Solution**:

1. Check firewall: `sudo ufw status` (port 80 should be open)
2. Check containers: `docker compose ps`
3. Check Nginx logs: `docker compose logs nginx`

---

## Local development (optional)

If you want to run without Docker:

1. Install Python 3.11+ and dependencies:

   ```bash
   pip install -r requirements.txt
   pip install -r backend/requirements.txt
   pip install -r frontend/requirements.txt
   ```

2. Run services separately:

   - Weaviate: `docker compose up -d weaviate` (or use cloud Weaviate)
   - Backend: `uvicorn backend.main:app --reload --port 8000`
   - Frontend: `streamlit run frontend/app.py --server.port 8501`

3. Access:

   - Frontend: `http://localhost:8501`
   - Backend API: `http://localhost:8000`
