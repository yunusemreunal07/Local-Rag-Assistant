# Production Deployment Recommendation
## Wikipedia RAG Assistant

**Version:** 1.0  
**Date:** 2026-05-04  
**Audience:** Engineering leads, DevOps, product stakeholders

---

## 1. Current Architecture & Its Limitations

The local prototype runs entirely on a single developer machine. While this is excellent for development, experimentation, and coursework, it has several structural limitations that prevent production use.

| Limitation | Impact |
|------------|--------|
| Single-process ChromaDB | Cannot handle concurrent queries; no horizontal scaling |
| Ollama on localhost | One model instance; no load balancing; GPU-bound |
| SQLite file-based DB | Not safe for concurrent writes; no replication |
| No authentication | Anyone with network access can query the system |
| No observability | No logging, tracing, or alerting |
| Ingestion is manual CLI | Cannot be triggered automatically or by end-users |
| Data freshness | Wikipedia content frozen at ingestion time |

---

## 2. Recommended Production Stack

### 2.1 Infrastructure Overview

```
                         ┌────────────────────────────────────┐
                         │           Load Balancer            │
                         │         (AWS ALB / nginx)          │
                         └──────────────┬─────────────────────┘
                                        │
                    ┌───────────────────▼───────────────────────┐
                    │             API Layer (FastAPI)            │
                    │   /chat   /ingest   /health   /admin       │
                    │       (containerised, horizontally         │
                    │        scalable, stateless)                │
                    └──────┬──────────────┬────────────┬────────┘
                           │              │            │
              ┌────────────▼──┐   ┌───────▼───┐  ┌────▼──────────┐
              │  Managed      │   │ PostgreSQL │  │  LLM Serving  │
              │  Vector DB    │   │  + pgvector│  │  (vLLM /      │
              │  (Pinecone    │   │  metadata  │  │   TGI /       │
              │   or Qdrant   │   │  & audit   │  │   Ollama      │
              │   Cloud)      │   │   logs     │  │   cluster)    │
              └───────────────┘   └───────────┘  └───────────────┘
```

### 2.2 Component Recommendations

#### Vector Database → Pinecone Serverless or Qdrant Cloud

**Why replace ChromaDB?**
- ChromaDB's `PersistentClient` is single-process. The `HttpClient` variant is better but still requires a self-managed server.
- Managed services like **Pinecone Serverless** or **Qdrant Cloud** provide:
  - Automatic scaling (reads and writes)
  - Built-in replication and high availability
  - REST + gRPC APIs
  - Metadata filtering (same capability we rely on today)
  - SLA-backed uptime

**Migration effort:** Low. The retriever only calls `collection.query()` — swap the client initialisation to point at the managed endpoint. The rest of the code is identical.

#### Relational Store → PostgreSQL (managed: AWS RDS / Supabase)

**Why replace SQLite?**
- SQLite is single-writer only and file-based — unsuitable for concurrent API requests.
- PostgreSQL supports concurrent connections, ACID transactions, row-level locking, and JSONB for semi-structured metadata.
- AWS RDS or Supabase provide automated backups, point-in-time recovery, and read replicas.

**Optional enhancement:** Use `pgvector` as a secondary vector store for hybrid BM25 + dense retrieval without an additional service.

#### LLM Serving → vLLM or Text Generation Inference (TGI)

**Why replace Ollama?**
- Ollama is designed for single-user developer use. It processes one request at a time.
- For concurrent users, use **vLLM** or **HuggingFace TGI**:
  - Continuous batching: multiple requests served in a single forward pass.
  - PagedAttention (vLLM): GPU memory utilised up to ~95% vs ~40% with naïve serving.
  - OpenAI-compatible REST API: minimal code change required (just update the base URL).
  - Horizontal scaling: run multiple replicas behind a load balancer.

**Recommended model upgrade for production:** `llama3.1:8b` or `Mistral-7B-Instruct-v0.3` for better instruction-following. For highest quality, consider a quantized `llama3.1:70b` on an A100 instance.

#### Embedding Service → Dedicated microservice or AWS Bedrock Titan Embeddings

**Option A (keep local model):** Package `all-MiniLM-L6-v2` into a FastAPI microservice. Horizontally scalable, no per-call cost, latency ~5 ms/query on CPU.

**Option B (managed API):** Use AWS Bedrock Titan Embeddings (`amazon.titan-embed-text-v2`) or Cohere Embed. Eliminates model hosting but adds per-call cost (~$0.0001/1K tokens).

**Recommendation:** Option A for high-volume or cost-sensitive deployments; Option B for rapid prototyping or low-volume internal tools.

#### Application Layer → FastAPI + Uvicorn

Expose the RAG pipeline as a REST API:

```
POST /chat          { query: str, session_id: str? }
POST /ingest        { entity: str, type: str }  (admin only)
GET  /health        returns component statuses
GET  /stats         ingestion statistics
```

Benefits over a Streamlit-only interface:
- Decouples frontend from backend (React, mobile apps, Slack bots can all consume the same API).
- Enables proper authentication (OAuth2 / JWT).
- Supports async request handling and connection pooling.

Keep Streamlit as an internal admin/demo UI, or replace it with a React/Next.js frontend.

---

## 3. Scaling Considerations

### Read scaling (query traffic)
- The API layer is stateless → add replicas behind the load balancer.
- ChromaDB/Pinecone queries are read-heavy → managed vector DBs handle this automatically.
- Cache frequent queries in Redis (TTL 1 hour) to avoid redundant embedding + vector search.

### Write scaling (ingestion)
- Move ingestion to an async background job queue (Celery + Redis, or AWS SQS + Lambda).
- Trigger nightly Wikipedia re-ingestion to keep content fresh.
- Use a diff-based approach: only re-ingest articles whose Wikipedia revision ID has changed.

### LLM throughput
- vLLM on a single A10G GPU (24 GB VRAM) can serve ~100 req/min for a 7B model.
- For higher throughput, use tensor parallelism across multiple GPUs or scale replicas.
- Implement request queuing to shed load gracefully under peak traffic.

---

## 4. Cost Analysis

| Component | Local (prototype) | Production (small scale, ~1 K req/day) |
|-----------|-------------------|----------------------------------------|
| Vector DB | $0 (disk) | ~$70/mo (Pinecone Starter) or $0 (Qdrant Cloud free tier) |
| LLM serving | $0 (local GPU/CPU) | ~$200–600/mo (g4dn.xlarge on AWS, spot pricing) |
| Embeddings | $0 (local CPU) | ~$0 (self-hosted microservice on t3.small ~$15/mo) |
| Relational DB | $0 (SQLite file) | ~$25/mo (RDS db.t3.micro) |
| API servers | $0 | ~$30/mo (2× t3.small auto-scaled) |
| **Total** | **$0** | **~$320–650/mo** |

> At 10 K req/day, LLM serving becomes the dominant cost. Consider quantized models (GGUF Q4) or API-based inference (Groq, Fireworks AI) at that scale.

---

## 5. Security & Compliance

| Concern | Recommendation |
|---------|---------------|
| API authentication | JWT tokens via OAuth2 (FastAPI `python-jose`) |
| Rate limiting | API gateway throttling (AWS API Gateway / nginx `limit_req`) |
| Input sanitisation | Validate and truncate query length (max 500 chars) |
| Prompt injection | Wrap user input in delimiters; never concatenate raw user text into system instructions |
| Data residency | Deploy to AWS region matching user geography for GDPR / CCPA compliance |
| Secrets management | AWS Secrets Manager / HashiCorp Vault for DB credentials and API keys |
| Dependency scanning | GitHub Dependabot or Snyk in CI/CD pipeline |

---

## 6. Observability

| Layer | Tool |
|-------|------|
| Structured logging | Python `structlog` → AWS CloudWatch / Datadog |
| Distributed tracing | OpenTelemetry → Jaeger or AWS X-Ray |
| LLM-specific metrics | LangFuse or Helicone (token usage, latency, error rate) |
| Alerting | PagerDuty on P95 latency > 10 s or error rate > 1% |
| Dashboard | Grafana with Prometheus exporters for all components |

---

## 7. Migration Path: Local → Cloud

### Phase 1 — Containerise (Week 1–2)
1. Write `Dockerfile` for the API layer and embedding service.
2. Write `docker-compose.yml` to run all services locally (API, ChromaDB server, Ollama, PostgreSQL).
3. Validate end-to-end tests pass in Docker.

### Phase 2 — Managed Database (Week 3)
1. Provision Pinecone Serverless index and RDS PostgreSQL instance.
2. Migrate data: run `ingest.py` pointing at the managed services.
3. Switch API to read from managed stores; verify query results match local baseline.

### Phase 3 — LLM Serving (Week 4–5)
1. Deploy vLLM on a GPU instance (AWS `g4dn.xlarge` or `g5.xlarge`).
2. Update `generator.py` base URL to point at the vLLM OpenAI-compatible endpoint.
3. Load-test with `locust` targeting 50 concurrent users.

### Phase 4 — Frontend & Auth (Week 6–7)
1. Build React chat frontend consuming the FastAPI backend.
2. Add OAuth2 authentication (Auth0 or AWS Cognito).
3. Deploy behind AWS ALB with HTTPS.

### Phase 5 — Automated Ingestion & Monitoring (Week 8)
1. Set up nightly ingestion cron job (AWS EventBridge → Lambda → SQS → ingestion worker).
2. Instrument all components with OpenTelemetry.
3. Configure PagerDuty alerts.

---

## 8. Summary Recommendation

For an **internal tool or academic demo** with < 50 users: keep the current local architecture, but containerise it with Docker for portability and reproducibility.

For a **production service with external users**: adopt the managed stack in Section 2 — Pinecone/Qdrant for vectors, PostgreSQL for metadata, vLLM for LLM serving, FastAPI for the API layer. The migration is incremental and the code changes required are minimal (mostly configuration, not logic).

The most impactful single change is **replacing Ollama with vLLM**: it unlocks concurrent request handling and is the bottleneck at any meaningful traffic volume.
