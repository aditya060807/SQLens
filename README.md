---
title: SQLens
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: AI SQL analyzer — score queries 0-100, detect issues
---

# SQLens — AI SQL Analyzer

AI-powered SQL analysis platform. Paste any SQL query and get:
- Quality score (0–100) across security, performance, and readability
- Detected issues with exact fixes
- Production-ready optimized rewrite

## OpenEnv Environment

6 tasks for AI agent training:
- **easy_select** — Indian Customer Directory (tutor, SQLite)
- **medium_join** — Customer Revenue Report (tutor, SQLite)
- **hard_subquery** — Category Champion Report (tutor, SQLite)
- **sql_injection_fix** — SQL Injection Fix (optimizer, PostgreSQL)
- **query_optimization** — Query Performance Optimization (optimizer, MySQL)
- **complex_rewrite** — Complex Query Rewrite (optimizer, PostgreSQL)

## API

```
POST /reset      — Start episode
POST /step       — Submit SQL, get reward
GET  /state      — Episode state
GET  /tasks      — List all tasks
GET  /health     — Health check
POST /api/sql/analyze — Full AI analysis
```

## Environment Variables

```
HF_TOKEN=your_token
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
```
