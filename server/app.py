"""
SQLens — FastAPI Server
OpenEnv backend + AI SQL Analyzer UI
"""
from __future__ import annotations
import sys, os, json, uuid, re
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Optional, List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
except ImportError:
    pass

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from server.sql_environment import SQLQueryEnvironment
from models import SQLAction
from tasks import TASK_REGISTRY, SCHEMA_INFO, ALL_TASK_IDS

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")

_envs: dict = {}
_history: List[dict] = []
_feedback: List[dict] = []
_DEFAULT_TASK = "easy_select"


def _env(task_id: str) -> SQLQueryEnvironment:
    if task_id not in _envs:
        e = SQLQueryEnvironment()
        e.reset(task_id=task_id)
        _envs[task_id] = e
    return _envs[task_id]


@asynccontextmanager
async def lifespan(app: FastAPI):
    for tid in ALL_TASK_IDS:
        e = SQLQueryEnvironment()
        e.reset(task_id=tid)
        _envs[tid] = e
    print(f"[SQLens] Ready — {len(_envs)} tasks loaded", flush=True)
    yield


app = FastAPI(title="SQLens — AI SQL Analyzer", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
_UI_FILE    = os.path.join(_STATIC_DIR, "index.html")


def _read_ui() -> str:
    try:
        with open(_UI_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "<h1>SQLens</h1><p>UI not found. Use <a href='/docs'>/docs</a>.</p>"


class ResetReq(BaseModel):
    task_id: Optional[str] = "easy_select"
    task:    Optional[str] = None
    seed:    Optional[int] = 42

class StepReq(BaseModel):
    query:           Optional[str] = None
    optimized_query: Optional[str] = None
    task_id:         Optional[str] = None
    task:            Optional[str] = None

class AnalyzeReq(BaseModel):
    query:   str
    dialect: str = "postgres"
    context: Optional[str] = None

class FeedbackReq(BaseModel):
    analysisId: str
    rating:     str
    comment:    Optional[str] = None

class RunReq(BaseModel):
    query: str

class SuggestReq(BaseModel):
    query: str
    cursor_pos: Optional[int] = None


@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(content=_read_ui(), status_code=200,
        headers={"Cache-Control":"no-cache, no-store, must-revalidate","Pragma":"no-cache","Expires":"0"})

@app.get("/health")
def health():
    return {"status": "ok", "env": "sqlens", "version": "2.0.0", "tasks": len(_envs)}

@app.get("/healthz")
def healthz():
    return {"status": "ok", "env": "sqlens", "version": "2.0.0", "tasks": len(_envs)}

@app.get("/api/info")
def api_info():
    return {"name":"SQLens","version":"2.0.0","tasks":ALL_TASK_IDS}

@app.post("/reset")
def reset(req: ResetReq = None):
    if req is None: req = ResetReq()
    task_id = req.task or req.task_id or _DEFAULT_TASK
    if task_id not in TASK_REGISTRY:
        raise HTTPException(400, detail=f"Unknown task '{task_id}'. Available: {ALL_TASK_IDS}")
    obs = _env(task_id).reset(task_id=task_id, seed=req.seed or 42)
    return {k: getattr(obs, k) for k in [
        "task_description","schema_info","query_result","execution_error",
        "reward","done","partial_scores","hint","step_number","max_steps",
        "task_name","current_query","dialect","feedback","schema_context"]}

@app.post("/step")
def step(req: StepReq):
    task_id = req.task or req.task_id or _DEFAULT_TASK
    if task_id not in TASK_REGISTRY:
        raise HTTPException(400, detail=f"Unknown task '{task_id}'")
    env = _env(task_id)
    if env._state.step_count >= env._state.max_steps or env._state.task_solved:
        env.reset(task_id=task_id)
    sql = req.optimized_query or req.query or ""
    res = env.step(SQLAction(query=sql, optimized_query=sql))
    obs = res.observation
    return {
        "observation": {k: getattr(obs, k) for k in [
            "task_description","schema_info","query_result","execution_error",
            "reward","done","partial_scores","hint","step_number","max_steps",
            "task_name","current_query","dialect","last_reward","cumulative_reward",
            "feedback","schema_context"]},
        "reward": res.reward, "done": res.done, "info": res.info,
    }

@app.get("/state")
def state(task_id: Optional[str] = None, task: Optional[str] = None):
    tid = task or task_id or _DEFAULT_TASK
    if tid not in TASK_REGISTRY:
        raise HTTPException(400, detail=f"Unknown task '{tid}'")
    s = _env(tid).state()
    return {k: getattr(s, k) for k in [
        "episode_id","step_count","current_task","task_difficulty",
        "max_steps","attempts","best_reward","task_solved","started_at"]}

@app.get("/tasks")
def list_tasks():
    return {"tasks": [
        {"task_id":t.task_id,"name":t.task_id,"title":t.title,
         "difficulty":t.difficulty,"description":t.description,
         "task_type":t.task_type,"dialect":t.dialect,
         "max_steps":t.max_steps,"success_threshold":t.success_threshold,
         "tags":t.tags,
         "original_query":t.original_query if t.task_type=="optimizer" else "",
         "schema_context":t.schema_context or ""}
        for t in TASK_REGISTRY.values()
    ]}

@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    local = SQLQueryEnvironment()
    try:
        while True:
            msg = json.loads(await websocket.receive_text())
            m, p = msg.get("method"), msg.get("params", {})
            if m == "reset":
                obs = local.reset(task_id=p.get("task_id","easy_select"))
                await websocket.send_text(json.dumps({"type":"reset","observation":
                    {k:getattr(obs,k) for k in ["task_description","schema_info",
                     "reward","done","partial_scores","hint","step_number","max_steps",
                     "task_name","current_query","feedback"]}}))
            elif m == "step":
                sql = p.get("query","") or p.get("optimized_query","")
                r = local.step(SQLAction(query=sql, optimized_query=sql))
                obs = r.observation
                await websocket.send_text(json.dumps({"type":"step","observation":
                    {k:getattr(obs,k) for k in ["query_result","execution_error",
                     "reward","done","partial_scores","hint","step_number","max_steps",
                     "feedback","current_query"]},
                    "reward":r.reward,"done":r.done,"info":r.info}))
    except WebSocketDisconnect:
        local.close()


def _clean_sql_output(text: str) -> str:
    if not text: return ""
    t = text.strip()
    t = t.replace('&quot;','"').replace('&gt;','>').replace('&lt;','<').replace('&amp;','&')
    t = re.sub(r'&#\d+;','',t); t = re.sub(r'&[a-z]+;','',t)
    t = t.replace('\uff1e','>').replace('\uff1c','<')
    t = re.sub(r'\d+["\']\s*>','',t); t = re.sub(r'\d{3,}>','',t)
    for ch in ['\u201c','\u201d','\u02ba','\u02dd','\uff02']: t = t.replace(ch,'"')
    for ch in ['\u2018','\u2019','\u02bc','\u0060','\uff07']: t = t.replace(ch,"'")
    t = re.sub(r'<[^>]+>','',t)
    t = re.sub(r'```[a-zA-Z]*','',t)
    clean_lines = []
    for line in t.split('\n'):
        prev = None
        while prev != line:
            prev = line
            line = re.sub(r'^\s*\d+\s*["\']?\s*>','',line)
        clean_lines.append(line)
    t = '\n'.join(clean_lines)
    t = re.sub(r'\d+\s*["\']?\s*>','',t)
    m = re.search(r'(?:SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|--)',t,re.I)
    if m: t = t[m.start():].strip()
    else: return ""
    t = re.sub(r'\b\d{3,}\b\s*["\']?\s*>','',t)
    t = re.sub(r'\n{3,}','\n\n',t)
    t = re.sub(r'\d+["\']\s*>','',t); t = re.sub(r'\d{3,}>','',t)
    t = re.sub(r'  +',' ',t)
    t = re.sub(r'<[a-z][a-z_0-9]*>','',t)
    return t.strip()


def _ai_analyze(query: str, dialect: str, context: Optional[str]) -> dict:
    from openai import OpenAI
    ctx = f"\nSchema context:\n{context}\n" if context else ""
    prompt = f"""You are a world-class SQL expert. Analyze this SQL query and return ONLY valid JSON.

SQL DIALECT: {dialect}
QUERY:
```sql
{query}
```
{ctx}

Check ALL of these:
SECURITY: SQL injection (string concat), OR 1=1, DELETE/UPDATE without WHERE, exposed sensitive columns
PERFORMANCE: SELECT *, YEAR()/MONTH() on indexed columns, leading LIKE '%', correlated subqueries, missing LIMIT
CORRECTNESS: syntax errors, wrong aggregation, NULL handling, wrong JOIN type
READABILITY: missing aliases, inconsistent casing, missing LIMIT

SCORING: Start at 100. Deduct: critical=-30, warning=-15, info=-5.
rewardScore = round((performanceScore + securityScore + readabilityScore) / 3)
Any critical issue → max score 40. SELECT * → max 85. YEAR()/MONTH() → max 80.

optimizedQuery MUST: start with SELECT/WITH/INSERT/UPDATE/DELETE, fix ALL issues, be valid executable SQL.

Return ONLY this JSON:
{{
  "optimizedQuery": "<valid SQL only>",
  "rewardScore": <0-100>,
  "performanceScore": <0-100>,
  "securityScore": <0-100>,
  "readabilityScore": <0-100>,
  "issues": [{{"severity":"critical|warning|info","category":"performance|security|readability|correctness","message":"<problem>","suggestion":"<fix>","lineHint":"<clause>"}}],
  "explanation": "<2-3 sentences>",
  "realWorldImpact": "<production consequence>"
}}"""

    llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN, max_retries=1)
    raw = llm.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role":"system","content":"You are a world-class SQL expert. Return ONLY valid JSON. optimizedQuery must contain ONLY valid SQL."},
            {"role":"user","content":prompt}
        ],
        max_tokens=1600, temperature=0.0,
    ).choices[0].message.content.strip()

    m = re.search(r'\{[\s\S]*\}', raw)
    if not m: raise ValueError("No JSON in response")
    json_str = m.group(0)
    json_str = re.sub(r'\\n\d+\\?">', r'\\n', json_str)
    json_str = re.sub(r'\\n\d+\\?\'>', r'\\n', json_str)
    p = json.loads(json_str)
    clamp = lambda v: min(100, max(0, round(float(v or 50))))
    raw_opt = _clean_sql_output(str(p.get("optimizedQuery") or "").strip())
    return {
        "optimizedQuery":   raw_opt,
        "rewardScore":      clamp(p.get("rewardScore",   50)),
        "performanceScore": clamp(p.get("performanceScore", 50)),
        "securityScore":    clamp(p.get("securityScore",  50)),
        "readabilityScore": clamp(p.get("readabilityScore", 50)),
        "issues":           (p.get("issues") or [])[:10],
        "explanation":      str(p.get("explanation", "")),
        "realWorldImpact":  str(p.get("realWorldImpact", "")),
    }


def _rule_analyze(query: str) -> dict:
    lower = query.lower()
    issues = []
    perf, sec, read = 100, 100, 100

    syntax_err = _check_syntax(query)
    if syntax_err:
        return {"optimizedQuery":"","rewardScore":0,"performanceScore":0,"securityScore":0,"readabilityScore":0,
                "issues":[{"severity":"critical","category":"correctness","message":f"Syntax Error: {syntax_err}",
                           "suggestion":_get_query_suggestion(query,syntax_err),"lineHint":"Query structure"}],
                "explanation":f"Syntax error: {syntax_err}","realWorldImpact":"Query will fail immediately."}

    if re.search(r"or\s+1\s*=\s*1|or\s+'1'\s*=\s*'1'", query, re.I):
        issues.append({"severity":"critical","category":"security","message":"SQL injection (OR 1=1)","suggestion":"Use parameterized queries","lineHint":"WHERE clause"})
        sec -= 60
    if re.search(r"'\s*\|\||\|\|\s*'", query):
        issues.append({"severity":"critical","category":"security","message":"String concatenation — SQL injection risk","suggestion":"Use $1, $2 parameterized placeholders","lineHint":"String concat"})
        sec -= 50
    if re.search(r'select\s+\*', query, re.I):
        issues.append({"severity":"warning","category":"performance","message":"SELECT * fetches all columns — wastes I/O, prevents index-only scans","suggestion":"List only needed columns","lineHint":"SELECT *"})
        perf -= 20; read -= 5
    if re.search(r"year\s*\(|month\s*\(|date_format\s*\(", query, re.I):
        issues.append({"severity":"warning","category":"performance","message":"YEAR()/MONTH() on indexed column causes full table scan","suggestion":"Use range: col >= '2024-03-01' AND col < '2024-04-01'","lineHint":"YEAR()/MONTH()"})
        perf -= 25
    if re.search(r"like\s+['\"]%", query, re.I):
        issues.append({"severity":"warning","category":"performance","message":"Leading wildcard LIKE defeats index","suggestion":"Use full-text search or trailing wildcard","lineHint":"LIKE '%...'"})
        perf -= 20
    if re.search(r'delete\s+from\s+\w+', query, re.I) and "where" not in lower:
        issues.append({"severity":"critical","category":"correctness","message":"DELETE without WHERE deletes ALL rows","suggestion":"Add a WHERE clause","lineHint":"DELETE missing WHERE"})
        perf -= 40; sec -= 20
    if re.search(r'update\s+\w+\s+set', query, re.I) and "where" not in lower:
        issues.append({"severity":"critical","category":"correctness","message":"UPDATE without WHERE modifies ALL rows","suggestion":"Add a WHERE clause","lineHint":"UPDATE missing WHERE"})
        perf -= 30; sec -= 20
    if "limit" not in lower and "select" in lower and "where" not in lower and "group by" not in lower:
        issues.append({"severity":"info","category":"performance","message":"No LIMIT — may return large result sets","suggestion":"Add LIMIT N","lineHint":"Consider LIMIT"})
        perf -= 5

    perf = max(0, min(100, perf))
    sec  = max(0, min(100, sec))
    read = max(0, min(100, read))
    reward = round((perf + sec + read) / 3)
    opt_query = _build_optimized_query(query, issues) if issues else query
    crit = any(i["severity"] == "critical" for i in issues)
    if not issues: explanation, real_impact = "Well-structured query with no detected issues.", "No issues — production ready."
    elif crit: explanation, real_impact = "Critical issues found that must be fixed before production.", "Risk of data loss or security breach."
    else: explanation, real_impact = "Minor issues found. Query works but could be improved.", "Performance concerns under load."
    return {"optimizedQuery":opt_query,"rewardScore":reward,"performanceScore":perf,
            "securityScore":sec,"readabilityScore":read,"issues":issues,
            "explanation":explanation,"realWorldImpact":real_impact}


def _build_optimized_query(query: str, issues: list) -> str:
    opt = query.strip()
    issue_msgs = " ".join(i.get("message","") for i in issues).lower()
    if re.search(r'select\s+\*', opt, re.I):
        tbl_match = re.search(r'from\s+(\w+)', opt, re.I)
        tbl = tbl_match.group(1).lower() if tbl_match else ""
        col_map = {
            "customers":"customer_id, name, email, country, city, tier",
            "products":"product_id, name, category, price, stock",
            "orders":"order_id, customer_id, order_date, status, discount_pct",
            "order_items":"item_id, order_id, product_id, quantity, unit_price",
            "reviews":"review_id, customer_id, product_id, rating, review_date",
            "users":"id, username, email, role","sessions":"id, user_id, created_at, expires_at",
        }
        cols = col_map.get(tbl, "id, name, created_at")
        opt = re.sub(r'select\s+\*', f'SELECT {cols}', opt, flags=re.I)
    if re.search(r'year\s*\((\w+)\)\s*=\s*(\d{4})', opt, re.I):
        m = re.search(r'year\s*\((\w+)\)\s*=\s*(\d{4})', opt, re.I)
        col, yr = m.group(1), m.group(2)
        opt = re.sub(r'year\s*\(\w+\)\s*=\s*\d{4}', f"{col} >= '{yr}-01-01' AND {col} < '{int(yr)+1}-01-01'", opt, flags=re.I)
        opt = re.sub(r'\s*and\s*month\s*\(\w+\)\s*=\s*\d+','',opt,flags=re.I)
    if re.search(r"'\s*\|\||\|\|\s*'", opt):
        lower_opt2 = opt.lower()
        if 'username' in lower_opt2 and 'password' in lower_opt2:
            opt = "SELECT id, username, role\nFROM users\nWHERE username = $1\n  AND password_hash = $2"
        else:
            # Generic: replace all '' || var || '' patterns with $N placeholders
            param_n = [0]
            def _replace_concat(m):
                param_n[0] += 1
                return f'${param_n[0]}'
            opt = re.sub(r"''\s*\|\|\s*\w+\s*\|\|\s*''", _replace_concat, opt)
            opt = re.sub(r"'\s*\|\|\s*\w+\s*\|\|\s*'", _replace_concat, opt)
    if re.search(r"or\s+1\s*=\s*1", opt, re.I):
        opt = re.sub(r"\s*or\s+1\s*=\s*1", '', opt, flags=re.I)
        if 'username' in opt.lower():
            opt = "SELECT id, username, role\nFROM users\nWHERE username = $1\n  AND password_hash = $2"
    if re.search(r"like\s+['\"]%(\w+)", opt, re.I):
        m = re.search(r"like\s+['\"]%(\w+)['\"]", opt, re.I)
        if m: opt = re.sub(r"like\s+['\"]%\w+['\"]", f"LIKE '{m.group(1)}%'", opt, flags=re.I)
    lower_opt = opt.lower()
    if ('select' in lower_opt and 'limit' not in lower_opt and 'group by' not in lower_opt
            and not re.search(r'delete|update|insert', lower_opt)):
        opt = opt.rstrip(';').rstrip() + '\nLIMIT 100;'
    if re.search(r'delete\s+from\s+\w+\s*$', opt, re.I):
        opt = opt.rstrip(';') + "\nWHERE id = $1;"
    if re.search(r'update\s+\w+\s+set\b', opt, re.I) and 'where' not in opt.lower():
        opt = opt.rstrip(';') + "\nWHERE id = $1;"
    return opt


def _check_syntax(query: str) -> Optional[str]:
    q = query.strip().lower()
    if not q: return "Empty query"
    sq = query.count("'") - query.count("\\'")
    dq = query.count('"') - query.count('\\"')
    if sq % 2 != 0: return "Unclosed single quote (') in query"
    if dq % 2 != 0: return "Unclosed double quote (\") in query"
    op = query.count('('); cp = query.count(')')
    if op != cp: return f"Mismatched parentheses: {op} opening, {cp} closing"
    if q.startswith('select') and 'from' not in q and not re.search(r'select\s+\d+|select\s+\'',q):
        return "SELECT statement missing FROM clause"
    if 'join' in q and 'on' not in q and 'using' not in q and not re.search(r'\bcross\s+join\b',q):
        return "JOIN clause missing ON condition"
    return None


def _get_query_suggestion(query: str, error: str) -> str:
    if "unclosed" in error.lower() and "quote" in error.lower():
        return "Check string literals — make sure all quotes are properly closed."
    if "parentheses" in error.lower():
        return "Count your ( and ) — they must match."
    if "from clause" in error.lower():
        return "Add FROM clause: SELECT name FROM customers"
    if "join" in error.lower() and "on" in error.lower():
        return "Add ON condition: JOIN orders ON customers.customer_id = orders.customer_id"
    return "Check your SQL syntax and try again"


def _get_error_suggestion(error: str, query: str) -> str:
    err_lower = error.lower()
    if "no such table" in err_lower:
        return "Available tables: customers, products, orders, order_items, reviews"
    if "no such column" in err_lower:
        m = re.search(r"no such column:\s*(\w+)", err_lower)
        return f"Column '{m.group(1)}' not found. Check schema." if m else "Column not found. Check schema."
    if "ambiguous" in err_lower:
        return "Column is ambiguous — use table prefix: table_name.column_name"
    if "misuse of aggregate" in err_lower:
        return "Aggregate functions need GROUP BY when mixed with non-aggregated columns"
    if "syntax error" in err_lower:
        m = re.search(r'near "([^"]+)"', error)
        return f"Syntax error near '{m.group(1)}'." if m else "SQL syntax error."
    return "Query execution failed. Check SQL syntax and table/column names."


@app.post("/api/sql/analyze")
def analyze(req: AnalyzeReq):
    if not req.query or len(req.query.strip()) < 5:
        return JSONResponse({"error": "Query too short"}, status_code=400)
    ai_data = None
    try:
        ai_data = _ai_analyze(req.query, req.dialect, req.context)
    except Exception:
        pass
    rule_data = _rule_analyze(req.query)
    if ai_data:
        data = ai_data
        opt = (data.get("optimizedQuery") or "").strip()
        if not opt or not re.match(r'^\s*(SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|--)', opt, re.I):
            data["optimizedQuery"] = rule_data.get("optimizedQuery") or req.query
        if not data.get("issues") and rule_data.get("issues"):
            data["issues"] = rule_data["issues"]
            data["rewardScore"] = rule_data["rewardScore"]
            data["performanceScore"] = rule_data["performanceScore"]
            data["securityScore"] = rule_data["securityScore"]
            data["readabilityScore"] = rule_data["readabilityScore"]
    else:
        data = rule_data
    opt = (data.get("optimizedQuery") or "").strip()
    if data.get("rewardScore", 100) < 100 and (not opt or opt == req.query.strip()):
        data["optimizedQuery"] = _build_optimized_query(req.query, data.get("issues", []))
    data["optimizedQuery"] = _clean_sql_output(data.get("optimizedQuery") or "")
    data["optimizedQuery"] = re.sub(r'\d+["\']\s*>', '', data.get("optimizedQuery", ""))
    data["optimizedQuery"] = re.sub(r'\d{3,}>', '', data.get("optimizedQuery", ""))
    record = {"id":str(uuid.uuid4()),"originalQuery":req.query,"dialect":req.dialect,
              "analyzedAt":datetime.now(timezone.utc).isoformat(),"feedbackRating":None,**data}
    _history.insert(0, record)
    if len(_history) > 200: _history.pop()
    return record


@app.post("/api/sql/feedback")
def feedback(req: FeedbackReq):
    for r in _history:
        if r["id"] == req.analysisId:
            r["feedbackRating"] = req.rating
            break
    _feedback.append({"analysisId":req.analysisId,"rating":req.rating})
    return {"success":True,"message":"Feedback recorded — thank you!"}


@app.get("/api/sql/history")
def history():
    items = [{"id":r["id"],"queryPreview":r["originalQuery"][:120],"rewardScore":r["rewardScore"],
              "dialect":r["dialect"],"issueCount":len(r.get("issues",[])),"analyzedAt":r["analyzedAt"],
              "feedbackRating":r.get("feedbackRating")} for r in _history[:50]]
    return {"items":items,"total":len(items)}


@app.get("/api/sql/stats")
def stats():
    total = len(_history)
    avg   = round(sum(r["rewardScore"] for r in _history)/total) if total else 0
    tl_fb = len(_feedback)
    helpful = sum(1 for f in _feedback if f["rating"]=="helpful")
    hp_pct  = round(helpful/tl_fb*100) if tl_fb else 0
    dist = [{"range":"0-20","count":0},{"range":"21-40","count":0},{"range":"41-60","count":0},
            {"range":"61-80","count":0},{"range":"81-100","count":0}]
    for r in _history:
        s = r["rewardScore"]
        if   s<=20: dist[0]["count"]+=1
        elif s<=40: dist[1]["count"]+=1
        elif s<=60: dist[2]["count"]+=1
        elif s<=80: dist[3]["count"]+=1
        else:       dist[4]["count"]+=1
    return {"totalAnalyzed":total,"averageRewardScore":avg,"totalFeedbackGiven":tl_fb,
            "helpfulFeedbackPercent":hp_pct,"scoreDistribution":dist}


@app.post("/api/sql/suggest")
def suggest(req: SuggestReq):
    q = (req.query or "").strip()
    if not q: return {"suggestions":[],"hint":""}
    lower = q.lower()
    suggestions, hint = [], ""
    if re.search(r'select\s+\*', q, re.I):
        hint = "💡 Replace SELECT * with specific column names for better performance."
        suggestions.append({"type":"warning","text":"Replace SELECT * with explicit columns"})
    if re.search(r"year\s*\(|month\s*\(", q, re.I):
        hint = "⚡ YEAR()/MONTH() on indexed columns causes full table scans. Use date ranges."
        suggestions.append({"type":"performance","text":"Use date range: col >= '2024-01-01' AND col < '2025-01-01'"})
    if re.search(r"'\s*\|\||\|\|\s*'", q):
        hint = "🔒 String concatenation in SQL is a SQL injection risk. Use $1, $2 placeholders."
        suggestions.append({"type":"security","text":"Use parameterized queries: WHERE username = $1"})
    if re.search(r"like\s+['\"]%", q, re.I):
        hint = "⚡ Leading wildcard LIKE '%value' defeats indexes."
        suggestions.append({"type":"performance","text":"Use trailing wildcard: LIKE 'value%'"})
    if re.search(r'delete\s+from\s+\w+\s*$', q, re.I) and 'where' not in lower:
        hint = "🚨 DELETE without WHERE will delete ALL rows!"
        suggestions.append({"type":"critical","text":"Add WHERE clause to DELETE statement"})
    if re.search(r'update\s+\w+\s+set', q, re.I) and 'where' not in lower:
        hint = "🚨 UPDATE without WHERE will modify ALL rows!"
        suggestions.append({"type":"critical","text":"Add WHERE clause to UPDATE statement"})
    syntax_err = _check_syntax(q)
    if syntax_err:
        hint = f"❌ Syntax: {syntax_err}"
        suggestions.insert(0, {"type":"error","text":syntax_err})
    return {"suggestions":suggestions[:5],"hint":hint}


@app.get("/api/sql/examples")
def examples():
    return {"examples":[
        {"label":"🔒 SQL Injection","query":"SELECT id, username, role\nFROM users\nWHERE username = '' || user_input || ''\n  AND password_hash = '' || pass_input || ''","dialect":"postgres","hint":"Fix: use $1, $2 parameterized placeholders"},
        {"label":"⚡ Full Table Scan","query":"SELECT *\nFROM orders\nWHERE YEAR(created_at) = 2024\n  AND MONTH(created_at) = 3\nORDER BY total_amount DESC","dialect":"mysql","hint":"Fix: use date range, replace SELECT *, add LIMIT 100"},
        {"label":"🔄 N+1 Subqueries","query":"SELECT\n  (SELECT email FROM users WHERE id = o.customer_id),\n  (SELECT name  FROM users WHERE id = o.customer_id),\n  SUM(o.total_amount)\nFROM orders o\nWHERE (SELECT email FROM users WHERE id = o.customer_id) LIKE '%@gmail.com'\nGROUP BY o.customer_id\nORDER BY SUM(o.total_amount) DESC","dialect":"postgres","hint":"Fix: replace correlated subqueries with JOIN, fix LIKE, add LIMIT 50"},
        {"label":"🗑️ DELETE No WHERE","query":"DELETE FROM sessions","dialect":"postgres","hint":"Danger: deletes ALL rows — add WHERE clause"},
        {"label":"✅ Clean Query","query":"SELECT u.id, u.name, u.email,\n       COUNT(o.id) AS order_count,\n       SUM(o.total_amount) AS total_spend\nFROM users u\nLEFT JOIN orders o ON o.user_id = u.id\n  AND o.status = 'completed'\nWHERE u.active = true\nGROUP BY u.id, u.name, u.email\nORDER BY total_spend DESC\nLIMIT 20","dialect":"postgres","hint":"Well-optimized query — should score 100"},
        {"label":"🔍 Leading Wildcard","query":"SELECT id, email, name\nFROM users\nWHERE email LIKE '%@gmail.com'\nORDER BY created_at DESC","dialect":"postgres","hint":"Fix: use RIGHT(email, 10) = '@gmail.com'"},
    ]}


@app.post("/api/sql/run")
def run_sql(req: RunReq):
    from tasks import make_fresh_db, run_query, SCHEMA_INFO
    if not req.query or len(req.query.strip()) < 3:
        return JSONResponse({"error":"Query too short"}, status_code=400)
    query = req.query.strip()
    syntax_err = _check_syntax(query)
    if syntax_err:
        return {"success":False,"error":f"Syntax Error: {syntax_err}","rows":[],"columns":[],"rowCount":0,
                "syntaxError":True,"suggestion":_get_query_suggestion(query,syntax_err)}
    conn = make_fresh_db()
    try:
        rows, err = run_query(conn, query)
    finally:
        conn.close()
    if err:
        return {"success":False,"error":err,"rows":[],"columns":[],"rowCount":0,
                "suggestion":_get_error_suggestion(err,query)}
    cols = list(rows[0].keys()) if rows else []
    return {"success":True,"rows":rows[:100],"columns":cols,"rowCount":len(rows),"error":"","schemaInfo":SCHEMA_INFO}
