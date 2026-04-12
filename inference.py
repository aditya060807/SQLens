"""
inference.py — SQLens Inference Script
Hackathon: Meta PyTorch × Hugging Face × Scaler

Mandatory stdout format:
  [START] task=<task> env=<env> model=<model>
  [STEP]  step=<n> action=<sql> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

Run:
  python inference.py
  python inference.py --base-url https://adit06-sqlens.hf.space
"""
from __future__ import annotations
import os, sys, time, textwrap, argparse, subprocess
from typing import List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

from openai import OpenAI

HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
API_BASE_URL = os.environ.get("API_BASE_URL",  "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",    "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK    = "queryopt-tutor"
MAX_STEPS    = 6
TEMPERATURE  = 0.2
MAX_TOKENS   = 512

SYSTEM_PROMPTS = {
    "easy_select": textwrap.dedent("""
        You are a SQL expert. Write a single correct SQLite SELECT query.
        Task: Retrieve name, city, tier of customers from India ordered by name ASC.
        Schema: customers(customer_id, name, email, country, city, joined_date, tier)
        Output ONLY raw SQL — no markdown, no explanation, no code fences.
    """).strip(),
    "medium_join": textwrap.dedent("""
        You are a SQL expert. Write a single correct SQLite SELECT query.
        Task: For each customer with at least one delivered order, show name and
        total revenue (SUM of quantity*unit_price). Order by total_revenue DESC.
        Tables: customers, orders (status='delivered'), order_items
        Output ONLY raw SQL — no markdown, no explanation, no code fences.
    """).strip(),
    "hard_subquery": textwrap.dedent("""
        You are a SQL expert. Write a single correct SQLite SELECT query using CTE + RANK().
        Task: For each product category, find the top-selling product by revenue.
        Show: category, product_name AS top_product, revenue, avg_rating.
        Tables: products, order_items, reviews (LEFT JOIN)
        Use WITH ... AS (...) and RANK() OVER (PARTITION BY category ORDER BY revenue DESC).
        Output ONLY raw SQL — no markdown, no explanation, no code fences.
    """).strip(),
    "sql_injection_fix": textwrap.dedent("""
        You are an expert SQL security engineer.
        Rewrite the given SQL query to use parameterized placeholders.
        For PostgreSQL: use $1, $2 placeholders.
        Remove ALL string concatenation. Keep query logic identical.
        Output ONLY raw SQL — no markdown, no explanation, no code fences.
    """).strip(),
    "query_optimization": textwrap.dedent("""
        You are a senior database performance engineer.
        Fix the given slow MySQL query:
        1. Replace YEAR()/MONTH() on created_at with range:
           created_at >= '2024-03-01' AND created_at < '2024-04-01'
        2. Replace SELECT * with: id, customer_id, total_amount, status
        3. Add LIMIT 100
        Output ONLY raw SQL — no markdown, no explanation, no code fences.
    """).strip(),
    "complex_rewrite": textwrap.dedent("""
        You are a PostgreSQL expert. Completely rewrite the given query:
        1. Replace ALL correlated subqueries with JOIN users u ON u.id = o.customer_id
        2. Fix LIKE '%@gmail.com' — use RIGHT(u.email, 10) = '@gmail.com'
        3. Add LIMIT 50
        4. Use GROUP BY u.id, u.email, u.name
        5. Add aliases: SUM(o.total_amount) AS total_spend, MAX(o.created_at) AS last_order_at
        Output ONLY raw SQL — no markdown, no explanation, no code fences.
    """).strip(),
}


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    safe = action.replace("\n","\\n").replace("\r","").strip()
    if len(safe) > 200: safe = safe[:200] + "..."
    print(f"[STEP] step={step} action={safe!r} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)


def get_action(client: OpenAI, task_id: str, current_query: str,
               feedback: str, schema: str, step: int, history: List[str]) -> str:
    system = SYSTEM_PROMPTS.get(task_id, SYSTEM_PROMPTS["query_optimization"])
    hist_block = "\n".join(history[-3:]) if history else "First attempt."
    user = textwrap.dedent(f"""
        === CURRENT SQL (step {step}) ===
        {current_query or '(none — write from scratch)'}

        === SCHEMA / CONTEXT ===
        {schema or 'See system prompt.'}

        === GRADER FEEDBACK ===
        {feedback or 'No feedback yet — submit your best attempt.'}

        === HISTORY ===
        {hist_block}

        Write the SQL query now (raw SQL only):
    """).strip()
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
        )
        text = (resp.choices[0].message.content or "").strip()
        text = "\n".join(l for l in text.split("\n") if not l.strip().startswith("```")).strip()
        return text or current_query or "SELECT 1"
    except Exception as e:
        print(f"[DEBUG] LLM error step {step}: {e}", file=sys.stderr, flush=True)
        return current_query or "SELECT 1"


class _Env:
    def __init__(self, base_url: str, task_id: str):
        import requests
        self._s = requests.Session()
        self._base = base_url.rstrip("/")
        self._task = task_id
        self.obs = {}

    def reset(self):
        r = self._s.post(f"{self._base}/reset", json={"task_id":self._task,"task":self._task}, timeout=30)
        r.raise_for_status()
        self.obs = r.json()
        return self.obs

    def step(self, sql: str):
        r = self._s.post(f"{self._base}/step",
                         json={"query":sql,"optimized_query":sql,"task_id":self._task,"task":self._task}, timeout=30)
        r.raise_for_status()
        d = r.json()
        self.obs = d.get("observation", d)
        return d

    def close(self): self._s.close()


def run_task(client: OpenAI, base_url: str, task_id: str) -> dict:
    env = _Env(base_url, task_id)
    obs = env.reset()
    current_query = obs.get("current_query", "")
    schema        = obs.get("schema_info","") or obs.get("schema_context","")
    feedback      = obs.get("feedback","") or obs.get("hint","")
    max_steps     = obs.get("max_steps", MAX_STEPS)
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    best_score  = 0.0
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    for step_num in range(1, max_steps + 1):
        sql = get_action(client, task_id, current_query, feedback, schema, step_num, history)
        try:
            result = env.step(sql)
        except Exception as e:
            log_step(step=step_num, action=sql, reward=0.0, done=False, error=str(e)[:80])
            break
        obs      = result.get("observation", result)
        reward   = float(result.get("reward", 0.0))
        done     = bool(result.get("done", False))
        info     = result.get("info", {})
        score    = float(info.get("score", reward))
        rewards.append(reward)
        steps_taken = step_num
        best_score  = max(best_score, score, reward)
        current_query = obs.get("current_query", sql)
        feedback      = obs.get("feedback","") or obs.get("hint","")
        err_msg       = obs.get("execution_error","") or None
        log_step(step=step_num, action=sql, reward=reward, done=done, error=err_msg)
        history.append(f"Step {step_num} | score={score:.2f} | reward={reward:+.2f} | {sql[:60]}")
        if done: break
    success = best_score >= 0.70
    log_end(success=success, steps=steps_taken, score=best_score, rewards=rewards)
    env.close()
    return {"task":task_id,"success":success,"score":best_score,"steps":steps_taken,"rewards":rewards}


_proc: Optional[subprocess.Popen] = None

def _start_server() -> str:
    global _proc
    import requests
    root = os.path.dirname(os.path.abspath(__file__))
    env  = os.environ.copy()
    env["PYTHONPATH"] = root
    _proc = subprocess.Popen(
        [sys.executable,"-m","uvicorn","server.app:app","--host","0.0.0.0","--port","7860","--workers","1"],
        cwd=root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(40):
        try:
            if requests.get("http://localhost:7860/health", timeout=2).ok:
                return "http://localhost:7860"
        except Exception: pass
        time.sleep(1)
    raise RuntimeError("Server did not start within 40s")

def _stop_server():
    global _proc
    if _proc: _proc.terminate(); _proc = None


def main():
    ap = argparse.ArgumentParser(description="SQLens inference script")
    ap.add_argument("--base-url", default=None, help="Environment base URL (default: start local)")
    ap.add_argument("--tasks", default=None, help="Comma-separated task IDs")
    args = ap.parse_args()
    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN not set.", file=sys.stderr, flush=True)
        sys.exit(1)
    local = args.base_url is None
    base_url = args.base_url
    if local:
        print("[INFO] Starting local server...", file=sys.stderr, flush=True)
        base_url = _start_server()
        print(f"[INFO] Server ready at {base_url}", file=sys.stderr, flush=True)
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN, max_retries=2)
        task_ids = [t.strip() for t in args.tasks.split(",")] if args.tasks else [
            "easy_select","medium_join","hard_subquery",
            "sql_injection_fix","query_optimization","complex_rewrite",
        ]
        print(f"[INFO] Model : {MODEL_NAME}", file=sys.stderr, flush=True)
        print(f"[INFO] API   : {API_BASE_URL}", file=sys.stderr, flush=True)
        print(f"[INFO] Tasks : {task_ids}", file=sys.stderr, flush=True)
        results = []
        t0 = time.time()
        for tid in task_ids:
            results.append(run_task(client, base_url, tid))
        elapsed = time.time() - t0
        passed  = sum(1 for r in results if r["success"])
        avg     = sum(r["score"] for r in results) / len(results) if results else 0
        print("="*60, file=sys.stderr, flush=True)
        print(f"FINAL  {passed}/{len(results)} passed | avg={avg:.3f} | {elapsed:.1f}s", file=sys.stderr, flush=True)
        for r in results:
            flag = "PASS" if r["success"] else "FAIL"
            print(f"  [{flag}] {r['task']:<26} score={r['score']:.3f}  steps={r['steps']}", file=sys.stderr, flush=True)
        print("="*60, file=sys.stderr, flush=True)
        sys.exit(0 if passed > 0 else 1)
    finally:
        if local: _stop_server()

if __name__ == "__main__":
    main()
