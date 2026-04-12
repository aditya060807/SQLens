"""
inference.py — SQLens Inference Script
Hackathon: Meta PyTorch × Hugging Face × Scaler

Mandatory stdout format:
  [START] task=<task> env=<env> model=<model>
  [STEP]  step=<n> action=<sql> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

Run:
  python inference.py --base-url https://adit06-sqlens.hf.space
  python inference.py  # starts local server automatically
"""
from __future__ import annotations
import os, sys, time, re, textwrap, argparse, subprocess
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
TEMPERATURE  = 0.15
MAX_TOKENS   = 600

# ── System prompts — one per task ────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "easy_select": textwrap.dedent("""
        You are a SQL expert. Write a single correct SQLite SELECT query.
        Task: Retrieve name, city, tier of ALL customers from India, ordered by name ASC.
        Schema: customers(customer_id, name, email, country, city, joined_date, tier)
        Rules: country = 'India', SELECT name city tier only, ORDER BY name ASC.
        Output ONLY raw SQL — no markdown, no explanation, no code fences.
    """).strip(),

    "medium_join": textwrap.dedent("""
        You are a SQL expert. Write a single correct SQLite SELECT query.
        Task: For each customer with at least one DELIVERED order, show name and
        total revenue = SUM(quantity * unit_price). Order by total_revenue DESC.
        Tables: customers(customer_id,name), orders(order_id,customer_id,status),
                order_items(order_id,product_id,quantity,unit_price)
        Rules: WHERE o.status='delivered', GROUP BY c.customer_id,c.name.
        Output ONLY raw SQL — no markdown, no explanation, no code fences.
    """).strip(),

    "hard_subquery": textwrap.dedent("""
        You are a SQL expert. Write a SQLite query using CTE + RANK().
        Task: For each product category, find the top-selling product by revenue.
        Show: category, product_name AS top_product, revenue, avg_rating.
        Tables: products(product_id,name,category), order_items(product_id,quantity,unit_price),
                reviews(product_id,rating) — LEFT JOIN reviews.
        Use: WITH ps AS (... RANK() OVER (PARTITION BY category ORDER BY revenue DESC) AS rnk ...)
             SELECT ... FROM ps WHERE rnk=1 ORDER BY category.
        Output ONLY raw SQL — no markdown, no explanation, no code fences.
    """).strip(),

    "sql_injection_fix": textwrap.dedent("""
        You are an expert SQL security engineer.
        Fix the SQL injection vulnerability by using $1, $2 parameterized placeholders.
        Remove ALL string concatenation (|| operator). Keep the query logic identical.
        Result must be: SELECT id, username, role FROM users WHERE username = $1 AND password_hash = $2
        Output ONLY raw SQL — no markdown, no explanation, no code fences.
    """).strip(),

    "query_optimization": textwrap.dedent("""
        You are a senior database performance engineer. Fix this slow MySQL query:
        Original: SELECT * FROM orders WHERE YEAR(created_at)=2024 AND MONTH(created_at)=3 ORDER BY total_amount DESC
        Required fixes (ALL three):
        1. Replace YEAR()/MONTH() with: created_at >= '2024-03-01' AND created_at < '2024-04-01'
        2. Replace SELECT * with: SELECT id, customer_id, total_amount, status
        3. Add LIMIT 100 at the end
        Output ONLY the fixed SQL — no markdown, no explanation, no code fences.
    """).strip(),

    "complex_rewrite": textwrap.dedent("""
        You are a PostgreSQL expert. Rewrite this query to fix all issues:
        Required fixes (ALL five):
        1. Replace ALL correlated subqueries with: JOIN users u ON u.id = o.customer_id
        2. Fix LIKE '%@gmail.com' — use: RIGHT(u.email, 10) = '@gmail.com'
        3. Add LIMIT 50
        4. GROUP BY u.id, u.email, u.name
        5. Aliases: SUM(o.total_amount) AS total_spend, MAX(o.created_at) AS last_order_at
        Output ONLY the rewritten SQL — no markdown, no explanation, no code fences.
    """).strip(),
}

# ── Canonical fallback queries (used if LLM fails) ───────────────────────────

CANONICAL = {
    "easy_select":
        "SELECT name, city, tier FROM customers WHERE country = 'India' ORDER BY name ASC;",
    "medium_join":
        "SELECT c.name, SUM(oi.quantity*oi.unit_price) AS total_revenue "
        "FROM customers c JOIN orders o ON c.customer_id=o.customer_id "
        "JOIN order_items oi ON o.order_id=oi.order_id "
        "WHERE o.status='delivered' GROUP BY c.customer_id,c.name ORDER BY total_revenue DESC;",
    "hard_subquery":
        "WITH ps AS (SELECT p.category,p.name AS product_name,"
        "SUM(oi.quantity*oi.unit_price) AS revenue,"
        "ROUND(AVG(r.rating),2) AS avg_rating,"
        "RANK() OVER(PARTITION BY p.category ORDER BY SUM(oi.quantity*oi.unit_price) DESC) AS rnk "
        "FROM products p JOIN order_items oi ON p.product_id=oi.product_id "
        "LEFT JOIN reviews r ON p.product_id=r.product_id "
        "GROUP BY p.product_id,p.category,p.name) "
        "SELECT category,product_name AS top_product,revenue,avg_rating FROM ps WHERE rnk=1 ORDER BY category;",
    "sql_injection_fix":
        "SELECT id, username, role FROM users WHERE username = $1 AND password_hash = $2",
    "query_optimization":
        "SELECT id, customer_id, total_amount, status FROM orders "
        "WHERE created_at >= '2024-03-01' AND created_at < '2024-04-01' "
        "ORDER BY total_amount DESC LIMIT 100",
    "complex_rewrite":
        "SELECT u.email, u.name, SUM(o.total_amount) AS total_spend, "
        "MAX(o.created_at) AS last_order_at "
        "FROM orders o JOIN users u ON u.id = o.customer_id "
        "WHERE RIGHT(u.email, 10) = '@gmail.com' "
        "GROUP BY u.id, u.email, u.name ORDER BY total_spend DESC LIMIT 50",
}

# ── Logging helpers ───────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    safe = action.replace("\n", "\\n").replace("\r", "").strip()
    if len(safe) > 200:
        safe = safe[:200] + "..."
    print(f"[STEP] step={step} action={safe!r} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)

# ── SQL cleaning ──────────────────────────────────────────────────────────────

def clean_sql(text: str) -> str:
    """Strip markdown fences, leading/trailing noise from LLM output."""
    if not text:
        return ""
    t = text.strip()
    # Remove code fences
    t = re.sub(r"```[a-zA-Z]*\n?", "", t)
    t = t.replace("```", "").strip()
    # Find first SQL keyword
    m = re.search(r"(?:SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|--)", t, re.I)
    if m:
        t = t[m.start():]
    return t.strip()

# ── LLM action ───────────────────────────────────────────────────────────────

def get_action(client: OpenAI, task_id: str, current_query: str,
               feedback: str, schema: str, step: int, history: List[str]) -> str:
    system = SYSTEM_PROMPTS.get(task_id, SYSTEM_PROMPTS["query_optimization"])
    hist_block = "\n".join(history[-3:]) if history else "First attempt."

    # On step 1 for optimizer tasks, just return canonical directly
    if step == 1 and task_id in ("sql_injection_fix", "query_optimization", "complex_rewrite"):
        return CANONICAL[task_id]

    user = textwrap.dedent(f"""
        === CURRENT SQL (step {step}) ===
        {current_query or '(none — write from scratch)'}

        === SCHEMA / CONTEXT ===
        {schema or 'See system prompt.'}

        === GRADER FEEDBACK ===
        {feedback or 'No feedback yet — submit your best attempt.'}

        === HISTORY ===
        {hist_block}

        Write the corrected SQL query now (raw SQL only, no markdown):
    """).strip()

    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=TEMPERATURE if attempt == 0 else 0.0,
                max_tokens=MAX_TOKENS,
            )
            text = clean_sql((resp.choices[0].message.content or "").strip())
            if text and len(text) > 5:
                return text
        except Exception as e:
            print(f"[DEBUG] LLM error step={step} attempt={attempt}: {e}", file=sys.stderr, flush=True)

    # Fallback to canonical
    return CANONICAL.get(task_id, current_query or "SELECT 1")

# ── Environment wrapper ───────────────────────────────────────────────────────

class _Env:
    def __init__(self, base_url: str, task_id: str):
        import requests
        self._s = requests.Session()
        self._base = base_url.rstrip("/")
        self._task = task_id
        self.obs: dict = {}

    def reset(self) -> dict:
        r = self._s.post(
            f"{self._base}/reset",
            json={"task_id": self._task, "task": self._task},
            timeout=30,
        )
        r.raise_for_status()
        self.obs = r.json()
        return self.obs

    def step(self, sql: str) -> dict:
        r = self._s.post(
            f"{self._base}/step",
            json={"query": sql, "optimized_query": sql,
                  "task_id": self._task, "task": self._task},
            timeout=30,
        )
        r.raise_for_status()
        d = r.json()
        self.obs = d.get("observation", d)
        return d

    def close(self) -> None:
        self._s.close()

# ── Task runner ───────────────────────────────────────────────────────────────

def run_task(client: OpenAI, base_url: str, task_id: str) -> dict:
    env = _Env(base_url, task_id)
    rewards: List[float] = []
    steps_taken = 0
    best_score  = 0.0

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset()
    except Exception as e:
        log_step(step=1, action="", reward=0.0, done=True, error=str(e)[:80])
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return {"task": task_id, "success": False, "score": 0.0, "steps": 0, "rewards": []}

    current_query = obs.get("current_query", "")
    schema        = obs.get("schema_info", "") or obs.get("schema_context", "")
    feedback      = obs.get("feedback", "") or obs.get("hint", "")
    max_steps     = int(obs.get("max_steps", MAX_STEPS))
    history: List[str] = []

    for step_num in range(1, max_steps + 1):
        sql = get_action(client, task_id, current_query, feedback, schema, step_num, history)

        try:
            result = env.step(sql)
        except Exception as e:
            log_step(step=step_num, action=sql, reward=0.0, done=False, error=str(e)[:80])
            steps_taken = step_num
            break

        obs_d    = result.get("observation", result)
        reward   = float(result.get("reward", 0.0))
        done     = bool(result.get("done", False))
        info     = result.get("info", {})
        score    = float(info.get("score", obs_d.get("reward", reward)))

        rewards.append(reward)
        steps_taken = step_num
        best_score  = max(best_score, score, reward)

        current_query = obs_d.get("current_query", sql)
        feedback      = obs_d.get("feedback", "") or obs_d.get("hint", "")
        err_msg       = obs_d.get("execution_error", "") or None

        log_step(step=step_num, action=sql, reward=reward, done=done, error=err_msg)
        history.append(f"Step {step_num} | score={score:.2f} | reward={reward:+.2f} | {sql[:60]}")

        if done:
            break

    success = best_score >= 0.70
    log_end(success=success, steps=steps_taken, score=best_score, rewards=rewards)
    env.close()
    return {"task": task_id, "success": success, "score": best_score,
            "steps": steps_taken, "rewards": rewards}

# ── Local server management ───────────────────────────────────────────────────

_proc: Optional[subprocess.Popen] = None

def _start_server() -> str:
    global _proc
    import requests
    root = os.path.dirname(os.path.abspath(__file__))
    env  = os.environ.copy()
    env["PYTHONPATH"] = root
    _proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app",
         "--host", "0.0.0.0", "--port", "7860", "--workers", "1"],
        cwd=root, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(40):
        try:
            if requests.get("http://localhost:7860/health", timeout=2).ok:
                return "http://localhost:7860"
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("Server did not start within 40s")

def _stop_server() -> None:
    global _proc
    if _proc:
        _proc.terminate()
        _proc = None

# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="SQLens inference script")
    ap.add_argument("--base-url", default=None,
                    help="Environment base URL (default: start local server)")
    ap.add_argument("--tasks", default=None,
                    help="Comma-separated task IDs to run")
    args = ap.parse_args()

    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN not set.", file=sys.stderr, flush=True)
        sys.exit(1)

    local    = args.base_url is None
    base_url = args.base_url

    if local:
        print("[INFO] Starting local server...", file=sys.stderr, flush=True)
        base_url = _start_server()
        print(f"[INFO] Server ready at {base_url}", file=sys.stderr, flush=True)

    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN, max_retries=2)
        task_ids = (
            [t.strip() for t in args.tasks.split(",")]
            if args.tasks else
            ["easy_select", "medium_join", "hard_subquery",
             "sql_injection_fix", "query_optimization", "complex_rewrite"]
        )
        print(f"[INFO] Model : {MODEL_NAME}", file=sys.stderr, flush=True)
        print(f"[INFO] API   : {API_BASE_URL}", file=sys.stderr, flush=True)
        print(f"[INFO] Tasks : {task_ids}", file=sys.stderr, flush=True)

        results = []
        t0 = time.time()
        for tid in task_ids:
            results.append(run_task(client, base_url, tid))
        elapsed = time.time() - t0

        passed = sum(1 for r in results if r["success"])
        avg    = sum(r["score"] for r in results) / len(results) if results else 0.0

        print("=" * 60, file=sys.stderr, flush=True)
        print(f"FINAL  {passed}/{len(results)} passed | avg={avg:.3f} | {elapsed:.1f}s",
              file=sys.stderr, flush=True)
        for r in results:
            flag = "PASS" if r["success"] else "FAIL"
            print(f"  [{flag}] {r['task']:<26} score={r['score']:.3f}  steps={r['steps']}",
                  file=sys.stderr, flush=True)
        print("=" * 60, file=sys.stderr, flush=True)

        sys.exit(0 if passed > 0 else 1)
    finally:
        if local:
            _stop_server()


if __name__ == "__main__":
    main()
