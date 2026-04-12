"""
validate.py — Pre-submission validation for SQLens OpenEnv
Run: python validate.py --base-url https://adit06-sqlens.hf.space
"""
from __future__ import annotations
import sys, os, yaml, argparse
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
results = []

def chk(name, ok, detail=""):
    results.append(ok)
    icon = "✅" if ok else "❌"
    print(f"  {icon}  {name}" + (f"  [{detail}]" if detail else ""))
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:7860")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    print(f"\n{'═'*62}")
    print(f"  SQLens — Pre-Submission Validation v2.0")
    print(f"  Target: {base}")
    print(f"{'═'*62}")

    print("\n[1] Server health & reset()")
    try:
        r = requests.get(f"{base}/health", timeout=20)
        chk("GET /health → 200", r.status_code==200, str(r.status_code))
        chk("Response contains status:ok", r.json().get("status")=="ok")
    except Exception as e:
        chk("GET /health reachable", False, str(e)[:60])
        chk("Response contains status:ok", False, "N/A")
    try:
        r = requests.get(f"{base}/healthz", timeout=20)
        chk("GET /healthz → 200", r.status_code==200, str(r.status_code))
    except Exception as e:
        chk("GET /healthz reachable", False, str(e)[:60])
    for tid in ["easy_select","sql_injection_fix"]:
        try:
            r = requests.post(f"{base}/reset", json={"task_id":tid}, timeout=20)
            d = r.json()
            chk(f"POST /reset ({tid}) → task_description", r.status_code==200 and "task_description" in d)
        except Exception as e:
            chk(f"POST /reset ({tid})", False, str(e)[:60])

    print("\n[2] OpenEnv spec compliance")
    yf = os.path.join(os.path.dirname(__file__), "openenv.yaml")
    if os.path.exists(yf):
        with open(yf) as f: m = yaml.safe_load(f)
        required = ["name","version","tasks","action","observation","state"]
        missing  = [k for k in required if k not in m]
        chk("openenv.yaml present & valid", not missing, f"missing:{missing}" if missing else "all keys present")
        chk("6 tasks in openenv.yaml", len(m.get("tasks",[]))>=6, f"{len(m.get('tasks',[]))} tasks found")
    else:
        chk("openenv.yaml present", False); chk("6+ tasks", False)
    from models import SQLAction, SQLObservation, SQLState, StepResult
    chk("Typed models importable", True)
    chk("SQLAction.query field exists", hasattr(SQLAction(),"query"))
    chk("SQLObservation.reward+done fields", hasattr(SQLObservation(),"reward"))
    chk("SQLState.episode_id field", hasattr(SQLState(),"episode_id"))
    try:
        r = requests.post(f"{base}/step", json={"query":"SELECT 1","task_id":"easy_select"}, timeout=20)
        d = r.json()
        chk("POST /step → observation+reward+done", "observation" in d and "reward" in d and "done" in d)
    except Exception as e:
        chk("POST /step → correct shape", False, str(e)[:60])
    try:
        r = requests.get(f"{base}/state?task_id=easy_select", timeout=15)
        d = r.json()
        chk("GET /state → episode_id+step_count", "episode_id" in d and "step_count" in d)
    except Exception as e:
        chk("GET /state → correct shape", False, str(e)[:60])

    print("\n[3] Dockerfile")
    df = os.path.join(os.path.dirname(__file__), "Dockerfile")
    if os.path.exists(df):
        c = open(df).read()
        chk("Dockerfile present", True)
        chk("FROM python:3.11", "FROM python:3.11" in c)
        chk("EXPOSE 7860", "EXPOSE 7860" in c)
        chk("HEALTHCHECK present", "HEALTHCHECK" in c)
        chk("CMD uses uvicorn", "uvicorn" in c)
    else:
        for _ in range(5): chk("Dockerfile", False, "missing")

    print("\n[4] inference.py")
    inf = os.path.join(os.path.dirname(__file__), "inference.py")
    chk("inference.py in project root", os.path.exists(inf))
    if os.path.exists(inf):
        src = open(inf).read()
        chk("Uses OpenAI client",  "from openai import OpenAI" in src)
        chk("Reads API_BASE_URL",  "API_BASE_URL" in src)
        chk("Reads MODEL_NAME",    "MODEL_NAME" in src)
        chk("Reads HF_TOKEN",      "HF_TOKEN" in src)
        chk("[START] log format",  "[START]" in src)
        chk("[STEP]  log format",  "[STEP]" in src)
        chk("[END]   log format",  "[END]" in src)

    print("\n[5] Graders — score range & canonical correctness")
    from tasks import TASK_REGISTRY, GRADER_REGISTRY, make_fresh_db, TUTOR_TASKS, OPTIMIZER_TASKS
    chk("6 tasks registered", len(TASK_REGISTRY)>=6, f"{len(TASK_REGISTRY)} found")
    chk("6 graders registered", len(GRADER_REGISTRY)>=6, f"{len(GRADER_REGISTRY)} found")
    for tid in TUTOR_TASKS:
        task = TASK_REGISTRY[tid]
        conn = make_fresh_db()
        score, _, _ = GRADER_REGISTRY[tid](task.expected_query, conn)
        chk(f"tutor '{tid}' canonical ≥ 0.9", score>=0.9, f"score={score:.4f}")
        chk(f"tutor '{tid}' score in [0,1]",  0.0<=score<=1.0)
    canonical = {
        "sql_injection_fix":  "SELECT id, username, role FROM users WHERE username = $1 AND password_hash = $2",
        "query_optimization": "SELECT id, customer_id, total_amount, status FROM orders WHERE created_at >= '2024-03-01' AND created_at < '2024-04-01' ORDER BY total_amount DESC LIMIT 100",
        "complex_rewrite":    "SELECT u.email, u.name, SUM(o.total_amount) AS total_spend, MAX(o.created_at) AS last_order_at FROM orders o JOIN users u ON u.id = o.customer_id WHERE RIGHT(u.email, 10) = '@gmail.com' GROUP BY u.id, u.email, u.name ORDER BY total_spend DESC LIMIT 50",
    }
    for tid in OPTIMIZER_TASKS:
        score, _, _ = GRADER_REGISTRY[tid](canonical[tid], None)
        chk(f"optimizer '{tid}' canonical ≥ 0.9", score>=0.9, f"score={score:.4f}")
        chk(f"optimizer '{tid}' score in [0,1]",  0.0<=score<=1.0)

    passed = sum(results); total = len(results)
    print(f"\n{'═'*62}")
    print(f"  Result: {passed}/{total} checks passed")
    if passed == total:
        print(f"  🎉 ALL CHECKS PASSED — ready to submit!")
        print(f"  Submit URL: {base}")
    else:
        print(f"  ⚠  {total-passed} check(s) failed — fix before submitting.")
    print(f"{'═'*62}\n")
    sys.exit(0 if passed==total else 1)

if __name__ == "__main__":
    main()
