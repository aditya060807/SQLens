"""
SQLens — Task Registry
6 tasks: 3 tutor (SQLite execution) + 3 optimizer (pattern grading)
"""
from __future__ import annotations
import sqlite3, re
from typing import Dict, List, Tuple, Any
from models import TaskDefinition

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,
    email         TEXT    UNIQUE NOT NULL,
    country       TEXT    NOT NULL,
    city          TEXT    NOT NULL,
    joined_date   TEXT    NOT NULL,
    tier          TEXT    NOT NULL DEFAULT 'standard'
);
CREATE TABLE IF NOT EXISTS products (
    product_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,
    category      TEXT    NOT NULL,
    subcategory   TEXT    NOT NULL,
    price         REAL    NOT NULL,
    cost          REAL    NOT NULL,
    stock         INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS orders (
    order_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id   INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date    TEXT    NOT NULL,
    shipped_date  TEXT,
    status        TEXT    NOT NULL,
    discount_pct  REAL    NOT NULL DEFAULT 0.0
);
CREATE TABLE IF NOT EXISTS order_items (
    item_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id      INTEGER NOT NULL REFERENCES orders(order_id),
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    quantity      INTEGER NOT NULL,
    unit_price    REAL    NOT NULL
);
CREATE TABLE IF NOT EXISTS reviews (
    review_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id   INTEGER NOT NULL REFERENCES customers(customer_id),
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    rating        INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
    review_date   TEXT    NOT NULL
);
"""

SEED_DATA_SQL = """
INSERT OR IGNORE INTO customers(customer_id,name,email,country,city,joined_date,tier) VALUES
(1,'Priya Sharma','priya@example.com','India','Mumbai','2021-03-12','premium'),
(2,'Aarav Patel','aarav@example.com','India','Pune','2022-07-08','standard'),
(3,'James Carter','james@example.com','USA','New York','2020-11-01','premium'),
(4,'Mei Lin','mei@example.com','China','Shanghai','2023-01-20','standard'),
(5,'Carlos Ruiz','carlos@example.com','India','Delhi','2021-09-15','standard'),
(6,'Sophie Dubois','sophie@example.com','France','Paris','2022-04-03','premium'),
(7,'Riya Nair','riya@example.com','India','Bangalore','2023-06-10','standard'),
(8,'Noah Williams','noah@example.com','USA','Chicago','2021-12-25','standard');

INSERT OR IGNORE INTO products(product_id,name,category,subcategory,price,cost,stock) VALUES
(1,'ThinkPad X1 Carbon','Electronics','Laptops',95000.00,60000.00,8),
(2,'Logitech MX Master 3','Electronics','Peripherals',8500.00,4000.00,45),
(3,'Clean Code (Book)','Books','Programming',699.00,200.00,120),
(4,'Standing Desk Pro','Furniture','Desks',32000.00,18000.00,4),
(5,'USB-C Dock 12-in-1','Electronics','Peripherals',4500.00,2000.00,30),
(6,'System Design Interview','Books','Engineering',850.00,250.00,90),
(7,'Ergonomic Mesh Chair','Furniture','Seating',22000.00,12000.00,6),
(8,'4K Webcam Pro','Electronics','Peripherals',6200.00,2800.00,20),
(9,'The Pragmatic Programmer','Books','Programming',749.00,220.00,80),
(10,'Monitor Arm Deluxe','Furniture','Accessories',3800.00,1500.00,15);

INSERT OR IGNORE INTO orders(order_id,customer_id,order_date,shipped_date,status,discount_pct) VALUES
(101,1,'2024-01-05','2024-01-07','delivered',0.0),(102,2,'2024-01-18',NULL,'pending',5.0),
(103,1,'2024-02-02','2024-02-05','delivered',10.0),(104,3,'2024-02-14','2024-02-16','delivered',0.0),
(105,5,'2024-03-01',NULL,'cancelled',0.0),(106,6,'2024-03-20','2024-03-23','delivered',0.0),
(107,7,'2024-04-10','2024-04-13','delivered',5.0),(108,1,'2024-04-25','2024-04-28','shipped',0.0),
(109,3,'2024-05-05','2024-05-07','delivered',15.0),(110,4,'2024-05-12',NULL,'pending',0.0),
(111,8,'2024-05-18','2024-05-21','delivered',0.0),(112,2,'2024-06-01','2024-06-03','delivered',0.0);

INSERT OR IGNORE INTO order_items(item_id,order_id,product_id,quantity,unit_price) VALUES
(1,101,1,1,95000.00),(2,101,2,1,8500.00),(3,102,3,2,699.00),
(4,103,5,1,4500.00),(5,103,6,1,850.00),(6,103,9,1,749.00),
(7,104,1,1,95000.00),(8,104,2,2,8500.00),(9,105,4,1,32000.00),
(10,106,7,1,22000.00),(11,107,8,1,6200.00),(12,107,10,1,3800.00),
(13,108,5,2,4500.00),(14,109,3,3,699.00),(15,109,6,2,850.00),
(16,110,2,1,8500.00),(17,111,4,1,32000.00),(18,111,7,1,22000.00),
(19,112,1,1,95000.00),(20,112,8,1,6200.00);

INSERT OR IGNORE INTO reviews(review_id,customer_id,product_id,rating,review_date) VALUES
(1,1,1,5,'2024-01-10'),(2,1,2,4,'2024-01-10'),(3,3,1,5,'2024-02-20'),
(4,6,7,4,'2024-03-25'),(5,7,8,3,'2024-04-15'),(6,7,10,5,'2024-04-15'),
(7,3,3,5,'2024-05-10'),(8,3,6,4,'2024-05-10'),(9,8,4,4,'2024-05-25'),
(10,8,7,5,'2024-05-25'),(11,2,1,4,'2024-06-05'),(12,2,8,5,'2024-06-05');
"""

SCHEMA_INFO = """
Tables (SQLite e-commerce):
customers   : customer_id(PK), name, email, country, city, joined_date, tier
products    : product_id(PK), name, category, subcategory, price, cost, stock
orders      : order_id(PK), customer_id(FK), order_date, shipped_date, status, discount_pct
order_items : item_id(PK), order_id(FK), product_id(FK), quantity, unit_price
reviews     : review_id(PK), customer_id(FK), product_id(FK), rating(1-5), review_date

orders.status  → 'pending'|'shipped'|'delivered'|'cancelled'
customers.tier → 'standard'|'premium'
Revenue = quantity × unit_price
"""


def make_fresh_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    conn.executescript(SEED_DATA_SQL)
    conn.commit()
    return conn


def run_query(conn: sqlite3.Connection, sql: str) -> Tuple[List[Dict], str]:
    try:
        cur = conn.execute(sql)
        return [dict(r) for r in cur.fetchall()], ""
    except Exception as exc:
        return [], str(exc)


def rows_to_table(rows: List[Dict], max_rows: int = 25) -> str:
    if not rows:
        return "(no rows returned)"
    cols = list(rows[0].keys())
    widths = [max(len(c), max((len(str(r.get(c, ""))) for r in rows[:max_rows]), default=0)) for c in cols]
    sep  = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    head = "|" + "|".join(f" {c.ljust(w)} " for c, w in zip(cols, widths)) + "|"
    lines = [sep, head, sep]
    for row in rows[:max_rows]:
        lines.append("|" + "|".join(f" {str(row.get(c,'')).ljust(w)} " for c, w in zip(cols, widths)) + "|")
    lines.append(sep)
    if len(rows) > max_rows:
        lines.append(f"  ... {len(rows)-max_rows} more rows")
    return "\n".join(lines)


def _row_overlap(got: List[Dict], expected: List[Dict]) -> float:
    if not expected: return 1.0 if not got else 0.0
    if not got: return 0.0
    def norm(r):
        items = []
        for k, v in r.items():
            key = k.lower().strip()
            try: val = str(round(float(v), 2))
            except: val = str(v).strip().lower()
            items.append((key, val))
        return frozenset(items)
    g = {norm(r) for r in got}
    e = {norm(r) for r in expected}
    inter = len(g & e)
    precision = inter / len(g) if g else 0.0
    recall    = inter / len(e) if e else 0.0
    if precision + recall == 0: return 0.0
    return 2 * (precision * recall) / (precision + recall)


def _schema_score(sql: str, required: List[str]) -> float:
    s = sql.lower()
    hits = sum(1 for t in required if t in s)
    return hits / len(required) if required else 1.0


def _efficiency_score(sql: str) -> float:
    return max(0.0, 1.0 - (0.25 if re.search(r'select\s+\*', sql, re.I) else 0.0))


# ── TUTOR TASK 1 — EASY ─────────────────────────────────────────

_EASY_EXPECTED = [
    {"name": "Carlos Ruiz",  "city": "Delhi",     "tier": "standard"},
    {"name": "Priya Sharma", "city": "Mumbai",    "tier": "premium"},
    {"name": "Riya Nair",    "city": "Bangalore", "tier": "standard"},
    {"name": "Aarav Patel",  "city": "Pune",      "tier": "standard"},
]

def grade_easy(sql: str, conn: sqlite3.Connection) -> Tuple[float, Dict, str]:
    rows, err = run_query(conn, sql)
    if err: return 0.0, {"correct_results":0.0,"schema_compliance":0.0,"query_efficiency":0.0}, f"Query error: {err}"
    norm = []
    for r in rows:
        nr = {}
        for k, v in r.items():
            lk = k.lower()
            if "name" in lk and "email" not in lk: nr["name"] = v
            elif "city" in lk: nr["city"] = v
            elif "tier" in lk: nr["tier"] = v
        if nr: norm.append(nr)
    row_sc = _row_overlap(norm, _EASY_EXPECTED)
    sch_sc = _schema_score(sql, ["customers"])
    eff_sc = _efficiency_score(sql)
    total  = 0.6*row_sc + 0.2*sch_sc + 0.2*eff_sc
    if row_sc < 0.3: hint = "❌ Filter WHERE country='India', SELECT name,city,tier, ORDER BY name ASC"
    elif row_sc < 0.7: hint = "⚠️ Partial — check India filter and name ordering"
    else: hint = "✅ Correct!"
    return round(total,4), {"correct_results":round(row_sc,3),"schema_compliance":round(sch_sc,3),"query_efficiency":round(eff_sc,3)}, hint

TASK_EASY = TaskDefinition(
    task_id="easy_select", difficulty="easy", title="Indian Customer Directory",
    description="Retrieve name, city, and tier of all customers from India, ordered alphabetically by name (A→Z).",
    business_context="Regional sales team needs a sorted directory of Indian customers for Q3 outreach.",
    schema_sql=SCHEMA_SQL, seed_data_sql=SEED_DATA_SQL,
    expected_query="SELECT name, city, tier FROM customers WHERE country = 'India' ORDER BY name ASC;",
    expected_rows=_EASY_EXPECTED,
    tags=["SELECT","WHERE","ORDER BY"], task_type="tutor",
)

# ── TUTOR TASK 2 — MEDIUM ────────────────────────────────────────

_MEDIUM_EXPECTED = [
    {"name":"James Carter",  "total_revenue":115797.0},
    {"name":"Priya Sharma",  "total_revenue":109599.0},
    {"name":"Aarav Patel",   "total_revenue":101200.0},
    {"name":"Noah Williams", "total_revenue":54000.0},
    {"name":"Sophie Dubois", "total_revenue":22000.0},
    {"name":"Riya Nair",     "total_revenue":10000.0},
]

def grade_medium(sql: str, conn: sqlite3.Connection) -> Tuple[float, Dict, str]:
    rows, err = run_query(conn, sql)
    if err: return 0.0, {"correct_results":0.0,"schema_compliance":0.0,"query_efficiency":0.0}, f"Query error: {err}"
    norm = []
    for r in rows:
        nr = {}
        for k, v in r.items():
            lk = k.lower()
            if "name" in lk and "total" not in lk: nr["name"] = v
            elif any(x in lk for x in ["revenue","total","spent","amount","sum"]):
                try: nr["total_revenue"] = round(float(v),1)
                except: nr["total_revenue"] = v
        if len(nr) >= 2: norm.append(nr)
    row_sc = _row_overlap(norm, _MEDIUM_EXPECTED)
    sch_sc = _schema_score(sql, ["customers","orders","order_items"])
    eff_sc = _efficiency_score(sql)
    total  = 0.6*row_sc + 0.2*sch_sc + 0.2*eff_sc
    if sch_sc < 0.5: hint = "❌ JOIN customers→orders→order_items"
    elif row_sc < 0.3: hint = "❌ Filter WHERE status='delivered', GROUP BY customer, SUM(quantity*unit_price)"
    else: hint = "✅ Correct!" if row_sc >= 0.9 else "⚠️ Check delivered filter and revenue calculation"
    return round(total,4), {"correct_results":round(row_sc,3),"schema_compliance":round(sch_sc,3),"query_efficiency":round(eff_sc,3)}, hint

TASK_MEDIUM = TaskDefinition(
    task_id="medium_join", difficulty="medium", title="Customer Revenue Report",
    description="For each customer with at least one delivered order, show name and total revenue (SUM of quantity×unit_price). Order by revenue descending.",
    business_context="Finance needs a revenue report to identify top spenders for the loyalty programme.",
    schema_sql=SCHEMA_SQL, seed_data_sql=SEED_DATA_SQL,
    expected_query="SELECT c.name, SUM(oi.quantity*oi.unit_price) AS total_revenue FROM customers c JOIN orders o ON c.customer_id=o.customer_id JOIN order_items oi ON o.order_id=oi.order_id WHERE o.status='delivered' GROUP BY c.customer_id,c.name ORDER BY total_revenue DESC;",
    expected_rows=_MEDIUM_EXPECTED,
    tags=["JOIN","GROUP BY","SUM"], task_type="tutor",
)

# ── TUTOR TASK 3 — HARD ──────────────────────────────────────────

_HARD_EXPECTED = [
    {"category":"Books",       "top_product":"Clean Code (Book)",    "revenue":3495.0,  "avg_rating":5.0},
    {"category":"Electronics", "top_product":"ThinkPad X1 Carbon",  "revenue":855000.0,"avg_rating":4.67},
    {"category":"Furniture",   "top_product":"Ergonomic Mesh Chair", "revenue":88000.0, "avg_rating":4.5},
]

def grade_hard(sql: str, conn: sqlite3.Connection) -> Tuple[float, Dict, str]:
    rows, err = run_query(conn, sql)
    if err: return 0.0, {"correct_results":0.0,"schema_compliance":0.0,"query_efficiency":0.0}, f"Query error: {err}"
    norm = []
    for r in rows:
        nr = {}
        for k, v in r.items():
            lk = k.lower().strip()
            if lk == "category": nr["category"] = v
            elif lk in ("top_product","product_name","name","product"): nr["top_product"] = v
            elif any(x in lk for x in ["revenue","total","amount"]):
                try: nr["revenue"] = round(float(v),1)
                except: nr["revenue"] = v
            elif "rating" in lk:
                try: nr["avg_rating"] = round(float(v),2)
                except: nr["avg_rating"] = v
        if len(nr) >= 2: norm.append(nr)
    row_sc = _row_overlap(norm, _HARD_EXPECTED)
    sch_sc = _schema_score(sql, ["products","order_items","reviews"])
    eff_sc = _efficiency_score(sql)
    total  = 0.6*row_sc + 0.2*sch_sc + 0.2*eff_sc
    if sch_sc < 0.5: hint = "❌ JOIN products+order_items+reviews"
    elif row_sc < 0.3: hint = "❌ Use RANK() OVER (PARTITION BY category ORDER BY revenue DESC), filter WHERE rnk=1"
    else: hint = "✅ Correct!" if row_sc >= 0.9 else "⚠️ Check window function and category partitioning"
    return round(total,4), {"correct_results":round(row_sc,3),"schema_compliance":round(sch_sc,3),"query_efficiency":round(eff_sc,3)}, hint

TASK_HARD = TaskDefinition(
    task_id="hard_subquery", difficulty="hard", title="Category Champion Report",
    description="For each product category, find the top-selling product by revenue. Show: category, product name, total revenue, avg rating. Use CTE + RANK().",
    business_context="Merchandising team wants the category champion in each segment for the homepage.",
    schema_sql=SCHEMA_SQL, seed_data_sql=SEED_DATA_SQL,
    expected_query="WITH ps AS (SELECT p.category,p.name AS product_name,SUM(oi.quantity*oi.unit_price) AS revenue,ROUND(AVG(r.rating),2) AS avg_rating,RANK() OVER(PARTITION BY p.category ORDER BY SUM(oi.quantity*oi.unit_price) DESC) AS rnk FROM products p JOIN order_items oi ON p.product_id=oi.product_id LEFT JOIN reviews r ON p.product_id=r.product_id GROUP BY p.product_id,p.category,p.name) SELECT category,product_name AS top_product,revenue,avg_rating FROM ps WHERE rnk=1 ORDER BY category;",
    expected_rows=_HARD_EXPECTED,
    tags=["CTE","Window Functions","RANK","LEFT JOIN"], task_type="tutor",
)

# ── OPTIMIZER TASKS ──────────────────────────────────────────────

def _has_parameterized(q): return bool(re.search(r"\$\d+|\?|:\w+|%\(\w+\)s", q))
def _has_concat_injection(q): return bool(re.search(r"'\s*\|\||\|\|\s*'", q) or re.search(r"or\s+1\s*=\s*1", q, re.I))
def _logic_preserved(q): lower=q.lower(); return "from users" in lower and ("username" in lower or "email" in lower) and "password" in lower
def _no_date_fn(q): return not bool(re.search(r"year\s*\(|month\s*\(|date_format\s*\(", q, re.I))
def _has_range(q): return bool(re.search(r"between|>=.*<|created_at\s*>=", q, re.I))
def _no_star(q): return not bool(re.search(r"select\s+\*", q, re.I))
def _has_limit(q): return bool(re.search(r"\blimit\s+\d+", q, re.I))
def _has_join(q): return bool(re.search(r"\bjoin\b", q, re.I))
def _no_correlated(q):
    lower = q.lower()
    if lower.count("select") <= 1: return True
    return not bool(re.search(r"select.*from.*where.*\bo\.", q, re.I|re.DOTALL))
def _no_leading_wildcard(q): return not bool(re.search(r"like\s+['\"]%", q, re.I))
def _has_group_by(q): return bool(re.search(r"\bgroup\s+by\b", q, re.I))
def _has_order_by(q): return bool(re.search(r"\border\s+by\b", q, re.I))


def grade_injection_fix(sql: str, conn=None) -> Tuple[float, Dict, str]:
    score, issues = 0.0, []
    if _has_parameterized(sql): score += 0.50
    else: issues.append("Missing $1/$2 parameterized placeholders")
    if not _has_concat_injection(sql): score += 0.25
    else: issues.append("String concatenation still present")
    if _logic_preserved(sql): score += 0.25
    else: issues.append("Query logic not preserved (need users/username/password)")
    hint = " | ".join(issues) if issues else "✅ All injection issues resolved!"
    return round(min(score,1.0),4), {"parameterized":score>=0.5,"no_concat":score>=0.75,"logic_ok":score>=1.0}, hint


def grade_query_optimization(sql: str, conn=None) -> Tuple[float, Dict, str]:
    score, issues = 0.0, []
    if _no_date_fn(sql): score += 0.35
    else: issues.append("Remove YEAR()/MONTH() — use date range")
    if _has_range(sql): score += 0.20
    else: issues.append("Add range: created_at >= '2024-03-01' AND created_at < '2024-04-01'")
    if _no_star(sql): score += 0.25
    else: issues.append("Replace SELECT * with specific columns")
    if _has_limit(sql): score += 0.20
    else: issues.append("Add LIMIT 100")
    hint = " | ".join(issues) if issues else "✅ All performance issues resolved!"
    return round(min(score,1.0),4), {"no_date_fn":_no_date_fn(sql),"has_range":_has_range(sql),"no_star":_no_star(sql),"has_limit":_has_limit(sql)}, hint


def grade_complex_rewrite(sql: str, conn=None) -> Tuple[float, Dict, str]:
    score, issues = 0.0, []
    if _has_join(sql): score += 0.30
    else: issues.append("Add JOIN users u ON u.id = o.customer_id")
    if _no_correlated(sql): score += 0.20
    else: issues.append("Replace correlated subqueries with JOIN")
    if _no_leading_wildcard(sql): score += 0.20
    else: issues.append("Use RIGHT(u.email,10)='@gmail.com' instead of LIKE '%@gmail.com'")
    if _has_limit(sql): score += 0.15
    else: issues.append("Add LIMIT 50")
    if _has_group_by(sql) and _has_order_by(sql): score += 0.15
    else:
        if not _has_group_by(sql): issues.append("Add GROUP BY u.id,u.email,u.name")
        if not _has_order_by(sql): issues.append("Add ORDER BY total_spend DESC")
    hint = " | ".join(issues) if issues else "✅ All rewrite objectives met!"
    return round(min(score,1.0),4), {"has_join":_has_join(sql),"no_correlated":_no_correlated(sql),"no_wildcard":_no_leading_wildcard(sql),"has_limit":_has_limit(sql)}, hint


TASK_INJECTION = TaskDefinition(
    task_id="sql_injection_fix", difficulty="easy", title="SQL Injection Fix",
    description="A production login query concatenates user input directly into SQL. Rewrite it using $1, $2 parameterized placeholders.",
    business_context="Security audit found a critical SQL injection in the login endpoint.",
    original_query="SELECT id, username, role\nFROM users\nWHERE username = '' || user_input || ''\n  AND password_hash = '' || pass_input || ''",
    dialect="postgres",
    schema_context="Table: users(id SERIAL PK, username VARCHAR(64) UNIQUE, password_hash VARCHAR(128), role VARCHAR(32), active BOOLEAN). ~50K rows.",
    objective="Use $1, $2 parameterized placeholders. Remove all string concatenation.",
    max_steps=5, success_threshold=0.80, task_type="optimizer",
    tags=["Security","SQL Injection","Parameterized Queries"],
)

TASK_OPTIMIZATION = TaskDefinition(
    task_id="query_optimization", difficulty="medium", title="Query Performance Optimization",
    description="A reporting query causes a full table scan on 10M+ rows. Fix YEAR()/MONTH() on indexed column, replace SELECT *, add LIMIT 100.",
    business_context="The orders report is timing out in production.",
    original_query="SELECT *\nFROM orders\nWHERE YEAR(created_at) = 2024\n  AND MONTH(created_at) = 3\nORDER BY total_amount DESC",
    dialect="mysql",
    schema_context="Table: orders(id BIGINT PK, customer_id INT FK, created_at DATETIME, status VARCHAR(32), total_amount DECIMAL(12,2)). ~10M rows. Index on created_at.",
    objective="1) Replace YEAR()/MONTH() with range. 2) Replace SELECT *. 3) Add LIMIT 100.",
    max_steps=6, success_threshold=0.75, task_type="optimizer",
    tags=["Performance","Index","Query Optimization"],
)

TASK_REWRITE = TaskDefinition(
    task_id="complex_rewrite", difficulty="hard", title="Complex Query Rewrite",
    description="An analytics query uses correlated subqueries (N+1), a leading-wildcard LIKE, and no pagination. Rewrite using JOINs, fix LIKE, add LIMIT 50.",
    business_context="Analytics dashboard fires 50M+ subquery executions per report.",
    original_query="SELECT\n  (SELECT email FROM users WHERE id = o.customer_id),\n  (SELECT name  FROM users WHERE id = o.customer_id),\n  SUM(o.total_amount),\n  (SELECT MAX(created_at) FROM orders WHERE customer_id = o.customer_id)\nFROM orders o\nWHERE (SELECT email FROM users WHERE id = o.customer_id) LIKE '%@gmail.com'\nGROUP BY o.customer_id\nORDER BY SUM(o.total_amount) DESC",
    dialect="postgres",
    schema_context="Tables: users(id SERIAL PK, email VARCHAR(128), name VARCHAR(128)). orders(id BIGINT PK, customer_id INT FK, created_at TIMESTAMPTZ, total_amount DECIMAL(12,2)). Return top 50.",
    objective="1) Replace correlated subqueries with JOIN. 2) Fix leading LIKE. 3) Add LIMIT 50. 4) GROUP BY u.id,u.email,u.name.",
    max_steps=8, success_threshold=0.70, task_type="optimizer",
    tags=["JOIN","Correlated Subquery","Performance","Pagination"],
)

TASK_REGISTRY: Dict[str, TaskDefinition] = {
    "easy_select":        TASK_EASY,
    "medium_join":        TASK_MEDIUM,
    "hard_subquery":      TASK_HARD,
    "sql_injection_fix":  TASK_INJECTION,
    "query_optimization": TASK_OPTIMIZATION,
    "complex_rewrite":    TASK_REWRITE,
}

GRADER_REGISTRY = {
    "easy_select":        grade_easy,
    "medium_join":        grade_medium,
    "hard_subquery":      grade_hard,
    "sql_injection_fix":  grade_injection_fix,
    "query_optimization": grade_query_optimization,
    "complex_rewrite":    grade_complex_rewrite,
}

TUTOR_TASKS     = ["easy_select","medium_join","hard_subquery"]
OPTIMIZER_TASKS = ["sql_injection_fix","query_optimization","complex_rewrite"]
ALL_TASK_IDS    = TUTOR_TASKS + OPTIMIZER_TASKS
