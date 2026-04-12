"""
SQLens — OpenEnv Environment
reset() / step() / state() — handles tutor (SQLite) and optimizer (pattern grading) tasks.
"""
from __future__ import annotations
import sys, os, uuid, sqlite3
from datetime import datetime, timezone
from typing import Optional, List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import SQLAction, SQLObservation, SQLState, StepResult, RewardBreakdown
from tasks import (
    TASK_REGISTRY, GRADER_REGISTRY, SCHEMA_INFO,
    make_fresh_db, run_query, rows_to_table,
    TUTOR_TASKS, OPTIMIZER_TASKS,
)

_FORBIDDEN = ("drop","insert","update","delete","alter","create","truncate","pragma","attach","detach")
MAX_QUERY_LEN = 4000
STEP_PENALTY       = -0.02
REPETITION_PENALTY = -0.10
IMPROVEMENT_BONUS  =  0.10
OBJECTIVE_BONUS    =  0.20


class SQLQueryEnvironment:
    def __init__(self):
        self._state: SQLState = SQLState()
        self._conn: Optional[sqlite3.Connection] = None
        self._task_def = None
        self._last_score: float = 0.0
        self._last_query: str = ""
        self._cumulative_reward: float = 0.0
        self._episode_rewards: List[float] = []
        self._best_score: float = 0.0

    def reset(self, task_id: Optional[str] = None, seed: int = 42) -> SQLObservation:
        if task_id not in TASK_REGISTRY:
            task_id = "easy_select"
        self._close_db()
        self._task_def = TASK_REGISTRY[task_id]
        self._last_score = 0.0
        self._last_query = ""
        self._cumulative_reward = 0.0
        self._episode_rewards = []
        self._best_score = 0.0

        if task_id in TUTOR_TASKS:
            self._conn = make_fresh_db()

        self._state = SQLState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            current_task=task_id,
            task_difficulty=self._task_def.difficulty,
            max_steps=self._task_def.max_steps,
            attempts=0,
            best_reward=0.0,
            task_solved=False,
            schema_snapshot=SCHEMA_INFO if task_id in TUTOR_TASKS else self._task_def.schema_context,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        hint = f"Business context: {self._task_def.business_context}" if self._task_def.business_context else ""
        schema_info = SCHEMA_INFO if task_id in TUTOR_TASKS else self._task_def.schema_context
        current_q = self._task_def.original_query if task_id in OPTIMIZER_TASKS else ""

        return SQLObservation(
            task_description=self._task_def.description,
            schema_info=schema_info,
            query_result="",
            execution_error="",
            reward=0.0,
            done=False,
            partial_scores={},
            hint=hint,
            step_number=0,
            max_steps=self._state.max_steps,
            task_name=task_id,
            current_query=current_q,
            dialect=self._task_def.dialect,
            last_reward=0.0,
            cumulative_reward=0.0,
            feedback="Episode started." if task_id in OPTIMIZER_TASKS else "",
            schema_context=self._task_def.schema_context,
        )

    def step(self, action: SQLAction) -> StepResult:
        if self._task_def is None:
            self.reset()
        self._state.step_count += 1
        self._state.attempts += 1
        task_id = self._state.current_task
        sql = (action.get_query() or "").strip()
        if len(sql) > MAX_QUERY_LEN:
            return self._reject(f"Query too long ({len(sql)} chars). Max {MAX_QUERY_LEN}.", "Shorten your query.")
        if task_id in TUTOR_TASKS:
            return self._step_tutor(sql, task_id)
        return self._step_optimizer(sql, task_id)

    def _step_tutor(self, sql: str, task_id: str) -> StepResult:
        sql_lower = sql.lower().lstrip()
        for prefix in _FORBIDDEN:
            if sql_lower.startswith(prefix):
                return self._reject(f"Only SELECT statements permitted. Got: {sql[:60]}", "Write a SELECT query.")
        if not sql_lower.startswith("select") and not sql_lower.startswith("with"):
            return self._reject("Query must begin with SELECT or WITH.", "Start with SELECT or WITH.")
        try:
            reward, partial, hint = GRADER_REGISTRY[task_id](sql, self._conn)
        except Exception as exc:
            return self._reject(f"Grader error: {exc}", "")
        if reward > self._state.best_reward:
            self._state.best_reward = reward
        solved    = reward >= 0.95
        exhausted = self._state.step_count >= self._state.max_steps
        done      = solved or exhausted
        if solved: self._state.task_solved = True
        rows, exec_err = run_query(self._conn, sql)
        obs = SQLObservation(
            task_description=self._task_def.description, schema_info=SCHEMA_INFO,
            query_result=rows_to_table(rows) if rows else "", execution_error=exec_err,
            reward=reward, done=done, partial_scores=partial, hint=hint,
            step_number=self._state.step_count, max_steps=self._state.max_steps,
            task_name=task_id, current_query=sql, dialect="sqlite",
        )
        return StepResult(observation=obs, reward=reward, done=done,
                          info={"step":self._state.step_count,"best_reward":self._state.best_reward,
                                "task_id":task_id,"solved":solved,"episode_id":self._state.episode_id})

    def _step_optimizer(self, sql: str, task_id: str) -> StepResult:
        try:
            score, partial, feedback = GRADER_REGISTRY[task_id](sql, None)
        except Exception as exc:
            return self._reject(f"Grader error: {exc}", "")
        reward = self._shape_reward(sql, score)
        self._last_query = sql
        self._last_score = score
        self._best_score = max(self._best_score, score)
        self._cumulative_reward += reward
        self._episode_rewards.append(reward)
        if score > self._state.best_reward:
            self._state.best_reward = score
        objective = score >= self._task_def.success_threshold
        exhausted = self._state.step_count >= self._state.max_steps
        done = objective or exhausted
        if objective: self._state.task_solved = True
        obs = SQLObservation(
            task_description=self._task_def.description, schema_info=self._task_def.schema_context,
            query_result="", execution_error="", reward=reward, done=done,
            partial_scores={k:float(v) for k,v in partial.items()}, hint=feedback,
            step_number=self._state.step_count, max_steps=self._state.max_steps,
            task_name=task_id, current_query=sql, dialect=self._task_def.dialect,
            last_reward=round(reward,4), cumulative_reward=round(self._cumulative_reward,4),
            feedback=feedback, schema_context=self._task_def.schema_context,
        )
        return StepResult(observation=obs, reward=round(reward,4), done=done,
                          info={"score":score,"best_score":self._best_score,
                                "objective_achieved":objective,"task_id":task_id,
                                "episode_id":self._state.episode_id})

    def _shape_reward(self, proposed: str, score: float) -> float:
        delta = max(0.0, score - self._last_score)
        correctness_bonus = OBJECTIVE_BONUS if (score >= self._task_def.success_threshold and self._last_score < self._task_def.success_threshold) else 0.0
        improvement_bonus = IMPROVEMENT_BONUS if delta > 0.05 else 0.0
        repetition_penalty = REPETITION_PENALTY if proposed == self._last_query else 0.0
        raw = delta + correctness_bonus + improvement_bonus + repetition_penalty + STEP_PENALTY
        return round(max(-0.5, min(1.0, raw)), 4)

    def state(self) -> SQLState:
        return self._state

    def _reject(self, error: str, hint: str) -> StepResult:
        obs = SQLObservation(
            task_description=self._task_def.description if self._task_def else "",
            schema_info=SCHEMA_INFO, query_result="", execution_error=error,
            reward=0.0, done=False, partial_scores={}, hint=hint,
            step_number=self._state.step_count, max_steps=self._state.max_steps,
        )
        return StepResult(observation=obs, reward=0.0, done=False)

    def _close_db(self):
        if self._conn:
            try: self._conn.close()
            except: pass
            self._conn = None

    def close(self):
        self._close_db()
