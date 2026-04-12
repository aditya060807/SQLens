"""
SQLens — Typed Models (OpenEnv spec)
Action / Observation / State / StepResult
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TaskDefinition(BaseModel):
    task_id: str
    difficulty: str
    title: str
    description: str
    business_context: str = ""
    schema_sql: str = ""
    seed_data_sql: str = ""
    expected_query: str = ""
    expected_rows: List[Dict] = Field(default_factory=list)
    grader_hints: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    original_query: str = ""
    dialect: str = "sqlite"
    schema_context: str = ""
    objective: str = ""
    max_steps: int = 5
    success_threshold: float = 0.80
    task_type: str = "tutor"  # "tutor" | "optimizer"


class IssueDetail(BaseModel):
    severity: str = Field(description="critical | warning | info")
    category: str = Field(description="performance | security | readability | correctness")
    message: str
    suggestion: str


class SQLAction(BaseModel):
    query: str = Field(default="", description="A SQL SELECT or WITH (CTE) statement")
    optimized_query: Optional[str] = Field(default=None, description="Alias for optimizer tasks")
    explanation: Optional[str] = None

    def get_query(self) -> str:
        return self.optimized_query or self.query


class SQLObservation(BaseModel):
    task_description: str = ""
    schema_info: str = ""
    query_result: str = ""
    execution_error: str = ""
    reward: float = 0.0
    done: bool = False
    partial_scores: Dict[str, float] = Field(default_factory=dict)
    hint: str = ""
    step_number: int = 0
    max_steps: int = 5
    task_name: str = ""
    current_query: str = ""
    dialect: str = "sqlite"
    last_reward: float = 0.0
    cumulative_reward: float = 0.0
    issues_found: List[IssueDetail] = Field(default_factory=list)
    feedback: str = ""
    schema_context: str = ""


class SQLState(BaseModel):
    episode_id: str = ""
    step_count: int = 0
    current_task: str = ""
    task_difficulty: str = "easy"
    max_steps: int = 5
    attempts: int = 0
    best_reward: float = 0.0
    task_solved: bool = False
    schema_snapshot: str = ""
    started_at: str = ""


class RewardBreakdown(BaseModel):
    performance_delta: float = 0.0
    security_delta: float = 0.0
    readability_delta: float = 0.0
    correctness_bonus: float = 0.0
    improvement_bonus: float = 0.0
    repetition_penalty: float = 0.0
    step_penalty: float = 0.0


class StepResult(BaseModel):
    observation: SQLObservation
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


class EnvironmentState(BaseModel):
    task_name: str
    step_number: int
    max_steps: int
    original_query: str
    current_query: str
    dialect: str
    cumulative_reward: float
    done: bool
    episode_rewards: List[float]
    best_score: float
