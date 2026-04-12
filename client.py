"""SQLens — HTTP client (OpenEnv EnvClient interface)."""
from __future__ import annotations
import requests
from models import SQLAction, SQLObservation, SQLState, StepResult


class SQLQueryEnv:
    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self._session = None

    def __enter__(self):
        self._session = requests.Session()
        return self

    def __exit__(self, *args):
        if self._session: self._session.close()

    def _s(self): return self._session or requests.Session()

    def reset(self, task_id: str = "easy_select", seed: int = 42) -> SQLObservation:
        r = self._s().post(f"{self.base_url}/reset", json={"task_id":task_id,"task":task_id,"seed":seed}, timeout=30)
        r.raise_for_status()
        d = r.json()
        return SQLObservation(**{k:d.get(k,v) for k,v in SQLObservation().model_dump().items()})

    def step(self, action: SQLAction) -> StepResult:
        sql = action.get_query()
        r = self._s().post(f"{self.base_url}/step", json={"query":sql,"optimized_query":sql}, timeout=30)
        r.raise_for_status()
        d = r.json()
        o = d.get("observation", {})
        obs = SQLObservation(**{k:o.get(k,v) for k,v in SQLObservation().model_dump().items()})
        return StepResult(observation=obs, reward=d.get("reward",0.0), done=d.get("done",False), info=d.get("info",{}))

    def state(self) -> SQLState:
        r = self._s().get(f"{self.base_url}/state", timeout=15)
        r.raise_for_status()
        d = r.json()
        return SQLState(**{k:d.get(k,v) for k,v in SQLState().model_dump().items()})

    def health(self) -> bool:
        try: return self._s().get(f"{self.base_url}/health", timeout=10).status_code == 200
        except: return False
