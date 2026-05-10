import json
import os
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_STORAGE_DIR = os.path.join(os.getcwd(), "discovered_equations")


@dataclass
class EquationRecord:
    equation_id: str
    expression: str
    sympy_expression: Optional[str]
    r2_score: float
    mse: float
    mae: float
    complexity: int
    variable_names: list
    kan_architecture: list
    symbolic_lib: list
    iteration: int
    auxiliary_variables: list
    hypothesis: Optional[str]
    llm_interpretation: Optional[str]
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "EquationRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, json_str: str) -> "EquationRecord":
        return cls.from_dict(json.loads(json_str))


class EquationStorage:
    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = storage_dir or DEFAULT_STORAGE_DIR
        self._equations: dict[str, EquationRecord] = {}
        os.makedirs(self.storage_dir, exist_ok=True)
        self._load_existing()

    def _load_existing(self):
        index_path = os.path.join(self.storage_dir, "index.json")
        if os.path.exists(index_path):
            try:
                with open(index_path, "r") as f:
                    index = json.load(f)
                for eq_id in index.get("equation_ids", []):
                    eq_path = os.path.join(self.storage_dir, f"{eq_id}.json")
                    if os.path.exists(eq_path):
                        with open(eq_path, "r") as f:
                            record = EquationRecord.from_dict(json.load(f))
                            self._equations[eq_id] = record
                logger.info(f"Loaded {len(self._equations)} existing equations from storage")
            except Exception as e:
                logger.warning(f"Failed to load existing equations: {e}")

    def _save_index(self):
        index_path = os.path.join(self.storage_dir, "index.json")
        index = {
            "equation_ids": list(self._equations.keys()),
            "last_updated": time.time(),
        }
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

    def store(self, record: EquationRecord) -> str:
        self._equations[record.equation_id] = record
        eq_path = os.path.join(self.storage_dir, f"{record.equation_id}.json")
        with open(eq_path, "w") as f:
            json.dump(record.to_dict(), f, indent=2)
        self._save_index()
        logger.info(f"Stored equation {record.equation_id}")
        return record.equation_id

    def retrieve(self, equation_id: str) -> Optional[EquationRecord]:
        return self._equations.get(equation_id)

    def list_equations(self) -> list[EquationRecord]:
        return sorted(self._equations.values(), key=lambda r: r.r2_score, reverse=True)

    def search_by_r2(self, min_r2: float = 0.0) -> list[EquationRecord]:
        return [r for r in self._equations.values() if r.r2_score >= min_r2]

    def search_by_complexity(self, max_complexity: int) -> list[EquationRecord]:
        return [r for r in self._equations.values() if r.complexity <= max_complexity]

    def get_best_equation(self) -> Optional[EquationRecord]:
        if not self._equations:
            return None
        return max(self._equations.values(), key=lambda r: r.r2_score)

    def delete(self, equation_id: str) -> bool:
        if equation_id not in self._equations:
            return False
        del self._equations[equation_id]
        eq_path = os.path.join(self.storage_dir, f"{equation_id}.json")
        if os.path.exists(eq_path):
            os.remove(eq_path)
        self._save_index()
        return True

    def export_all(self) -> list[dict]:
        return [r.to_dict() for r in self._equations.values()]
