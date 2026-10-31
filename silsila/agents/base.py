"""Base class for SILSILA agents."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):
    """Abstract base agent in the workflow chain."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent task given input state, returning new output state."""
        pass
