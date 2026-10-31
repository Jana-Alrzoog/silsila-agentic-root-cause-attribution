"""Configuration settings for SILSILA."""

import os

# Ollama local endpoint and model
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))
SILSILA_DISABLE_LLM = os.getenv("SILSILA_DISABLE_LLM", "0") in ("1", "true", "True")

# Simulated HR Policy Defaults
DEFAULT_ACCRUED_ANNUAL_LEAVE = 21  # Standard annual leave days in Saudi labor system
MAX_CONSECUTIVE_LEAVE_WITHOUT_VP_APPROVAL = 14
DEFAULT_MONTHLY_SALARY = 15000.0  # SAR
DAYS_IN_PAYROLL_MONTH = 30
DEFAULT_DAILY_RATE = DEFAULT_MONTHLY_SALARY / DAYS_IN_PAYROLL_MONTH  # 500.0 SAR/day
