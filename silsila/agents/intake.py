"""Intake Agent: Parses Arabic leave requests into structured format using Ollama."""

import json
import logging
import re
from typing import Any, Dict, Optional
from silsila.agents.base import BaseAgent
from silsila.models import LeaveRequest, LeaveType
from silsila.llm_client import (
    OllamaClient,
    extract_arabic_number_from_text,
    normalize_arabic_numerals,
)

logger = logging.getLogger(__name__)

INTAKE_SYSTEM_PROMPT = """You are an HR Leave Intake Specialist for a Saudi organization.
Your duty is to parse incoming employee leave requests written in Arabic into strict JSON.

Extract:
- "employee_id": employee identifier (default to "EMP-101" if not mentioned)
- "leave_type": one of ["annual", "sick", "emergency", "unpaid"]
  * "إجازة", "سنوية", "اعتيادية", "أبي إجازة" -> "annual"
  * "مرضية", "مريض" -> "sick"
  * "اضطرارية", "طارئة" -> "emergency"
  * "بدون راتب" -> "unpaid"
- "days": number of days requested as an integer. Notice Arabic-Indic numerals like ٥ = 5, ٣ = 3, ١٠ = 10.
- "notes": brief notes in Arabic or English summarizing the intent.

Return ONLY a valid JSON object."""


class IntakeAgent(BaseAgent):
    """Parses Arabic leave requests using local LLM (qwen2.5:3b)."""

    def __init__(self, name: str = "IntakeAgent", client: Optional[OllamaClient] = None):
        super().__init__(name)
        self.client = client or OllamaClient()

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input state expects: {"query": str, Optional["employee_id": str]}
        Output state produces: LeaveRequest.to_dict()
        """
        raw_query = state.get("query", "").strip()
        default_emp_id = state.get("employee_id", "EMP-101")

        if not raw_query:
            raise ValueError("IntakeAgent received empty query.")

        user_prompt = f"""Request: "{raw_query}"
Extract the leave details into JSON with keys: "employee_id", "leave_type", "days", "notes"."""

        # Attempt LLM inference
        extracted_data = {}
        llm_raw_output = ""
        try:
            import inspect
            gen_kwargs = {
                "prompt": user_prompt,
                "system_prompt": INTAKE_SYSTEM_PROMPT,
                "format_json": True,
                "temperature": 0.0,
            }
            if "num_predict" in inspect.signature(self.client.generate).parameters:
                gen_kwargs["num_predict"] = 100
            llm_raw_output = self.client.generate(**gen_kwargs)
            extracted_data = json.loads(llm_raw_output)
        except Exception as e:
            logger.warning(f"Ollama generation or JSON parse failed: {e}. Falling back to rule-based parser.")
            extracted_data = {}

        # Validate and normalize fields
        # Check if query explicitly contains an employee ID, else honor state default_emp_id
        emp_match = re.search(r"EMP-\d+", raw_query, re.IGNORECASE)
        if emp_match:
            employee_id = emp_match.group(0).upper()
        elif default_emp_id:
            employee_id = default_emp_id
        else:
            employee_id = extracted_data.get("employee_id") or "EMP-101"

        leave_type_raw = str(extracted_data.get("leave_type", "annual")).lower()

        # Normalize leave type
        if "sick" in leave_type_raw or "مرض" in raw_query:
            leave_type = LeaveType.SICK
        elif "emergency" in leave_type_raw or "اضطرار" in raw_query or "طارئ" in raw_query:
            leave_type = LeaveType.EMERGENCY
        elif "unpaid" in leave_type_raw or "بدون راتب" in raw_query or "غير مدفوع" in raw_query or "غير مدفوعة" in raw_query:
            leave_type = LeaveType.UNPAID
        else:
            leave_type = LeaveType.ANNUAL

        # Normalize days
        days = None
        if "days" in extracted_data:
            try:
                days = int(extracted_data["days"])
            except (ValueError, TypeError):
                days = None

        if days is None or days <= 0:
            # Fallback to direct Arabic extractor
            fallback_days = extract_arabic_number_from_text(raw_query)
            days = fallback_days if fallback_days and fallback_days > 0 else 1

        notes = extracted_data.get("notes") or f"Leave request parsed: {days} days"

        leave_request = LeaveRequest(
            employee_id=employee_id,
            leave_type=leave_type,
            days=days,
            notes=str(notes),
            original_query=raw_query,
        )

        result = leave_request.to_dict()
        # Attach internal metadata about parsing
        result["_intake_meta"] = {
            "llm_output": llm_raw_output,
            "normalized_query": normalize_arabic_numerals(raw_query),
        }
        return result
