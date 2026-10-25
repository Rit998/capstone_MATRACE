"""The checkpointed shared state. PINs and API keys never belong here."""

from typing import Any, TypedDict


class BankingState(TypedDict, total=False):
    thread_id: str
    operation_id: str
    user_messages: list[str]
    request: dict[str, Any]
    missing_fields: list[str]
    question: str
    clarification_count: int
    account_result: dict[str, Any]
    policy_result: dict[str, Any]
    result: dict[str, Any]
    confirmed: bool
    final_response: str
