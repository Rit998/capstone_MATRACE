"""Data records and the structured decisions requested from the LLM."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Customer(Record):
    customer_id: str
    name: str
    demo_pin: str  # Fictional, public demo credentials; never sent to the model.


class Account(Record):
    account_id: str
    customer_id: str
    account_type: Literal["checking", "savings"]
    balance_cents: StrictInt = Field(ge=0)
    status: Literal["active", "frozen", "closed"]


class Transaction(Record):
    transaction_id: str
    account_id: str
    amount_cents: StrictInt  # Signed: debit is negative, credit is positive.
    description: str
    created_at: str
    transfer_id: str | None = None


class BankingPolicy(Record):
    currency: Literal["USD"]
    max_transfer_cents: StrictInt = Field(gt=0)
    daily_transfer_limit_cents: StrictInt = Field(gt=0)
    recent_transaction_count: StrictInt = Field(gt=0, le=100)
    rules: list[str]


class BankingRequest(Record):
    operation: Literal["balance", "transactions", "transfer", "unsupported"]
    account_id: str | None
    source_account_id: str | None
    destination_account_id: str | None
    amount_cents: StrictInt | None


class SupervisorDecision(Record):
    request: BankingRequest
    clarification_question: str


class AccountDecision(Record):
    action: Literal["balance", "transactions", "verify_transfer", "stop"]


class PolicyDecision(Record):
    action: Literal["assess_transfer", "stop"]


class TransactionDecision(Record):
    action: Literal["execute_transfer", "stop"]


class FinalAnswer(Record):
    message: str
