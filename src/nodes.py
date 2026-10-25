"""Specialist agents and deterministic, SQLite-backed simulated banking tools.

The LLM proposes actions. Only these Python tools can authenticate, read private
account data, approve policy conditions, or change balances.
"""

import json
import re
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from langgraph.types import interrupt
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import OpenAI

from src import prompts
from src.objects import (
    Account, AccountDecision, BankingPolicy, Customer, FinalAnswer,
    PolicyDecision, SupervisorDecision, Transaction, TransactionDecision,
)
from src.states import BankingState

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def safe_inputs(inputs: dict) -> dict:
    """Dedicated login arguments are excluded from traces, including the PIN."""
    return {key: value for key, value in inputs.items()
            if key not in {"self", "pin", "api_key"}}


def failure(code: str, message: str) -> dict:
    return {"status": "rejected", "code": code, "message": message}


class BankTools:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS bank (id INTEGER PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS sessions
                    (thread_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS session_owners
                    (thread_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS confirmations
                    (operation_id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, payload TEXT NOT NULL);
            """)

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def initialized(self) -> bool:
        with self.connect() as db:
            return db.execute("SELECT 1 FROM bank WHERE id = 1").fetchone() is not None

    def reset(self, scenario_id: str = "normal") -> None:
        """Copy immutable fixtures and apply only the scenario's initial conditions."""
        scenarios = read_json(DATA_DIR / "scenarios.json")
        scenario = next((s for s in scenarios if s["scenario_id"] == scenario_id), None)
        if scenario is None:
            raise ValueError(f"Unknown scenario: {scenario_id}")
        state = {
            "customers": read_json(DATA_DIR / "customers.json"),
            "accounts": read_json(DATA_DIR / "accounts.json"),
            "transactions": read_json(DATA_DIR / "transactions.json"),
            "policy": read_json(DATA_DIR / "banking_policy.json"),
            "transfers": {},
        }
        for override in scenario["account_overrides"]:
            account = next(a for a in state["accounts"]
                           if a["account_id"] == override["account_id"])
            account.update(override)
        for item in state["customers"]:
            Customer.model_validate(item)
        for item in state["accounts"]:
            Account.model_validate(item)
        for item in state["transactions"]:
            Transaction.model_validate(item)
        BankingPolicy.model_validate(state["policy"])
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT OR REPLACE INTO bank VALUES (1, ?)", (json.dumps(state),))
            db.execute("DELETE FROM sessions")
            db.execute("DELETE FROM session_owners")
            db.execute("DELETE FROM confirmations")

    def snapshot(self) -> dict:
        """Local task checks only. Never send this full database to an agent."""
        with self.connect() as db:
            return self._load(db)

    @staticmethod
    def _load(db) -> dict:
        row = db.execute("SELECT data FROM bank WHERE id = 1").fetchone()
        if row is None:
            raise ValueError("Bank is not initialized. Reset a scenario first.")
        return json.loads(row[0])

    @staticmethod
    def _customer(db, thread_id: str) -> str | None:
        row = db.execute("SELECT customer_id FROM sessions WHERE thread_id = ?",
                         (thread_id,)).fetchone()
        return row[0] if row else None

    @staticmethod
    def _owned(state: dict, customer_id: str | None, account_ids: list[str]) -> dict:
        if not customer_id:
            return failure("authentication_required", "Log in before accessing accounts.")
        for account_id in account_ids:
            account = next((a for a in state["accounts"]
                            if a["account_id"] == account_id), None)
            if account is None:
                return failure("account_not_found", "The requested account does not exist.")
            if account["customer_id"] != customer_id:
                return failure("unauthorized", "You may access only your own accounts.")
        return {"status": "ok"}

    @traceable(name="tool.authenticate_customer", process_inputs=safe_inputs)
    def authenticate(self, thread_id: str, customer_id: str, pin: str) -> dict:
        # Demo credentials only. A production authentication system is out of scope.
        import hmac

        with self.connect() as db:
            state = self._load(db)
            db.execute("DELETE FROM sessions WHERE thread_id = ?", (thread_id,))
            customer = next((c for c in state["customers"]
                             if c["customer_id"] == customer_id), None)
            if not customer or not hmac.compare_digest(customer["demo_pin"], pin):
                return failure("authentication_failed", "Invalid demo customer ID or PIN.")
            owner = db.execute("SELECT customer_id FROM session_owners WHERE thread_id = ?",
                               (thread_id,)).fetchone()
            if owner and owner[0] != customer_id:
                return failure("session_owner_mismatch", "Resume with this session's original customer ID.")
            db.execute("INSERT OR IGNORE INTO session_owners VALUES (?, ?)", (thread_id, customer_id))
            db.execute("INSERT INTO sessions VALUES (?, ?)", (thread_id, customer_id))
            return {"status": "ok", "customer_id": customer_id}

    @traceable(name="tool.verify_authenticated_session", process_inputs=safe_inputs)
    def authenticated_session(self, thread_id: str) -> dict:
        with self.connect() as db:
            customer_id = self._customer(db, thread_id)
            if not customer_id:
                return failure("authentication_required", "Log in before accessing accounts.")
            return {"status": "ok", "customer_id": customer_id}

    @traceable(name="tool.get_balance", process_inputs=safe_inputs)
    def balance(self, thread_id: str, account_id: str) -> dict:
        with self.connect() as db:
            state = self._load(db)
            access = self._owned(state, self._customer(db, thread_id), [account_id])
            if access["status"] != "ok":
                return access
            account = next(a for a in state["accounts"] if a["account_id"] == account_id)
            return {"status": "ok", "code": "balance_retrieved", "account": account}

    @traceable(name="tool.get_recent_transactions", process_inputs=safe_inputs)
    def transactions(self, thread_id: str, account_id: str) -> dict:
        with self.connect() as db:
            state = self._load(db)
            access = self._owned(state, self._customer(db, thread_id), [account_id])
            if access["status"] != "ok":
                return access
            rows = [t for t in state["transactions"] if t["account_id"] == account_id]
            rows.sort(key=lambda t: (t["created_at"], t["transaction_id"]), reverse=True)
            return {"status": "ok", "code": "transactions_retrieved",
                    "account_id": account_id,
                    "transactions": rows[:state["policy"]["recent_transaction_count"]]}

    @traceable(name="tool.verify_account_ownership", process_inputs=safe_inputs)
    def verify_transfer(self, thread_id: str, request: dict) -> dict:
        with self.connect() as db:
            state = self._load(db)
            return self._owned(state, self._customer(db, thread_id),
                               [request["source_account_id"], request["destination_account_id"]])

    def _eligibility(self, state: dict, customer_id: str | None, request: dict) -> dict:
        source_id = request["source_account_id"]
        destination_id = request["destination_account_id"]
        access = self._owned(state, customer_id, [source_id, destination_id])
        if access["status"] != "ok":
            return access
        amount = request["amount_cents"]
        if type(amount) is not int or amount <= 0:
            return failure("invalid_amount", "Transfer amount must be positive integer cents.")
        if source_id == destination_id:
            return failure("same_account", "Source and destination must be different accounts.")
        source = next(a for a in state["accounts"] if a["account_id"] == source_id)
        destination = next(a for a in state["accounts"] if a["account_id"] == destination_id)
        if source["status"] != "active" or destination["status"] != "active":
            return failure("inactive_account", "Both transfer accounts must be active.")
        if source["balance_cents"] < amount:
            return failure("insufficient_funds", "Source account has insufficient funds.")
        policy = state["policy"]
        if amount > policy["max_transfer_cents"]:
            return failure("transfer_limit_exceeded", "The per-transfer limit would be exceeded.")
        today = datetime.now(timezone.utc).date().isoformat()
        spent = sum(t["amount_cents"] for t in state["transfers"].values()
                    if t["customer_id"] == customer_id and t["date"] == today)
        if spent + amount > policy["daily_transfer_limit_cents"]:
            return failure("daily_limit_exceeded", "The customer's daily transfer limit would be exceeded.")
        return {"status": "eligible", "code": "transfer_eligible", "transfer": {
            "source_account_id": source_id, "destination_account_id": destination_id,
            "amount_cents": amount, "currency": policy["currency"]}}

    @traceable(name="tool.assess_transfer_policy", process_inputs=safe_inputs)
    def assess_transfer(self, thread_id: str, request: dict) -> dict:
        with self.connect() as db:
            return self._eligibility(self._load(db), self._customer(db, thread_id), request)

    @staticmethod
    def _payload(request: dict) -> str:
        return json.dumps({key: request[key] for key in (
            "source_account_id", "destination_account_id", "amount_cents")}, sort_keys=True)

    @traceable(name="tool.record_explicit_confirmation", process_inputs=safe_inputs)
    def confirm(self, thread_id: str, operation_id: str, request: dict, reply: str) -> dict:
        if reply.strip().casefold() != "confirm":
            return failure("confirmation_required", "Type CONFIRM for the displayed transfer.")
        with self.connect() as db:
            if not self._customer(db, thread_id):
                return failure("authentication_required", "Log in before confirming.")
            db.execute("INSERT OR REPLACE INTO confirmations VALUES (?, ?, ?)",
                       (operation_id, thread_id, self._payload(request)))
        return {"status": "confirmed"}

    @traceable(name="tool.execute_simulated_transfer", process_inputs=safe_inputs)
    def execute_transfer(self, thread_id: str, operation_id: str, request: dict) -> dict:
        with self.connect() as db:
            # Serialize checking and writing: two simultaneous calls cannot spend twice.
            db.execute("BEGIN IMMEDIATE")
            state = self._load(db)
            customer_id = self._customer(db, thread_id)
            if not customer_id:
                return failure("authentication_required", "Log in before transferring.")
            payload = self._payload(request)
            confirmation = db.execute(
                "SELECT thread_id, payload FROM confirmations WHERE operation_id = ?",
                (operation_id,)).fetchone()
            if confirmation != (thread_id, payload):
                return failure("confirmation_required", "Explicit confirmation must match this transfer.")
            previous = state["transfers"].get(operation_id)
            if previous:
                if previous["payload"] != payload or previous["customer_id"] != customer_id:
                    return failure("idempotency_conflict", "Transfer ID was already used for another request.")
                return previous["result"]
            eligibility = self._eligibility(state, customer_id, request)
            if eligibility["status"] != "eligible":
                return eligibility
            transfer = eligibility["transfer"]
            source = next(a for a in state["accounts"]
                          if a["account_id"] == transfer["source_account_id"])
            destination = next(a for a in state["accounts"]
                               if a["account_id"] == transfer["destination_account_id"])
            amount = transfer["amount_cents"]
            source["balance_cents"] -= amount
            destination["balance_cents"] += amount
            now = datetime.now(timezone.utc).isoformat()
            for suffix, account, signed in (("debit", source, -amount),
                                             ("credit", destination, amount)):
                state["transactions"].append({
                    "transaction_id": f"{operation_id}-{suffix}",
                    "account_id": account["account_id"], "amount_cents": signed,
                    "description": "Simulated internal transfer", "created_at": now,
                    "transfer_id": operation_id})
            result = {"status": "completed", "code": "transfer_completed",
                      "transfer_id": operation_id, **transfer,
                      "source_balance_cents": source["balance_cents"],
                      "destination_balance_cents": destination["balance_cents"]}
            state["transfers"][operation_id] = {
                "customer_id": customer_id, "amount_cents": amount, "date": now[:10],
                "payload": payload, "result": result}
            db.execute("UPDATE bank SET data = ? WHERE id = 1", (json.dumps(state),))
            return result


class StructuredLLM:
    """Actual model calls; there is deliberately no silent offline fallback."""

    def __init__(self, model: str):
        self.model = model
        self.client = wrap_openai(OpenAI())

    def decide(self, agent: str, prompt: str, data: dict, schema):
        @traceable(name=f"{agent}.decision", process_inputs=safe_inputs)
        def call():
            completion = self.client.chat.completions.parse(
                model=self.model,
                messages=[{"role": "system", "content": prompt},
                          {"role": "user", "content": json.dumps(data)}],
                response_format=schema,
            )
            message = completion.choices[0].message
            if message.refusal or message.parsed is None:
                raise RuntimeError(f"{agent} did not return a structured decision.")
            return message.parsed
        return call()


def explicit_request(request: dict, messages: list[str]) -> dict:
    """Reject model-invented IDs/amounts. Numeric currency syntax is intentional."""
    text = "\n".join(messages)
    request = dict(request)
    for key in ("account_id", "source_account_id", "destination_account_id"):
        value = request[key]
        if value and not re.search(r"(?<![\w-])" + re.escape(value) + r"(?![\w-])", text, re.I):
            request[key] = None
    values = set()
    pattern = r"\$\s*(-?\d+(?:\.\d{1,2})?)(?![\d.])|(?<![\w.-])(-?\d+(?:\.\d{1,2})?)\s*(cents?|USD|dollars?)\b"
    for match in re.finditer(pattern, text, re.I):
        number = Decimal(match[1] or match[2])
        cents = number if (match[3] or "").lower().startswith("cent") else number * 100
        if cents == cents.to_integral_value():
            values.add(int(cents))
    if request["amount_cents"] not in values:
        request["amount_cents"] = None
    return request


def missing_fields(request: dict) -> list[str]:
    if request["operation"] in {"balance", "transactions"}:
        fields = ["account_id"]
    elif request["operation"] == "transfer":
        fields = ["source_account_id", "destination_account_id", "amount_cents"]
    else:
        return []
    return [field for field in fields if request.get(field) is None]


def make_nodes(bank: BankTools, llm):
    """Closures keep clients and credentials outside persisted graph state."""

    def supervisor(state: BankingState):
        decision = llm.decide("Supervisor", prompts.SUPERVISOR,
                              {"user_messages": state["user_messages"]}, SupervisorDecision)
        request = explicit_request(decision.request.model_dump(), state["user_messages"])
        missing = missing_fields(request)
        update = {"request": request, "missing_fields": missing,
                  "question": decision.clarification_question}
        if request["operation"] == "unsupported":
            update["result"] = failure("unsupported_operation", "Supported: balance, transactions, internal transfer.")
        elif missing and state.get("clarification_count", 0) >= 5:
            update["result"] = failure("missing_information", "Required request details were not supplied.")
        return update

    def clarify(state: BankingState):
        reply = interrupt({"kind": "missing_information", "missing_fields": state["missing_fields"],
                           "question": state["question"] or "Please supply the missing fields.",
                           "hint": "Use exact account IDs and numeric amounts, e.g. $25.00 or 2500 cents. Type CANCEL to stop."})
        if not isinstance(reply, str):
            raise ValueError("A clarification reply must be text.")
        if reply.strip().casefold() == "cancel":
            return {"result": {"status": "cancelled", "code": "request_cancelled"}}
        return {"user_messages": state["user_messages"] + [reply],
                "clarification_count": state.get("clarification_count", 0) + 1}

    def account_agent(state: BankingState):
        request = state["request"]
        authentication = bank.authenticated_session(state["thread_id"])
        if authentication["status"] != "ok":
            return {"result": authentication}
        choice = llm.decide("AccountAgent", prompts.ACCOUNT,
                            {"request": request, "authentication": authentication}, AccountDecision)
        expected = "verify_transfer" if request["operation"] == "transfer" else request["operation"]
        if choice.action != expected:
            return {"result": failure("agent_stopped", "Account Agent did not select the required tool.")}
        if expected == "balance":
            result = bank.balance(state["thread_id"], request["account_id"])
        elif expected == "transactions":
            result = bank.transactions(state["thread_id"], request["account_id"])
        else:
            result = bank.verify_transfer(state["thread_id"], request)
        if expected != "verify_transfer" or result["status"] != "ok":
            return {"account_result": result, "result": result}
        return {"account_result": result}

    def policy_agent(state: BankingState):
        choice = llm.decide("PolicyAgent", prompts.POLICY,
                            {"request": state["request"], "account_result": state["account_result"]},
                            PolicyDecision)
        if choice.action != "assess_transfer":
            return {"result": failure("agent_stopped", "Policy Agent stopped the request.")}
        result = bank.assess_transfer(state["thread_id"], state["request"])
        if result["status"] != "eligible":
            return {"policy_result": result, "result": result}
        return {"policy_result": result}

    def confirmation(state: BankingState):
        transfer = state["policy_result"]["transfer"]
        amount = transfer["amount_cents"]
        payload = {"kind": "transfer_confirmation", **transfer,
                   "message": f"Transfer ${amount // 100}.{amount % 100:02d} USD from "
                              f"{transfer['source_account_id']} to {transfer['destination_account_id']}? "
                              "Type CONFIRM to approve exactly this transfer, or CANCEL to decline."}
        while True:
            reply = interrupt(payload)
            if isinstance(reply, str) and reply.strip().casefold() == "cancel":
                return {"confirmed": False, "result": {
                    "status": "cancelled", "code": "confirmation_declined",
                    "message": "Transfer cancelled; no money was moved."}}
            if isinstance(reply, str) and reply.strip().casefold() == "confirm":
                result = bank.confirm(state["thread_id"], state["operation_id"], state["request"], reply)
                if result["status"] != "confirmed":
                    return {"confirmed": False, "result": result}
                return {"confirmed": True}
            payload = {**payload, "error": "No consent recorded. Reply CONFIRM or CANCEL."}

    def transaction_agent(state: BankingState):
        if not state.get("confirmed"):
            return {"result": failure("confirmation_required", "Transfer was not confirmed.")}
        choice = llm.decide("TransactionAgent", prompts.TRANSACTION,
                            {"request": state["request"], "policy_result": state["policy_result"],
                             "confirmed": state["confirmed"]}, TransactionDecision)
        if choice.action != "execute_transfer":
            return {"result": failure("agent_stopped", "Transaction Agent stopped the transfer.")}
        return {"result": bank.execute_transfer(state["thread_id"], state["operation_id"], state["request"])}

    def final_response(state: BankingState):
        response = llm.decide("SupervisorFinal", prompts.FINAL,
                              {"tool_result": state["result"]}, FinalAnswer)
        return {"final_response": response.message}

    return {"Supervisor": supervisor, "Clarification": clarify, "AccountAgent": account_agent,
            "PolicyAgent": policy_agent, "Confirmation": confirmation,
            "TransactionAgent": transaction_agent, "SupervisorFinal": final_response}
