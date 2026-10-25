"""Interactive CLI and reproducible, independently checked task execution."""

import argparse
import getpass
import json
import os
import sqlite3
from pathlib import Path
from uuid import UUID, uuid4

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from openai import APIError

from src.edges import build_graph
from src.nodes import DATA_DIR, BankTools, StructuredLLM, read_json

ROOT = Path(__file__).resolve().parent


def initial_state(thread_id: str, message: str) -> dict:
    # Clear every per-request field even when reusing an interactive conversation.
    return {"thread_id": thread_id, "operation_id": str(uuid4()),
            "user_messages": [message], "request": {}, "missing_fields": [],
            "question": "", "clarification_count": 0, "account_result": {},
            "policy_result": {}, "result": {}, "confirmed": False, "final_response": ""}


def graph_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}


def check_task(task: dict, before: dict, after: dict, state: dict) -> list[str]:
    """Deterministic functional assertions. No LLM judges or research metrics."""
    errors = []

    def check(condition, description):
        if not condition:
            errors.append(description)

    expected = task["expected_outcome"]
    result = state.get("result", {})
    check(result.get("status") == expected["status"], "Unexpected result status")
    check(result.get("code") == expected["code"], "Unexpected result code")
    for path, value in expected["result_fields"].items():
        actual = result
        try:
            for part in path.split("."):
                actual = actual[int(part)] if isinstance(actual, list) else actual[part]
        except (KeyError, IndexError, TypeError, ValueError):
            actual = None
        check(actual == value, f"Result field mismatch: {path}")

    checks = task["final_state_checks"]
    deltas = checks["account_balance_deltas_cents"]
    old_accounts = {a["account_id"]: a for a in before["accounts"]}
    new_accounts = {a["account_id"]: a for a in after["accounts"]}
    check(old_accounts.keys() == new_accounts.keys(), "Account IDs changed")
    for account_id, account in old_accounts.items():
        expected_account = {**account, "balance_cents": account["balance_cents"] + deltas.get(account_id, 0)}
        check(new_accounts.get(account_id) == expected_account, f"Account mismatch: {account_id}")
    check(after["customers"] == before["customers"], "Customer fixtures changed")
    check(after["policy"] == before["policy"], "Policy changed")
    check(len(after["transfers"]) == checks["transfer_count"], "Incorrect transfer count")
    old_rows = before["transactions"]
    new_rows = after["transactions"][len(old_rows):]
    check(after["transactions"][:len(old_rows)] == old_rows, "Existing transactions changed")
    check(len(new_rows) == checks["new_transaction_count"], "Incorrect new transaction count")
    if checks["transfer_count"]:
        transfer_id = result.get("transfer_id")
        check(transfer_id in after["transfers"], "Transfer is missing from the ledger")
        check(len({t["transaction_id"] for t in after["transactions"]}) == len(after["transactions"]),
              "Duplicate transaction IDs")
        actual_deltas = {}
        for row in new_rows:
            check(row["transfer_id"] == transfer_id, "Transaction references wrong transfer")
            actual_deltas[row["account_id"]] = actual_deltas.get(row["account_id"], 0) + row["amount_cents"]
        check(actual_deltas == deltas, "Transaction amounts do not match account changes")
        check(sum(actual_deltas.values()) == 0, "Transfer did not conserve money")
    return errors


def run_task(task: dict, llm, work_root: Path) -> dict:
    """Each task gets a fresh scenario copy AND a fresh checkpoint database."""
    thread_id = str(uuid4())
    folder = work_root / "tasks" / task["task_id"] / thread_id
    bank = BankTools(folder / "bank.sqlite")
    bank.reset(task["scenario_id"])
    before = bank.snapshot()
    customer = next(c for c in before["customers"] if c["customer_id"] == task["customer_id"])
    auth = bank.authenticate(thread_id, customer["customer_id"], customer["demo_pin"])
    if auth["status"] != "ok":
        raise RuntimeError("Task demo login failed")
    pauses = []
    with SqliteSaver.from_conn_string(str(folder / "checkpoints.sqlite")) as saver:
        graph = build_graph(bank, llm, saver)
        config = graph_config(thread_id)
        state = graph.invoke(initial_state(thread_id, task["user_message"]), config)
        for reply in task["scripted_follow_up_replies"]:
            if not state.get("__interrupt__"):
                raise RuntimeError("Task supplied a reply but the graph was not waiting for input")
            pauses.extend(item.value for item in state["__interrupt__"])
            state = graph.invoke(Command(resume=reply), config)
        pending = bool(state.get("__interrupt__"))
        if pending:
            pauses.extend(item.value for item in state["__interrupt__"])
        errors = check_task(task, before, bank.snapshot(), state)
        if pending:
            errors.append("Graph still waiting for user input")
        if not state.get("final_response"):
            errors.append("Supervisor did not produce a final response")
        report = {"task_id": task["task_id"], "passed": not errors, "errors": errors,
                  "result": state.get("result"), "final_response": state.get("final_response"),
                  "pauses": pauses, "working_directory": str(folder)}
        (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report


def show_interrupts(state: dict):
    for item in state.get("__interrupt__", ()):
        print(json.dumps(item.value, indent=2))


def interactive(llm, work_root: Path, resume: str | None):
    thread_id = str(UUID(resume)) if resume else str(uuid4())
    folder = work_root / "sessions" / thread_id
    if resume and not (folder / "bank.sqlite").exists():
        raise ValueError("That session does not exist in this working directory.")
    bank = BankTools(folder / "bank.sqlite")
    if not bank.initialized():
        bank.reset()
    print("Fictional banking demo. PINs are public fixtures; never enter real credentials.")
    print(f"Session: {thread_id}\nResume command: python main.py --resume {thread_id}")
    customer_id = input("Demo customer ID (e.g. C001): ").strip()
    pin = getpass.getpass("Fictional demo PIN (C001 uses 1101): ")
    auth = bank.authenticate(thread_id, customer_id, pin)
    del pin
    if auth["status"] != "ok":
        raise ValueError(auth["message"])
    config = graph_config(thread_id)
    with SqliteSaver.from_conn_string(str(folder / "checkpoints.sqlite")) as saver:
        graph = build_graph(bank, llm, saver)
        snapshot = graph.get_state(config)
        if snapshot.values and snapshot.next:
            # Interrupts persist across processes; failed model calls can also resume.
            if snapshot.interrupts:
                state = {**snapshot.values, "__interrupt__": snapshot.interrupts}
            else:
                state = graph.invoke(None, config)
        else:
            state = None
        while True:
            if state and state.get("__interrupt__"):
                show_interrupts(state)  # Display exact details before collecting consent.
                reply = input("Reply (or /exit to leave paused): ")
                if reply.strip() == "/exit":
                    return
                state = graph.invoke(Command(resume=reply), config)
                continue
            if state:
                print("\nSupervisor:", state["final_response"])
                print("Tool result:", json.dumps(state["result"], indent=2))
            message = input("\nBanking request (or /exit): ").strip()
            if message == "/exit":
                return
            if not message:
                state = None
                continue
            state = graph.invoke(initial_state(thread_id, message), config)


def main() -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="MA-TRACE stage-one simulated banking MAS")
    parser.add_argument("--task", help="Run a task ID, such as T03, or all")
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--resume", help="Resume an interactive session by its UUID")
    parser.add_argument("--work-dir", type=Path, default=ROOT / ".runtime")
    args = parser.parse_args()
    tasks = read_json(DATA_DIR / "tasks.json")
    if args.list_tasks:
        for task in tasks:
            print(f"{task['task_id']} [{task['scenario_id']}]: {task['user_message']}")
        return 0
    selected = [t for t in tasks if args.task in {"all", t["task_id"]}]
    if args.task and not selected:
        parser.error("Unknown task ID. Run --list-tasks.")
    if args.task and args.resume:
        parser.error("--task and --resume cannot be combined.")
    if not os.getenv("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY securely in the environment or local .env. See .env.example.")
    tracing = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    if tracing and not os.getenv("LANGSMITH_API_KEY"):
        parser.error("LANGSMITH_TRACING=true requires LANGSMITH_API_KEY, or disable tracing.")
    os.environ["LANGSMITH_TRACING"] = "true" if tracing else "false"
    try:
        llm = StructuredLLM(os.getenv("OPENAI_MODEL") or "gpt-4o-mini")
        if args.task:
            failed = False
            for task in selected:
                report = run_task(task, llm, args.work_dir)
                print(json.dumps(report, indent=2))
                failed = failed or not report["passed"]
            return 1 if failed else 0
        interactive(llm, args.work_dir, args.resume)
        return 0
    except APIError as error:
        # Provider exception bodies can echo inputs. Keep them out of console logs.
        print(f"Model request failed ({type(error).__name__}). Check API credentials, model, and connectivity.")
        print("Interactive checkpoints remain saved; use the printed resume command.")
        return 1
    except (ValueError, RuntimeError, sqlite3.Error) as error:
        print(f"Cannot complete request: {error}")
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\nStopped. Any paused interactive session remains saved.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
