"""Offline functional checks. ScriptedLLM is a test double, never an app mode."""

import json
import io
import os
import re
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

# Unit tests must never upload traces or require provider credentials.
os.environ["LANGSMITH_TRACING"] = "false"

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from main import check_task, graph_config, initial_state, interactive, run_task
from src import prompts
from src.edges import build_graph
from src.nodes import BankTools, DATA_DIR, explicit_request, read_json, safe_inputs


class ScriptedLLM:
    """Predictable structured decisions used only to test wiring and tools.

    This does not test model interpretation quality. It has no access to task
    IDs, scenario labels, expected outcomes, or final-state checks.
    """

    def __init__(self):
        self.calls = []

    def decide(self, agent, prompt, data, schema):
        self.calls.append((agent, data))
        if agent == "Supervisor":
            text = " ".join(data["user_messages"])
            accounts = re.findall(r"\bA\d{3}\b", text)
            operation = ("transfer" if text.lower().startswith("transfer") else
                         "transactions" if "transactions" in text.lower() else "balance")
            amount = re.search(r"\$(\d+)\.(\d{2})", text)
            request = {"operation": operation,
                       "account_id": accounts[0] if accounts and operation != "transfer" else None,
                       "source_account_id": accounts[0] if accounts and operation == "transfer" else None,
                       "destination_account_id": accounts[1] if len(accounts) > 1 and operation == "transfer" else None,
                       "amount_cents": int(amount[1]) * 100 + int(amount[2]) if amount else None}
            return schema(request=request, clarification_question="Please supply the missing account ID or amount.")
        if agent == "AccountAgent":
            operation = data["request"]["operation"]
            return schema(action="verify_transfer" if operation == "transfer" else operation)
        if agent == "PolicyAgent":
            return schema(action="assess_transfer")
        if agent == "TransactionAgent":
            return schema(action="execute_transfer")
        if agent == "SupervisorFinal":
            return schema(message=json.dumps(data["tool_result"]))
        raise AssertionError(agent)


class BankingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bank = BankTools(self.root / "bank.sqlite")
        self.bank.reset()
        self.bank.authenticate("thread", "C001", "1101")
        self.request = {"operation": "transfer", "account_id": None,
                        "source_account_id": "A001", "destination_account_id": "A002",
                        "amount_cents": 2500}

    def tearDown(self):
        self.temp.cleanup()

    def confirm_and_execute(self, operation="op", request=None):
        request = request or self.request
        self.bank.confirm("thread", operation, request, "CONFIRM")
        return self.bank.execute_transfer("thread", operation, request)

    def test_all_ten_supplied_tasks(self):
        tasks = read_json(DATA_DIR / "tasks.json")
        self.assertEqual(len(tasks), 10)
        for task in tasks:
            with self.subTest(task=task["task_id"]):
                llm = ScriptedLLM()
                report = run_task(task, llm, self.root)
                self.assertTrue(report["passed"], report)
                names = [agent for agent, _ in llm.calls]
                if task["task_id"] in {"T01", "T02", "T05", "T06"}:
                    self.assertNotIn("PolicyAgent", names)
                    self.assertNotIn("TransactionAgent", names)
                for _, data in llm.calls:
                    serialized = json.dumps(data)
                    self.assertNotIn("demo_pin", serialized)
                    self.assertNotIn("expected_outcome", serialized)
                    self.assertNotIn("final_state_checks", serialized)

    def test_checkpoint_survives_reopening_and_requires_exact_consent(self):
        llm = ScriptedLLM()
        path = str(self.root / "checkpoints.sqlite")
        config = graph_config("thread")
        before = self.bank.snapshot()
        with SqliteSaver.from_conn_string(path) as saver:
            graph = build_graph(self.bank, llm, saver)
            state = graph.invoke(initial_state("thread", "Transfer $25.00 from A001 to A002."), config)
            payload = state["__interrupt__"][0].value
            self.assertEqual(payload["source_account_id"], "A001")
            self.assertEqual(payload["destination_account_id"], "A002")
            self.assertEqual(payload["amount_cents"], 2500)
            self.assertEqual(self.bank.snapshot(), before)
        with SqliteSaver.from_conn_string(path) as saver:
            graph = build_graph(self.bank, llm, saver)
            self.assertTrue(graph.get_state(config).interrupts)
            state = graph.invoke(Command(resume="yes, perhaps"), config)
            self.assertTrue(state["__interrupt__"])
            self.assertEqual(self.bank.snapshot(), before)
            state = graph.invoke(Command(resume="CONFIRM"), config)
            self.assertEqual(state["result"]["status"], "completed")
            self.assertEqual(len(self.bank.snapshot()["transfers"]), 1)
            # Replaying a completed checkpoint cannot execute a second transfer.
            graph.invoke(None, config)
            self.assertEqual(len(self.bank.snapshot()["transfers"]), 1)

    def test_missing_information_pauses_without_any_account_tool(self):
        llm = ScriptedLLM()
        with SqliteSaver.from_conn_string(str(self.root / "checkpoint.sqlite")) as saver:
            graph = build_graph(self.bank, llm, saver)
            state = graph.invoke(initial_state("thread", "Transfer $25.00 from A001."), graph_config("thread"))
            self.assertEqual(state["__interrupt__"][0].value["missing_fields"], ["destination_account_id"])
            self.assertEqual([agent for agent, _ in llm.calls], ["Supervisor"])
            self.assertFalse(self.bank.snapshot()["transfers"])

    def test_cancel_clarification(self):
        with SqliteSaver.from_conn_string(str(self.root / "checkpoint.sqlite")) as saver:
            graph = build_graph(self.bank, ScriptedLLM(), saver)
            config = graph_config("thread")
            graph.invoke(initial_state("thread", "Transfer $25.00 from A001."), config)
            state = graph.invoke(Command(resume="CANCEL"), config)
            self.assertEqual(state["result"]["code"], "request_cancelled")
            self.assertFalse(self.bank.snapshot()["transfers"])

    def test_second_request_clears_previous_result_and_consent(self):
        with SqliteSaver.from_conn_string(str(self.root / "checkpoint.sqlite")) as saver:
            graph = build_graph(self.bank, ScriptedLLM(), saver)
            config = graph_config("thread")
            graph.invoke(initial_state("thread", "Transfer $25.00 from A001 to A002."), config)
            graph.invoke(Command(resume="CONFIRM"), config)
            state = graph.invoke(initial_state("thread", "Transfer $25.00 from A001 to A002."), config)
            self.assertTrue(state["__interrupt__"])
            self.assertFalse(state["confirmed"])
            self.assertEqual(state["result"], {})
            self.assertEqual(len(self.bank.snapshot()["transfers"]), 1)

    def test_authentication_and_ownership(self):
        self.assertEqual(self.bank.balance("unknown", "A001")["code"], "authentication_required")
        self.assertEqual(self.bank.transactions("unknown", "A001")["code"], "authentication_required")
        self.assertEqual(self.bank.balance("thread", "A003")["code"], "unauthorized")
        self.assertEqual(self.bank.transactions("thread", "A003")["code"], "unauthorized")
        self.assertEqual(self.bank.balance("thread", "A999")["code"], "account_not_found")
        self.assertEqual(self.bank.authenticate("thread", "C001", "wrong")["code"], "authentication_failed")
        self.assertEqual(self.bank.balance("thread", "A001")["code"], "authentication_required")

    def test_checkpoint_session_cannot_be_reauthenticated_as_another_customer(self):
        result = self.bank.authenticate("thread", "C002", "2202")
        self.assertEqual(result["code"], "session_owner_mismatch")
        self.assertEqual(self.bank.balance("thread", "A001")["code"], "authentication_required")
        self.assertEqual(self.bank.authenticate("thread", "C001", "1101")["status"], "ok")

    def test_interactive_cli_pause_and_resume(self):
        # Exercise the CLI loop itself, including process-independent checkpoints.
        with patch("builtins.input", side_effect=["C001", "Transfer $25.00 from A001 to A002.", "/exit"]), \
             patch("main.getpass.getpass", return_value="1101"), \
             patch("sys.stdout", new_callable=io.StringIO) as output:
            interactive(ScriptedLLM(), self.root, None)
            self.assertIn("Transfer $25.00 USD from A001 to A002", output.getvalue())
        session = next((self.root / "sessions").iterdir())
        bank = BankTools(session / "bank.sqlite")
        self.assertFalse(bank.snapshot()["transfers"])
        with patch("builtins.input", side_effect=["C001", "CONFIRM", "/exit"]), \
             patch("main.getpass.getpass", return_value="1101"), \
             patch("sys.stdout", new_callable=io.StringIO) as output:
            interactive(ScriptedLLM(), self.root, session.name)
            self.assertIn("transfer_completed", output.getvalue())
        self.assertEqual(len(bank.snapshot()["transfers"]), 1)

    def test_execution_requires_confirmation(self):
        before = self.bank.snapshot()
        result = self.bank.execute_transfer("thread", "op", self.request)
        self.assertEqual(result["code"], "confirmation_required")
        self.assertEqual(self.bank.snapshot(), before)
        self.assertEqual(self.bank.confirm("thread", "op", self.request, "yes")["code"], "confirmation_required")

    def test_confirmation_is_bound_to_amount_and_accounts(self):
        self.bank.confirm("thread", "op", self.request, "CONFIRM")
        for change in ({"amount_cents": 2600}, {"destination_account_id": "A003"}):
            result = self.bank.execute_transfer("thread", "op", {**self.request, **change})
            self.assertEqual(result["code"], "confirmation_required")
        self.assertFalse(self.bank.snapshot()["transfers"])

    def test_positive_integer_amounts_and_distinct_accounts(self):
        for amount in (0, -1, 1.5, True):
            with self.subTest(amount=amount):
                result = self.confirm_and_execute(str(amount), {**self.request, "amount_cents": amount})
                self.assertEqual(result["code"], "invalid_amount")
        result = self.confirm_and_execute("same", {**self.request, "destination_account_id": "A001"})
        self.assertEqual(result["code"], "same_account")
        self.assertFalse(self.bank.snapshot()["transfers"])

    def test_execute_rechecks_ownership_and_existence(self):
        for account_id, code in (("A003", "unauthorized"), ("A999", "account_not_found")):
            result = self.confirm_and_execute(account_id, {**self.request, "destination_account_id": account_id})
            self.assertEqual(result["code"], code)

    def test_execute_rechecks_policy_after_confirmation(self):
        self.assertEqual(self.bank.assess_transfer("thread", self.request)["status"], "eligible")
        self.bank.confirm("thread", "op", self.request, "CONFIRM")
        # An ordinary competing operation has spent funds since the policy read.
        with self.bank.connect() as db:
            state = self.bank._load(db)
            state["accounts"][0]["balance_cents"] = 1000
            db.execute("UPDATE bank SET data = ? WHERE id = 1", (json.dumps(state),))
        result = self.bank.execute_transfer("thread", "op", self.request)
        self.assertEqual(result["code"], "insufficient_funds")
        self.assertFalse(self.bank.snapshot()["transfers"])

    def test_execute_rejects_frozen_source_and_closed_destination(self):
        for index, status in ((0, "frozen"), (1, "closed")):
            self.bank.reset()
            self.bank.authenticate("thread", "C001", "1101")
            with self.bank.connect() as db:
                state = self.bank._load(db)
                state["accounts"][index]["status"] = status
                db.execute("UPDATE bank SET data = ? WHERE id = 1", (json.dumps(state),))
            self.assertEqual(self.confirm_and_execute()["code"], "inactive_account")

    def test_per_transfer_and_daily_limits(self):
        self.assertEqual(self.confirm_and_execute("large", {**self.request, "amount_cents": 100001})["code"],
                         "transfer_limit_exceeded")
        for index in range(2):
            self.assertEqual(self.confirm_and_execute(f"allowed-{index}", {**self.request, "amount_cents": 100000})["status"],
                             "completed")
        self.assertEqual(self.confirm_and_execute("daily", {**self.request, "amount_cents": 1})["code"],
                         "daily_limit_exceeded")

    def test_duplicate_execution_is_idempotent_including_concurrent_calls(self):
        self.bank.confirm("thread", "op", self.request, "CONFIRM")
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: self.bank.execute_transfer("thread", "op", self.request), range(2)))
        self.assertEqual(results[0], results[1])
        state = self.bank.snapshot()
        self.assertEqual(len(state["transfers"]), 1)
        self.assertEqual(len(state["transactions"]), 22)
        self.assertEqual(state["accounts"][0]["balance_cents"], 247500)
        self.assertEqual(state["accounts"][1]["balance_cents"], 102500)

    def test_reused_operation_id_cannot_change_the_transfer(self):
        self.confirm_and_execute()
        changed = {**self.request, "amount_cents": 5000}
        self.bank.confirm("thread", "op", changed, "CONFIRM")
        self.assertEqual(self.bank.execute_transfer("thread", "op", changed)["code"], "idempotency_conflict")
        self.assertEqual(len(self.bank.snapshot()["transfers"]), 1)

    def test_scenario_reset_is_reproducible_and_preserves_fixture_files(self):
        original = {p: p.read_bytes() for p in DATA_DIR.glob("*.json")}
        self.confirm_and_execute()
        self.bank.reset("low_funds")
        first = self.bank.snapshot()
        self.assertEqual(first["accounts"][0]["balance_cents"], 1000)
        self.bank.reset("low_funds")
        self.assertEqual(self.bank.snapshot(), first)
        self.assertEqual({p: p.read_bytes() for p in DATA_DIR.glob("*.json")}, original)
        self.assertEqual(self.bank.balance("thread", "A001")["code"], "authentication_required")

    def test_tool_results_drive_supervisor_final_prompt(self):
        task = read_json(DATA_DIR / "tasks.json")[0]
        llm = ScriptedLLM()
        report = run_task(task, llm, self.root)
        self.assertEqual(llm.calls[-1], ("SupervisorFinal", {"tool_result": report["result"]}))
        for prompt in (prompts.SUPERVISOR, prompts.ACCOUNT, prompts.POLICY, prompts.TRANSACTION, prompts.FINAL):
            self.assertNotIn("T01", prompt)
            self.assertNotIn("expected_outcome", prompt)

    def test_model_cannot_invent_missing_values(self):
        result = explicit_request(self.request, ["Transfer money from A001."])
        self.assertIsNone(result["destination_account_id"])
        self.assertIsNone(result["amount_cents"])
        self.assertEqual(result["source_account_id"], "A001")
        self.assertEqual(explicit_request(self.request, ["Transfer 2500 cents from A001 to A002."])["amount_cents"], 2500)

    def test_outcome_checks_fail_for_incorrect_balances_or_result(self):
        task = read_json(DATA_DIR / "tasks.json")[0]
        before = self.bank.snapshot()
        after = self.bank.snapshot()
        after["accounts"][1]["balance_cents"] += 1
        errors = check_task(task, before, after, {"result": {"status": "completed", "code": "wrong"}})
        self.assertGreaterEqual(len(errors), 3)

    def test_trace_input_filter_excludes_credentials(self):
        self.assertEqual(safe_inputs({"self": self.bank, "thread_id": "t", "customer_id": "C001",
                                      "pin": "1101", "api_key": "secret"}),
                         {"thread_id": "t", "customer_id": "C001"})

    def test_real_llm_adapter_uses_structured_provider_api(self):
        # Test SDK wiring without network calls; this is NOT a live-model test.
        from src.nodes import StructuredLLM
        from src.objects import AccountDecision
        with patch("src.nodes.OpenAI") as client_factory, patch("src.nodes.wrap_openai", side_effect=lambda c: c):
            parsed = AccountDecision(action="balance")
            message = client_factory.return_value.chat.completions.parse.return_value.choices[0].message
            message.refusal = None
            message.parsed = parsed
            llm = StructuredLLM("configured-model")
            self.assertEqual(llm.decide("AccountAgent", prompts.ACCOUNT, {"request": {}}, AccountDecision), parsed)
            kwargs = client_factory.return_value.chat.completions.parse.call_args.kwargs
            self.assertEqual(kwargs["model"], "configured-model")
            self.assertIs(kwargs["response_format"], AccountDecision)


if __name__ == "__main__":
    unittest.main()
