# MA-TRACE: stage-one banking MAS

A small **fictional banking simulator** built with Python 3.12, LangGraph,
Pydantic structured LLM outputs, and optional LangSmith tracing. It adapts the
application MAS from the retail proposal to banking. No real banking services,
frontend, failure attribution, fault injection, recovery, dependency analysis,
or research evaluation are implemented.

## Setup

```bash
cd /workspace/capstone_MATRACE  # Or your local checkout
uv sync --locked
cp .env.example .env
# Set OPENAI_API_KEY in .env or securely in your environment.
uv run --locked main.py
```

Alternatively, using Python 3.12:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

In this cloud environment the home cache is read-only: set
`export UV_CACHE_DIR=/workspace/.cache/uv` before using uv. Existing dependencies
are installed in `.venv`. `.env.example` deliberately leaves credentials blank.
Restricted cloud networking must allow `api.openai.com` and, for optional tracing,
`api.smith.langchain.com`. These additions are saved in the environment draft;
review/save them in environment settings and publish when ready. Draft saving
does not apply network changes to the current instance.
`OPENAI_MODEL` defaults to `gpt-4o-mini`; use a model supporting OpenAI's
Pydantic/JSON-schema structured outputs. `OPENAI_BASE_URL` is an optional endpoint
override; compatibility with third-party endpoints has not been tested.

## Try it

Log in as fictional customer **C001**, demo PIN **1101**. Their checking account
is **A001** ($2,500.00) and savings is **A002** ($1,000.00). Five customers and
their public demo PINs are listed in `data/customers.json`; account pairs are
A001/A002 through A009/A010. Never enter real credentials or customer data.

Example requests:

```text
What is the balance of account A001?
Show recent transactions for A001.
Transfer $25.00 from A001 to A002.
Transfer $25.00 from A001.
```

The final example asks for the missing destination. Supply exact account IDs;
account types alone are not resolved automatically. Use numeric amounts with
currency, such as `$25.00`, `25 dollars`, or `2500 cents`. The supervisor interprets
intent using the model, and Python rejects IDs/amounts absent from your messages.

Eligible transfers pause and display source, destination, amount, and currency.
Type **CONFIRM** to approve exactly that transfer, or **CANCEL** to decline.
Other responses do not grant consent. Transfers cannot be changed at the
confirmation prompt: cancel and make a new request instead.

Type `/exit` to quit, including while paused. Restart using the printed session ID:

```bash
uv run --locked main.py --resume SESSION_UUID
```

Log in again as the original customer. SQLite checkpoints retain the pending
request across processes. This is ordinary LangGraph pause/resume, not a research
recovery subsystem. Each new interactive session starts from clean fixtures;
multiple requests within a session share its persisted working bank. Use the
same `--work-dir` if you set a custom one.

## Agents and tools

```text
Supervisor -> [clarify if needed] -> Account Agent -> Supervisor final (reads)
                                             |
                                             -> Policy Agent -> confirmation
                                                                  |
                                                         Transaction Agent
                                                                  |
                                                          Supervisor final
```

The supervisor interprets messages, asks for missing information, and produces
the final response from tool results. Each specialist makes a real structured
LLM decision. The Account Agent checks the authenticated session and selects
balance, history, or transfer-ownership tools. Its dedicated Python login tool
checks demo credentials before the request enters the graph. PINs are not sent
to the LLM or stored in graph state. Read-only requests bypass both transfer agents.

Python enforces authentication, ownership, distinct active accounts, positive
integer cents, sufficient funds, a **$1,000 per-transfer limit**, and a **$2,000
total per customer per UTC day**. Only the transfer tool writes balances. It
rechecks conditions in a SQLite write transaction, requires confirmation bound
to the exact request, and uses an operation ledger to prevent duplicate execution
even on checkpoint replay or concurrent calls. The CLI also displays the
authoritative tool result alongside the supervisor's generated wording.

## Scenarios, tasks, and functional checks

`scenarios.json` contains reusable initial conditions and account overrides.
`tasks.json` contains concrete requests, a `scenario_id`, scripted follow-ups,
expected outcomes, and deterministic final-state assertions. Expected outcomes
and checks are used only **after** graph execution and never enter agent prompts.
Task runs reset a fresh working bank from their scenario, with independent
checkpoints. Task login uses the customer's fictional fixture PIN.

```bash
uv run --locked main.py --list-tasks  # No API key needed
uv run --locked main.py --task T03    # Requires API key; runs real LLMs
uv run --locked main.py --task all
uv run --locked python -m unittest discover -s tests -v  # Offline
```

| Task | Situation | Expected result |
| --- | --- | --- |
| T01 | Balance lookup | A001 balance returned |
| T02 | Recent history | Latest A001 transactions returned |
| T03 | Normal transfer | $25 moved once after confirmation |
| T04 | Insufficient funds | Rejected; balances unchanged |
| T05 | Invalid account | Rejected; no changes |
| T06 | Another customer's account | Access denied |
| T07 | Transfer limit exceeded | Rejected; no changes |
| T08 | Missing destination | Clarification, then confirmed transfer |
| T09 | Declined confirmation | Cancelled; no changes |
| T10 | Frozen destination | Rejected; no changes |

Assertions check exact result codes and relevant returned fields, all account
balances and metadata, unchanged customers/policy/history, transfer counts, new
transaction counts, matching transfer IDs, and conservation of money. CLI task
runs write `report.json` and exit nonzero on assertion failure.

**Verification during implementation:** 23 offline tests passed, including all
10 supplied task cases with explicitly scripted LLM test doubles. Additional
checks covered CLI pause/resume across reopened checkpoints, read-only routing,
authentication, ownership, exact consent, limits, atomic duplicate prevention,
fresh requests, and immutable fixtures. The real OpenAI structured-output adapter
was checked with a mocked SDK. These checks verify tools and graph wiring, not
live model quality. Live LLM task runs and LangSmith uploads remain untested
because credentials were deliberately left blank. Unauthenticated connectivity
checks also returned proxy 403 errors for both endpoints in the current instance.
Run `--task all` after adding your API key and enabling the saved endpoint access.
No offline agent fallback exists in the application.

## Tracing and files

Set `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, and optionally
`LANGSMITH_PROJECT` to trace named graph agents, structured model decisions,
and banking tools. Leave tracing false to run without LangSmith credentials.
API keys stay in client configuration. The authentication tool filters its PIN
argument from traces; checkpoints contain no login PINs. User requests, account
results, and model outputs may be traced, so use only the supplied fictional data.
Live trace delivery has not been verified. Tracing does not implement MA-TRACE's
later diagnosis or evaluation features.

`src/nodes.py` holds agents and simulated tools; `edges.py` constructs the graph;
`prompts.py` holds prompts; `objects.py` defines Pydantic models; `states.py`
defines shared state. Original JSON fixtures stay unchanged. Working banks,
checkpoints, and task reports live under ignored `.runtime/`; `.env` is ignored.
These are local demo files, not a hardened authentication or banking service.

The supplied AgentTether URL was attempted but returned a proxy access error.
This project was implemented independently and claims no compatibility with
AgentTether, τ-bench, or τ²-bench.
