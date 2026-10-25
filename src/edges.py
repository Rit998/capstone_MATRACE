"""LangGraph construction and small, deterministic routing functions."""

from langgraph.graph import END, START, StateGraph

from src.nodes import make_nodes
from src.states import BankingState


def after_supervisor(state: BankingState):
    if state.get("result"):
        return "SupervisorFinal"
    return "Clarification" if state.get("missing_fields") else "AccountAgent"


def after_clarification(state: BankingState):
    return "SupervisorFinal" if state.get("result") else "Supervisor"


def after_account(state: BankingState):
    return "SupervisorFinal" if state.get("result") else "PolicyAgent"


def after_policy(state: BankingState):
    return "SupervisorFinal" if state.get("result") else "Confirmation"


def after_confirmation(state: BankingState):
    return "SupervisorFinal" if state.get("result") else "TransactionAgent"


def build_graph(bank, llm, checkpointer):
    graph = StateGraph(BankingState)
    for name, node in make_nodes(bank, llm).items():
        graph.add_node(name, node)
    graph.add_edge(START, "Supervisor")
    graph.add_conditional_edges("Supervisor", after_supervisor, ["SupervisorFinal", "Clarification", "AccountAgent"])
    graph.add_conditional_edges("Clarification", after_clarification, ["SupervisorFinal", "Supervisor"])
    graph.add_conditional_edges("AccountAgent", after_account, ["SupervisorFinal", "PolicyAgent"])
    graph.add_conditional_edges("PolicyAgent", after_policy, ["SupervisorFinal", "Confirmation"])
    graph.add_conditional_edges("Confirmation", after_confirmation, ["SupervisorFinal", "TransactionAgent"])
    graph.add_edge("TransactionAgent", "SupervisorFinal")
    graph.add_edge("SupervisorFinal", END)
    return graph.compile(checkpointer=checkpointer)
