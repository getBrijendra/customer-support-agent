"""LangGraph wiring for the chat harness. Built and compiled once at startup, run in-process.

load_context -> [route || retrieve_facts] -> after_retrieval
    -> (no_match) fallback_sections -> prefetch
    -> prefetch -> tool_agent <-> tools (bounded) -> generate -> validate_and_gate -> persist
Any node error short-circuits to persist (the chat is stored with status=error).
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.harness.nodes import HarnessNodes
from app.harness.state import GuidanceState


def _ok_or_persist(next_node: str) -> Any:
    def route(state: GuidanceState) -> str:
        return "persist" if state.get("errors") else next_node

    return route


async def _join(_: GuidanceState) -> dict[str, Any]:
    return {}


def build_graph(nodes: HarnessNodes) -> Any:
    g: Any = StateGraph(GuidanceState)
    g.add_node("load_context", nodes.load_context)
    g.add_node("route", nodes.route)
    g.add_node("retrieve_facts", nodes.retrieve_facts)
    g.add_node("after_retrieval", _join)
    g.add_node("fallback_sections", nodes.fallback_sections)
    g.add_node("prefetch", nodes.prefetch)
    g.add_node("tool_agent", nodes.tool_agent)
    g.add_node("tools", nodes.tools)
    g.add_node("generate", nodes.generate)
    g.add_node("validate_and_gate", nodes.validate_and_gate)
    g.add_node("persist", nodes.persist)

    g.add_edge(START, "load_context")
    # fan out (both branches skip if load_context failed), then join
    g.add_edge("load_context", "route")
    g.add_edge("load_context", "retrieve_facts")
    g.add_edge(["route", "retrieve_facts"], "after_retrieval")

    def after_retrieval(state: GuidanceState) -> str:
        if state.get("errors"):
            return "persist"
        return "fallback_sections" if state.get("no_match") else "prefetch"

    g.add_conditional_edges(
        "after_retrieval", after_retrieval, ["persist", "fallback_sections", "prefetch"]
    )
    g.add_conditional_edges(
        "fallback_sections", _ok_or_persist("prefetch"), ["persist", "prefetch"]
    )
    g.add_conditional_edges("prefetch", _ok_or_persist("tool_agent"), ["persist", "tool_agent"])

    def after_agent(state: GuidanceState) -> str:
        if state.get("errors"):
            return "persist"
        return "tools" if nodes.tool_loop_should_continue(state) else "generate"

    g.add_conditional_edges("tool_agent", after_agent, ["persist", "tools", "generate"])
    g.add_edge("tools", "tool_agent")
    g.add_conditional_edges(
        "generate", _ok_or_persist("validate_and_gate"), ["persist", "validate_and_gate"]
    )
    g.add_edge("validate_and_gate", "persist")
    g.add_edge("persist", END)
    return g.compile()


def recursion_limit(agent_max_steps: int) -> int:
    """Fixed nodes (~10 supersteps) + 2 per tool-loop iteration, with headroom."""
    return 16 + 2 * agent_max_steps
