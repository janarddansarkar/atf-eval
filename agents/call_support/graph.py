"""Nimbus Telecom call-support agent, built with LangGraph.

    identify_customer -> classify_intent -> billing | outage | escalate | close | clarify
                                              outage --(no outage found)--> escalate

One conversation = one LangGraph thread (MemorySaver); each customer turn is one
`graph.invoke`. Per turn the graph starts at `identify_customer` while the caller
is unverified, otherwise at `classify_intent`. An LLM (Claude through
langchain-anthropic) classifies intent and writes the replies; with no
ANTHROPIC_API_KEY it falls back to keyword rules and canned replies so the
graph still runs offline.
"""
from __future__ import annotations

import os
import re
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from agents.call_support import tools
from atf_eval.llm_provider import load_dotenv

MODEL = os.environ.get("CALL_SUPPORT_MODEL", "claude-haiku-5-5")
ACCOUNT_RE = re.compile(r"ACC-\d{4}", re.IGNORECASE)


def _calls(old: list, new: list | None) -> list:
    """Reducer: a None update clears the per-turn tool-call log."""
    return [] if new is None else old + new


class State(TypedDict, total=False):
    # conversation-level (persisted by the checkpointer)
    authenticated: bool
    account_id: str | None
    customer_name: str | None
    postcode: str | None
    intent: str | None
    topic: str | None  # last substantive issue (billing / outage), used when escalating
    pending_issue: str | None  # what the caller said before they were verified
    ticket_id: str | None
    resolved: bool
    # per-turn (the caller resets these every turn)
    user_input: str
    response: str | None
    route: dict | None  # {"path": ..., "target": ...}
    outcome: dict | None
    outage: bool
    calls: Annotated[list, _calls]  # [{"node", "tool", "args", "result"}]


class Intent(BaseModel):
    intent: Literal["billing", "outage", "escalate", "goodbye", "other"]


CLASSIFY_SYSTEM = (
    "You are the intent classifier for a telecom call-support agent. Label the customer's "
    "latest message:\n"
    "- billing: question about a bill or charges\n"
    "- outage: internet/phone down, slow or not working\n"
    "- escalate: wants a human, a complaint, or disputes an answer\n"
    "- goodbye: says they're done / nothing else needed / thanks and bye\n"
    "- other: anything else"
)
REPLY_SYSTEM = (
    "You are a friendly, concise phone-support agent for Nimbus Telecom. Reply in 1-2 sentences, "
    "using only the facts you are given. Do not invent facts."
)

_KEYWORDS = [
    ("goodbye", ("that's all", "thats all", "nothing else", "bye", "no thanks", "that is all")),
    ("escalate", ("human", "dispute", "complain", "manager", "not right", "doesn't look right")),
    ("billing", ("bill", "charge", "invoice", "payment")),
    ("outage", ("internet", "down", "outage", "slow", "wifi", "broadband", "connection")),
]


def _keyword_intent(text: str) -> str:
    t = text.lower()
    return next((name for name, words in _KEYWORDS if any(w in t for w in words)), "other")


def build_graph(llm=None, use_llm: bool | None = None):
    """Compile the graph. `llm` may be any LangChain chat model (tests pass a
    fake); by default a ChatAnthropic is used when ANTHROPIC_API_KEY is set."""
    load_dotenv()  # picks ANTHROPIC_API_KEY up from a .env, as the atf-eval judges do
    if use_llm is None:
        use_llm = llm is not None or bool(os.environ.get("ANTHROPIC_API_KEY"))
    if use_llm and llm is None:
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model=MODEL, max_tokens=200)
    classifier = llm.with_structured_output(Intent) if use_llm else None

    def say(facts: str, fallback: str) -> str:
        if not use_llm:
            return fallback
        reply = llm.invoke([SystemMessage(REPLY_SYSTEM), HumanMessage(f"Facts: {facts}")])
        return reply.content if isinstance(reply.content, str) else fallback

    def call(node: str, tool, **args) -> tuple[dict, dict]:
        result = tool.invoke(args)
        return result, {"node": node, "tool": tool.name, "args": args, "result": result}

    # ---- nodes ----------------------------------------------------------
    def identify_customer(s: State) -> dict:
        m = ACCOUNT_RE.search(s["user_input"])
        if not m:
            return {
                "pending_issue": s.get("pending_issue") or s["user_input"],
                "response": say("The caller has not given an account id. Ask for it (format ACC-1234).",
                                "I can help with that. Could you give me your account number, like ACC-1234?"),
                "route": {"path": "ask_account", "target": "customer"},
            }
        acct, log = call("identify_customer", tools.lookup_account, account_id=m.group(0).upper())
        if not acct:
            return {
                "calls": [log],
                "response": say("No account matches that id. Ask the caller to check it.",
                                "I couldn't find that account. Could you double-check the number?"),
                "route": {"path": "ask_account", "target": "customer"},
            }
        return {
            "calls": [log],
            "authenticated": True,
            "account_id": acct["account_id"],
            "customer_name": acct["name"],
            "postcode": acct["postcode"],
        }

    def classify_intent(s: State) -> dict:
        text = s["user_input"]
        if s.get("pending_issue"):  # the caller was verified mid-issue: classify what they originally said
            text = f"{s['pending_issue']} {text}"
        if use_llm:
            intent = classifier.invoke([SystemMessage(CLASSIFY_SYSTEM), HumanMessage(text)]).intent
        else:
            intent = _keyword_intent(text)
        return {"intent": intent, "pending_issue": None}

    def billing_node(s: State) -> dict:
        bill, log = call("billing_node", tools.get_bill, account_id=s["account_id"])
        facts = (f"Customer {s['customer_name']}. Latest bill {bill['amount']:.2f}, usual {bill['usual']:.2f}; "
                 f"{bill['note']}.")
        return {
            "calls": [log],
            "topic": "billing",
            "response": say(facts, f"Your latest bill is {bill['amount']:.2f} ({bill['note']})."),
            "route": {"path": "billing", "target": "self_service"},
        }

    def outage_node(s: State) -> dict:
        res, log = call("outage_node", tools.check_outage, postcode=s["postcode"])
        update = {"calls": [log], "topic": "outage", "outage": res["outage"]}
        if res["outage"]:
            update["response"] = say(f"There is a known outage in the caller's area: {res['detail']}. Apologise.",
                                     f"There's a known outage in your area: {res['detail']}.")
            update["route"] = {"path": "outage", "target": "self_service"}
        return update

    def escalate_node(s: State) -> dict:
        category = s.get("topic") or "general"
        ticket, log = call("escalate_node", tools.create_ticket, account_id=s["account_id"], category=category)
        return {
            "calls": [log],
            "ticket_id": ticket["ticket_id"],
            "response": say(f"Ticket {ticket['ticket_id']} was opened for a human agent about {category}. "
                            "Tell the caller they will be contacted.",
                            f"I've opened ticket {ticket['ticket_id']}; a specialist will contact you."),
            "route": {"path": "escalate", "target": "human_agent"},
            "outcome": {"id": "ticket_created", "attributes": {"category": category}},
        }

    def close_node(s: State) -> dict:
        return {
            "resolved": True,
            "response": say("The issue is resolved and the caller is finished. Say goodbye.",
                            "Glad I could help. Goodbye!"),
            "route": {"path": "close", "target": "customer"},
            "outcome": {"id": "issue_resolved", "attributes": {"category": s.get("topic") or "general"}},
        }

    def clarify_node(s: State) -> dict:
        return {
            "response": say("The request was unclear. Ask what they need help with.",
                            "Sorry, could you tell me more about what you need help with?"),
            "route": {"path": "clarify", "target": "customer"},
        }

    # ---- edges ----------------------------------------------------------
    def entry(s: State) -> str:
        return "classify_intent" if s.get("authenticated") else "identify_customer"

    def after_identify(s: State) -> str:
        return "classify_intent" if s.get("authenticated") else END

    def after_classify(s: State) -> str:
        return {"billing": "billing_node", "outage": "outage_node", "escalate": "escalate_node",
                "goodbye": "close_node"}.get(s["intent"], "clarify_node")

    def after_outage(s: State) -> str:
        return END if s.get("outage") else "escalate_node"

    g = StateGraph(State)
    for fn in (identify_customer, classify_intent, billing_node, outage_node, escalate_node, close_node,
               clarify_node):
        g.add_node(fn.__name__, fn)
    g.add_conditional_edges(START, entry, ["identify_customer", "classify_intent"])
    g.add_conditional_edges("identify_customer", after_identify, ["classify_intent", END])
    g.add_conditional_edges("classify_intent", after_classify,
                            ["billing_node", "outage_node", "escalate_node", "close_node", "clarify_node"])
    g.add_conditional_edges("outage_node", after_outage, ["escalate_node", END])
    for n in ("billing_node", "escalate_node", "close_node", "clarify_node"):
        g.add_edge(n, END)
    return g.compile(checkpointer=MemorySaver())
