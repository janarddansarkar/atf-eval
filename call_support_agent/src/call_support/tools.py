"""The call-support agent's backend tools: a tiny in-memory telecom "CRM".

They are real LangChain tools (`@tool`), so the graph invokes them the way any
LangChain agent would; only the data behind them is fake.
"""
from __future__ import annotations

from langchain_core.tools import tool

ACCOUNTS = {
    "ACC-1001": {"name": "Alex Morgan", "postcode": "N1 4AB", "plan": "Fibre 500"},
    "ACC-1002": {"name": "Sam Rivera", "postcode": "M3 2CD", "plan": "Fibre 100"},
    "ACC-1003": {"name": "Priya Nair", "postcode": "S2 9XY", "plan": "Fibre 300"},
}
BILLS = {
    "ACC-1001": {"amount": 45.00, "usual": 45.00, "note": "no unusual charges"},
    "ACC-1002": {"amount": 89.50, "usual": 49.50, "note": "includes a one-off 40.00 roaming charge"},
    "ACC-1003": {"amount": 62.00, "usual": 62.00, "note": "no unusual charges"},
}
OUTAGE_POSTCODES = {"N1 4AB": "fibre cable fault, estimated fix by 6pm today"}


@tool
def lookup_account(account_id: str) -> dict:
    """Look up a customer account by id (e.g. ACC-1001). Returns {} if unknown."""
    acct = ACCOUNTS.get(account_id.upper())
    return {"account_id": account_id.upper(), **acct} if acct else {}


@tool
def get_bill(account_id: str) -> dict:
    """Get the latest bill for an account."""
    return BILLS.get(account_id, {})


@tool
def check_outage(postcode: str) -> dict:
    """Check for a known network outage in a postcode area."""
    detail = OUTAGE_POSTCODES.get(postcode)
    return {"outage": detail is not None, "detail": detail}


@tool
def create_ticket(account_id: str, category: str) -> dict:
    """Open a support ticket for a human agent to follow up on."""
    return {"ticket_id": f"TCK-{account_id[-4:]}-{category[:3].upper()}"}
