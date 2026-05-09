"""Tools the ReAct classifier can call mid-reasoning."""


# Fake customer database — purely for learning purposes
SENDERS = {
    "alice@example.com":    {"is_existing_customer": True,  "tier": "enterprise", "open_tickets": 2},
    "bob@startup.io":       {"is_existing_customer": True,  "tier": "starter",    "open_tickets": 0},
    "carol@bigcorp.com":    {"is_existing_customer": True,  "tier": "enterprise", "open_tickets": 5},
    "spammer@nope.cn":      {"is_existing_customer": False, "tier": None,         "open_tickets": 0},
}


def lookup_sender(email: str) -> dict:
    """Look up basic info about an email sender from the (fake) database."""
    return SENDERS.get(email, {
        "is_existing_customer": False,
        "tier": None,
        "open_tickets": 0,
    })

