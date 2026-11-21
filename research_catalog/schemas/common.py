"""Shared JSON-schema fragments (kept in Python so they can be composed and unit tested)."""

NULLABLE_STR = {"type": ["string", "null"]}
NULLABLE_INT = {"type": ["integer", "null"]}
NULLABLE_URL = {"type": ["string", "null"], "pattern": r"^https?://\S+$"}
STR_LIST = {"type": "array", "items": {"type": "string"}}
NULLABLE_STR_LIST = {"type": ["array", "null"], "items": {"type": "string"}}
ANY = {}

VERIFICATION_STATUS = {
    "type": "string",
    "enum": [
        "verified",            # checked by a human against the authoritative page
        "verified_automated",  # reachable + metadata matched by scripts/verify_research_links.py
        "partially_verified",  # some fields verified, others pending
        "unverified",          # not yet checked
        "manual_required",     # site blocks automated checks; needs human confirmation
    ],
}
