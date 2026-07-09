import re

from flask import Blueprint, render_template, request

from app.auth import login_required
from app.database import get_db


detection_bp = Blueprint("detection", __name__, url_prefix="/modules/threat-detection")

# Statistics-based / signature-based detection: each entry pairs a regex
# tuned to a known attack shape with a human-readable reason shown on the
# dashboard. This intentionally mirrors the "Main Algorithms" slide from the
# project proposal (monitor request patterns, flag suspicious keywords).
SUSPICIOUS_PATTERNS = [
    (re.compile(r"(?:'|%27)\s*(or|and)\s+['\"]?\w+['\"]?\s*=\s*['\"]?\w+", re.IGNORECASE), "SQLi pattern: tautology (' OR 1=1)"),
    (re.compile(r"\bunion\b[\s\S]{0,60}\bselect\b", re.IGNORECASE), "SQLi pattern: UNION SELECT"),
    (re.compile(r"(--\s|#\s*(?:$|&))", re.IGNORECASE), "SQLi pattern: comment terminator"),
    (re.compile(r"<script[\s>]", re.IGNORECASE), "XSS pattern: <script> tag"),
    (re.compile(r"on(error|load|click|mouseover)\s*=", re.IGNORECASE), "XSS pattern: inline event handler"),
    (re.compile(r"\.\./"), "Path traversal pattern: ../"),
]

FAILED_LOGIN_REASON = "Failed login attempt"


def _collect_inspectable_text():
    parts = [request.full_path]
    if request.form:
        parts.extend(str(value) for value in request.form.values())
    return " ".join(parts)


def record_alert(reason):
    """Log one flagged request. Called from the hook below and from auth.py
    on failed logins (imported lazily there to avoid a circular import)."""
    database = get_db()
    database.execute(
        "INSERT INTO security_alerts (ip_address, method, path, reason) VALUES (?, ?, ?, ?)",
        (request.remote_addr, request.method, request.path, reason),
    )
    database.commit()


def register_detection(app):
    """Attach the statistics-based request logging/flagging hook to the app."""

    @app.before_request
    def _log_and_flag_request():
        if request.path.startswith("/static/"):
            return

        database = get_db()
        database.execute(
            "INSERT INTO request_logs (method, path, ip_address, user_agent) VALUES (?, ?, ?, ?)",
            (request.method, request.path, request.remote_addr, request.headers.get("User-Agent", "")),
        )
        database.commit()

        inspectable = _collect_inspectable_text()
        for pattern, reason in SUSPICIOUS_PATTERNS:
            if pattern.search(inspectable):
                record_alert(reason)
                break


@detection_bp.route("/")
@login_required
def index():
    """Show request volume, top requesters, failed-login clusters, and flags."""
    database = get_db()

    total_requests = database.execute("SELECT COUNT(*) FROM request_logs").fetchone()[0]

    top_ips = database.execute(
        """
        SELECT ip_address, COUNT(*) AS hits
        FROM request_logs
        GROUP BY ip_address
        ORDER BY hits DESC
        LIMIT 10
        """
    ).fetchall()

    failed_logins = database.execute(
        """
        SELECT ip_address, COUNT(*) AS attempts
        FROM security_alerts
        WHERE reason = ?
        GROUP BY ip_address
        ORDER BY attempts DESC
        LIMIT 10
        """,
        (FAILED_LOGIN_REASON,),
    ).fetchall()

    alerts = database.execute(
        """
        SELECT ip_address, method, path, reason, created_at
        FROM security_alerts
        ORDER BY id DESC
        LIMIT 50
        """
    ).fetchall()

    return render_template(
        "modules/threat_detection/index.html",
        total_requests=total_requests,
        top_ips=top_ips,
        failed_logins=failed_logins,
        alerts=alerts,
    )
