from flask import Blueprint, render_template, request
from werkzeug.security import check_password_hash

from app.auth import login_required
from app.database import get_db


modules_bp = Blueprint("modules", __name__, url_prefix="/modules")


# Source of truth for the Security Labs hub page. Each lab is a real,
# working route - nothing here is a "coming soon" placeholder.
LABS = [
    {
        "title": "SQL Injection",
        "summary": "Login bypass and unsafe search vs. parameterized queries.",
        "endpoint": "modules.sql_injection",
        "owner": "Khairy",
    },
    {
        "title": "Cross-Site Scripting (XSS)",
        "summary": "Stored XSS on a shared comment wall, including a cookie-theft demo.",
        "endpoint": "xss.index",
        "owner": "Yousef",
    },
    {
        "title": "Cross-Site Request Forgery (CSRF)",
        "summary": "A forged cross-site request moves points out of a victim's wallet.",
        "endpoint": "csrf.index",
        "owner": "Yousef",
    },
    {
        "title": "Server-Side Request Forgery (SSRF)",
        "summary": "The server fetches an attacker-chosen URL and reaches an internal-only route.",
        "endpoint": "ssrf.index",
        "owner": "Mohamed",
    },
    {
        "title": "Threat Detection",
        "summary": "Live request logging with statistics-based suspicious activity flags.",
        "endpoint": "detection.index",
        "owner": "Shared",
    },
    {
        "title": "Attack Automation",
        "summary": "Results recorded by the Python automation script that attacks every lab.",
        "endpoint": "modules.attack_automation",
        "owner": "Shared",
    },
]


@modules_bp.route("/")
@login_required
def index():
    """Show the central Security Labs hub linking every attack module."""
    return render_template("modules/index.html", labs=LABS)


@modules_bp.route("/sql-injection")
@login_required
def sql_injection():
    """Show the SQL Injection lab overview."""
    return render_template("modules/sql_injection/index.html")


@modules_bp.route("/sql-injection/vulnerable-login", methods=("GET", "POST"))
@login_required
def sql_injection_vulnerable_login():
    """Run an intentionally unsafe login-style SQL query for local education."""
    query = None
    result = None
    error = None

    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")

        # Intentionally unsafe: this string-formatted SQL is here only for the
        # local SQL Injection lab. Do not copy this pattern into real features.
        query = (
            "SELECT id, username, email FROM users "
            f"WHERE username = '{username}' AND password_hash = '{password}'"
        )

        try:
            rows = get_db().execute(query).fetchall()
            if rows:
                result = {
                    "success": True,
                    "message": f"The unsafe query returned {len(rows)} user record(s).",
                    "users": rows,
                }
            else:
                result = {
                    "success": False,
                    "message": "The unsafe query returned no users.",
                    "users": [],
                }
        except Exception as exc:
            error = str(exc)

    return render_template(
        "modules/sql_injection/vulnerable_login.html",
        query=query,
        result=result,
        error=error,
    )


@modules_bp.route("/sql-injection/secure-login", methods=("GET", "POST"))
@login_required
def sql_injection_secure_login():
    """Run a parameterized login query and verify the stored password hash."""
    query = None
    result = None

    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        query = (
            "SELECT id, username, email, password_hash FROM users "
            "WHERE username = ?"
        )

        user = get_db().execute(query, (username,)).fetchone()

        if user is not None and check_password_hash(user["password_hash"], password):
            result = {
                "success": True,
                "message": "The parameterized query found the user and the password hash matched.",
                "user": user,
            }
        else:
            result = {
                "success": False,
                "message": "No valid user was found with the provided username and password.",
                "user": None,
            }

    return render_template(
        "modules/sql_injection/secure_login.html",
        query=query,
        result=result,
    )


@modules_bp.route("/attack-automation")
@login_required
def attack_automation():
    """Show results recorded by attacks/automation.py's most recent run."""
    rows = get_db().execute(
        """
        SELECT attack_name, target, result, created_at
        FROM attack_results
        ORDER BY id DESC
        LIMIT 200
        """
    ).fetchall()
    return render_template("modules/attack_automation.html", results=rows)
