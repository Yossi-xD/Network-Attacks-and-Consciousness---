import secrets

from flask import Blueprint, make_response, render_template, request, session

from app.auth import login_required
from app.database import get_db


xss_bp = Blueprint("xss", __name__, url_prefix="/modules/xss")

# A 1x1 transparent GIF used as a beacon response for the cookie collector,
# so a stolen-cookie payload can use a plain <img> tag with no CORS issues.
TRANSPARENT_PIXEL = bytes.fromhex(
    "47494638396101000100800000000000ffffff21f90401000000002c00000000010001000002024401003b"
)


def _fetch_comments():
    rows = get_db().execute(
        """
        SELECT comments.id, comments.content, comments.created_at,
               COALESCE(users.username, 'Guest') AS author
        FROM comments
        LEFT JOIN users ON users.id = comments.user_id
        ORDER BY comments.id DESC
        """
    ).fetchall()
    return rows


def _post_comment():
    content = request.form.get("content", "")
    if content.strip():
        get_db().execute(
            "INSERT INTO comments (user_id, content) VALUES (?, ?)",
            (session.get("user_id"), content),
        )
        get_db().commit()


@xss_bp.route("/")
@login_required
def index():
    """Show the XSS lab overview linking to both rendering modes."""
    return render_template("modules/xss/index.html")


@xss_bp.route("/vulnerable-wall", methods=("GET", "POST"))
@login_required
def vulnerable_wall():
    """Post to the shared comment wall and render it WITHOUT escaping.

    WARNING: This intentionally renders stored comments with Jinja's `|safe`
    filter for local education only. Any HTML or <script> a visitor posts
    here executes in every other visitor's browser (stored/persistent XSS).
    """
    if request.method == "POST":
        _post_comment()

    response = make_response(
        render_template("modules/xss/vulnerable_wall.html", comments=_fetch_comments())
    )

    # Simulate an app that exposes a session-like token to client-side script
    # by leaving HttpOnly off. Never do this for a real auth cookie - Flask's
    # real session cookie stays HttpOnly the whole time.
    if not request.cookies.get("lab_session_token"):
        response.set_cookie(
            "lab_session_token",
            secrets.token_hex(16),
            httponly=False,
            samesite="Lax",
        )

    return response


@xss_bp.route("/secure-wall", methods=("GET", "POST"))
@login_required
def secure_wall():
    """Post to the same comment wall but render it with normal auto-escaping.

    Jinja escapes `{{ comment.content }}` by default, so identical stored
    payloads are displayed as inert text instead of executing. A
    Content-Security-Policy header is added as a second, independent layer:
    even if some future change to this page reintroduced an unescaped
    output, a strict CSP would still stop an injected <script> from running.
    """
    if request.method == "POST":
        _post_comment()

    response = make_response(
        render_template("modules/xss/secure_wall.html", comments=_fetch_comments())
    )
    response.headers["Content-Security-Policy"] = "script-src 'self'; object-src 'none'"
    return response


@xss_bp.route("/collect")
def collect():
    """Act as the attacker's cookie-theft collector endpoint.

    A payload posted on the vulnerable wall (e.g. an <img> tag pointing here
    with document.cookie appended) beacons stolen values to this route, which
    logs them and returns a harmless 1x1 pixel so the <img> renders normally.
    """
    stolen_value = request.args.get("value", "")
    if stolen_value:
        get_db().execute(
            "INSERT INTO xss_stolen_cookies (stolen_value, source_ip) VALUES (?, ?)",
            (stolen_value, request.remote_addr),
        )
        get_db().commit()

    response = make_response(TRANSPARENT_PIXEL)
    response.headers["Content-Type"] = "image/gif"
    return response


@xss_bp.route("/stolen")
@login_required
def stolen():
    """Show the attacker's-eye view of everything collect() has captured."""
    rows = get_db().execute(
        "SELECT stolen_value, source_ip, created_at FROM xss_stolen_cookies ORDER BY id DESC"
    ).fetchall()
    return render_template("modules/xss/stolen.html", stolen_items=rows)
