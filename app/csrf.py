import secrets

from flask import Blueprint, flash, redirect, render_template, request, session, url_for

from app.auth import login_required
from app.database import get_db


csrf_bp = Blueprint("csrf", __name__, url_prefix="/modules/csrf")

ATTACKER_USERNAME = "attacker"


def _get_csrf_token():
    """Return this session's synchronizer token, generating one if needed."""
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_hex(32)
    return session["csrf_token"]


def _get_or_create_wallet(user_id):
    database = get_db()
    wallet = database.execute(
        "SELECT balance FROM wallets WHERE user_id = ?", (user_id,)
    ).fetchone()

    if wallet is None:
        database.execute("INSERT INTO wallets (user_id, balance) VALUES (?, 1000)", (user_id,))
        database.commit()
        return 1000

    return wallet["balance"]


def _transfer(sender_id, to_username, amount):
    database = get_db()
    recipient = database.execute(
        "SELECT id FROM users WHERE username = ?", (to_username,)
    ).fetchone()

    if recipient is None:
        return False, f"No user named '{to_username}' exists."
    if recipient["id"] == sender_id:
        return False, "You cannot send points to yourself."
    if amount <= 0:
        return False, "Amount must be a positive number of points."

    sender_balance = _get_or_create_wallet(sender_id)
    if amount > sender_balance:
        return False, "Insufficient balance for this transfer."

    _get_or_create_wallet(recipient["id"])

    database.execute(
        "UPDATE wallets SET balance = balance - ? WHERE user_id = ?", (amount, sender_id)
    )
    database.execute(
        "UPDATE wallets SET balance = balance + ? WHERE user_id = ?", (amount, recipient["id"])
    )
    database.commit()
    return True, f"Sent {amount} points to {to_username}."


@csrf_bp.route("/")
@login_required
def index():
    """Show the CSRF lab overview with the current wallet balance."""
    balance = _get_or_create_wallet(session["user_id"])
    return render_template("modules/csrf/index.html", balance=balance)


@csrf_bp.route("/vulnerable-transfer", methods=("GET", "POST"))
@login_required
def vulnerable_transfer():
    """Send points to another user with no CSRF token and no method restriction.

    WARNING: This intentionally accepts both GET and POST and checks nothing
    but the session cookie, for local education only. Because it also accepts
    GET, a forged request can run from a plain cross-site <img> tag - the
    session cookie rides along automatically, so the app cannot tell a real
    click from a forged one. See static/attacker/csrf_transfer_poc.html for a
    working forged request.
    """
    to_username = (request.values.get("to") or "").strip()
    amount_raw = request.values.get("amount", "")
    result = None

    if to_username and amount_raw:
        try:
            amount = int(amount_raw)
        except ValueError:
            result = (False, "Amount must be a whole number.")
        else:
            result = _transfer(session["user_id"], to_username, amount)

        if result and request.method == "POST":
            flash(result[1])
            return redirect(url_for("csrf.index"))

    balance = _get_or_create_wallet(session["user_id"])
    return render_template(
        "modules/csrf/vulnerable_transfer.html", balance=balance, result=result
    )


@csrf_bp.route("/secure-transfer", methods=("GET", "POST"))
@login_required
def secure_transfer():
    """Send points, but only with a valid per-session CSRF token.

    A forged cross-site request can still ride along with the session
    cookie, but it cannot know the token value (it is never exposed to
    another origin), so the transfer is rejected. Only POST is accepted -
    the vulnerable version's GET support is itself part of what made it
    exploitable with a plain link/image.
    """
    result = None

    if request.method == "POST":
        submitted_token = request.form.get("csrf_token", "")
        expected_token = session.get("csrf_token", "")

        if not expected_token or not secrets.compare_digest(submitted_token, expected_token):
            result = (False, "Request rejected: missing or invalid CSRF token.")
        else:
            to_username = (request.form.get("to") or "").strip()
            amount_raw = request.form.get("amount", "")
            try:
                amount = int(amount_raw)
            except ValueError:
                result = (False, "Amount must be a whole number.")
            else:
                result = _transfer(session["user_id"], to_username, amount)

            # Rotate the token after each use so a leaked/replayed token
            # (e.g. via a compromised historical request) stops working.
            session["csrf_token"] = secrets.token_hex(32)

    balance = _get_or_create_wallet(session["user_id"])
    return render_template(
        "modules/csrf/secure_transfer.html",
        balance=balance,
        result=result,
        csrf_token=_get_csrf_token(),
    )
