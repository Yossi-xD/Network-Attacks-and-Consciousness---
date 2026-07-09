from functools import wraps

from flask import Blueprint, flash, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from app.database import get_db


# Authentication routes live in their own Blueprint to keep the app modular.
auth_bp = Blueprint("auth", __name__, url_prefix="/auth")


def login_required(view):
    """Redirect anonymous users before they can access protected pages."""
    @wraps(view)
    def wrapped_view(**kwargs):
        if session.get("user_id") is None:
            flash("Please log in to view that page.")
            return redirect(url_for("auth.login"))

        return view(**kwargs)

    return wrapped_view


@auth_bp.route("/register", methods=("GET", "POST"))
def register():
    """Create a local user account with a hashed password."""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        error = None

        if not username:
            error = "Username is required."
        elif not password:
            error = "Password is required."
        elif password != confirm_password:
            error = "Passwords do not match."

        if error is None:
            database = get_db()
            existing_user = database.execute(
                "SELECT id FROM users WHERE username = ? OR email = ?",
                (username, email),
            ).fetchone()

            if existing_user is not None:
                error = "A user with that username or email already exists."
            else:
                database.execute(
                    """
                    INSERT INTO users (username, email, password_hash)
                    VALUES (?, ?, ?)
                    """,
                    (username, email or None, generate_password_hash(password)),
                )
                database.commit()
                flash("Registration successful. Please log in.")
                return redirect(url_for("auth.login"))

        flash(error)

    return render_template("auth/register.html")


@auth_bp.route("/login", methods=("GET", "POST"))
def login():
    """Authenticate a user and store their id in the Flask session."""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        database = get_db()
        error = None

        user = database.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?",
            (username,),
        ).fetchone()

        if user is None:
            error = "Invalid username or password."
        elif not check_password_hash(user["password_hash"], password):
            error = "Invalid username or password."

        if error is not None:
            # Imported lazily to avoid a circular import: detection.py
            # imports login_required from this module at load time.
            from app.detection import record_alert

            record_alert("Failed login attempt")

        if error is None:
            session.clear()
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("auth.dashboard"))

        flash(error)

    return render_template("auth/login.html")


@auth_bp.route("/logout")
def logout():
    """Clear the current Flask session."""
    session.clear()
    flash("You have been logged out.")
    return redirect(url_for("main.index"))


@auth_bp.route("/dashboard")
@login_required
def dashboard():
    """Show a simple protected dashboard after login."""
    return render_template("auth/dashboard.html")
