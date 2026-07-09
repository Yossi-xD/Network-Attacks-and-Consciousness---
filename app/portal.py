from flask import Blueprint, abort, flash, redirect, render_template, request, session, url_for

from app.auth import login_required
from app.database import get_db


portal_bp = Blueprint("portal", __name__, url_prefix="/portal")


MESSAGES = [
    {
        "sender": "People Team",
        "subject": "Benefits enrollment reminder",
        "preview": "Please review your benefit selections before the end of the month.",
        "date": "Today",
        "unread": True,
    },
    {
        "sender": "IT Support",
        "subject": "Laptop refresh schedule",
        "preview": "Your equipment refresh window is available for booking.",
        "date": "Yesterday",
        "unread": True,
    },
    {
        "sender": "Finance",
        "subject": "Expense report approved",
        "preview": "Your June travel expense report has been approved.",
        "date": "Mon",
        "unread": False,
    },
    {
        "sender": "Office Operations",
        "subject": "Visitor registration update",
        "preview": "Front desk check-in steps have been simplified for registered guests.",
        "date": "Fri",
        "unread": False,
    },
]


def _employee_profile(employee):
    """Add display-only profile details without changing the database schema."""
    return {
        **dict(employee),
        "initials": "".join(part[0] for part in employee["full_name"].split()[:2]).upper(),
        "phone": "+972-3-555-01{:02d}".format(employee["id"]),
        "manager": "Maya Cohen" if employee["department"] != "Human Resources" else "Dana Shalev",
        "office": f"{employee['location']} HQ",
        "bio": (
            f"{employee['full_name']} supports the {employee['department']} team as "
            f"{employee['job_title']}, helping keep employee operations coordinated."
        ),
        "activity": [
            "Updated profile information",
            "Reviewed department announcements",
            "Acknowledged company policy update",
        ],
    }


@portal_bp.route("/employees")
@login_required
def employees():
    """Show the searchable employee directory."""
    search_term = request.args.get("q", "").strip()
    database = get_db()

    if search_term:
        like_pattern = f"%{search_term}%"

        # Keep employee search safe by passing all user input as query
        # parameters. SQLite treats these values as data, which prevents SQL
        # injection through the directory search box.
        employee_rows = database.execute(
            """
            SELECT id, full_name, email, department, job_title, location, status
            FROM employees
            WHERE full_name LIKE ?
               OR department LIKE ?
               OR job_title LIKE ?
               OR email LIKE ?
            ORDER BY full_name
            """,
            (like_pattern, like_pattern, like_pattern, like_pattern),
        ).fetchall()
    else:
        employee_rows = database.execute(
            """
            SELECT id, full_name, email, department, job_title, location, status
            FROM employees
            ORDER BY full_name
            """
        ).fetchall()

    return render_template(
        "portal/employees.html",
        employees=employee_rows,
        search_term=search_term,
    )


@portal_bp.route("/employees/<int:employee_id>")
@login_required
def employee_detail(employee_id):
    """Show a detailed employee profile using existing directory data."""
    employee = get_db().execute(
        """
        SELECT id, full_name, email, department, job_title, location, status
        FROM employees
        WHERE id = ?
        """,
        (employee_id,),
    ).fetchone()

    if employee is None:
        abort(404)

    return render_template(
        "portal/employee_detail.html",
        employee=_employee_profile(employee),
    )


@portal_bp.route("/developer/employee-search")
@login_required
def developer_employee_search():
    """Run a local-only employee search with intentionally unsafe SQL.

    WARNING: This route exists only for controlled educational comparison in a
    local environment. It intentionally formats user input directly into SQL and
    must never be exposed as a production feature.
    """
    search_term = request.args.get("q", "").strip()
    database = get_db()
    generated_query = None

    if search_term:
        generated_query = f"""
            SELECT id, full_name, email, department, job_title, location, status
            FROM employees
            WHERE full_name LIKE '%{search_term}%'
               OR department LIKE '%{search_term}%'
               OR job_title LIKE '%{search_term}%'
               OR email LIKE '%{search_term}%'
            ORDER BY full_name
        """
        employee_rows = database.execute(generated_query).fetchall()
    else:
        generated_query = """
            SELECT id, full_name, email, department, job_title, location, status
            FROM employees
            ORDER BY full_name
        """
        employee_rows = database.execute(generated_query).fetchall()

    return render_template(
        "portal/developer_employee_search.html",
        employees=employee_rows,
        search_term=search_term,
        generated_query=generated_query,
    )


@portal_bp.route("/announcements")
@login_required
def announcements():
    """Show company announcement cards from the database."""
    database = get_db()
    announcement_rows = database.execute(
        """
        SELECT id, title, body, author, created_at
        FROM announcements
        ORDER BY created_at DESC, id DESC
        """
    ).fetchall()

    return render_template(
        "portal/announcements.html",
        announcements=announcement_rows,
    )


@portal_bp.route("/announcements/create", methods=("POST",))
@login_required
def create_announcement():
    """Create a new company announcement.

    Announcement content is stored as plain text and rendered through normal
    Jinja variables, which are escaped by default. A future milestone can use
    this same data to compare unsafe rendering with secure escaping.
    """
    title = request.form.get("title", "").strip()
    body = request.form.get("body", "").strip()
    author = session.get("username", "Employee")

    if not title or not body:
        flash("Title and message are required to publish an announcement.")
        return redirect(url_for("portal.announcements"))

    database = get_db()
    database.execute(
        """
        INSERT INTO announcements (title, body, author)
        VALUES (?, ?, ?)
        """,
        (title, body, author),
    )
    database.commit()
    flash("Announcement published.")

    return redirect(url_for("portal.announcements"))


@portal_bp.route("/messages")
@login_required
def messages():
    """Show a simple inbox placeholder."""
    return render_template("portal/messages.html", messages=MESSAGES)


@portal_bp.route("/profile")
@login_required
def profile():
    """Show profile information for the signed-in user."""
    profile_data = {
        "username": session.get("username", "Employee"),
        "department": "Operations",
        "role": "Employee Portal User",
        "location": "Tel Aviv",
        "manager": "Maya Cohen",
        "email": "employee@company.local",
        "phone": "+972-3-555-0100",
    }
    return render_template("portal/profile.html", profile=profile_data)


@portal_bp.route("/settings")
@login_required
def settings():
    """Show account settings placeholders."""
    return render_template("portal/settings.html")
