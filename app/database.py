import os
import sqlite3
from flask import current_app, g
from werkzeug.security import generate_password_hash


SCHEMA_SQL = """
-- users stores local account records for the authentication lessons.
-- Passwords are stored as Werkzeug password hashes, never as plain text.
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    email TEXT UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- comments stores user-submitted text for future input-handling lessons.
-- No demo comments are inserted during database initialization.
CREATE TABLE IF NOT EXISTS comments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id)
);

-- request_logs stores basic HTTP request metadata for future monitoring lessons.
-- The application does not write request logs yet.
CREATE TABLE IF NOT EXISTS request_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    method TEXT NOT NULL,
    path TEXT NOT NULL,
    ip_address TEXT,
    user_agent TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- attack_results stores future educational scan or attack simulation outcomes.
-- No vulnerability checks or attacks are implemented yet.
CREATE TABLE IF NOT EXISTS attack_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    attack_name TEXT NOT NULL,
    target TEXT,
    result TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- employees stores the realistic employee directory used by the portal.
-- This data supports the normal searchable employee directory UI.
CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    department TEXT NOT NULL,
    job_title TEXT NOT NULL,
    location TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- announcements stores company updates submitted through the portal.
-- Content is rendered with Jinja's default escaping now; this feature can later
-- compare unsafe rendering against secure escaping in a controlled environment.
CREATE TABLE IF NOT EXISTS announcements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    author TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- xss_stolen_cookies stores whatever a malicious script beacons to the
-- collector endpoint in the XSS lab, simulating an attacker's loot page.
CREATE TABLE IF NOT EXISTS xss_stolen_cookies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stolen_value TEXT NOT NULL,
    source_ip TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- wallets stores a small play-money balance per user for the CSRF lab, so a
-- forged transfer request has a visible, demoable effect.
CREATE TABLE IF NOT EXISTS wallets (
    user_id INTEGER PRIMARY KEY,
    balance INTEGER NOT NULL DEFAULT 1000,
    FOREIGN KEY (user_id) REFERENCES users (id)
);

-- security_alerts stores requests flagged by the statistics-based threat
-- detection module (suspicious keywords, repeated failed logins, etc.).
CREATE TABLE IF NOT EXISTS security_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ip_address TEXT,
    method TEXT NOT NULL,
    path TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


SAMPLE_EMPLOYEES = [
    (
        "Maya Cohen",
        "maya.cohen@company.local",
        "Human Resources",
        "People Operations Manager",
        "Tel Aviv",
        "Active",
    ),
    (
        "Daniel Levi",
        "daniel.levi@company.local",
        "Engineering",
        "Backend Engineer",
        "Haifa",
        "Active",
    ),
    (
        "Ari Ben-David",
        "ari.bendavid@company.local",
        "Finance",
        "Finance Analyst",
        "Jerusalem",
        "Onboarding",
    ),
    (
        "Noa Friedman",
        "noa.friedman@company.local",
        "Product",
        "Product Designer",
        "Remote",
        "Active",
    ),
    (
        "Samir Haddad",
        "samir.haddad@company.local",
        "Security",
        "Security Operations Lead",
        "Tel Aviv",
        "Active",
    ),
    (
        "Lior Kaplan",
        "lior.kaplan@company.local",
        "Customer Success",
        "Enterprise Success Manager",
        "Remote",
        "Active",
    ),
]

SAMPLE_ANNOUNCEMENTS = [
    (
        "Quarterly Meeting",
        "All teams will join the quarterly business review to discuss goals, staffing plans, and delivery priorities.",
        "Leadership Team",
        "2026-07-08 09:00:00",
    ),
    (
        "Security Awareness Training",
        "A required security awareness session is scheduled for all employees and contractors.",
        "IT Operations",
        "2026-07-10 10:30:00",
    ),
    (
        "Office Maintenance",
        "The Tel Aviv office will have scheduled network and workspace maintenance on Friday evening.",
        "Facilities Team",
        "2026-07-12 16:00:00",
    ),
    (
        "Holiday Schedule",
        "The updated company holiday calendar is available for review in the employee portal.",
        "People Team",
        "2026-07-18 08:15:00",
    ),
]


def get_db():
    """Return one SQLite connection for the current request."""
    if "db" not in g:
        g.db = sqlite3.connect(current_app.config["DATABASE_PATH"])
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")

    return g.db


def close_db(error=None):
    """Close the request database connection if one was opened."""
    database = g.pop("db", None)

    if database is not None:
        database.close()


def _column_exists(connection, table_name, column_name):
    """Check whether a column exists before applying a small schema migration."""
    columns = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(column[1] == column_name for column in columns)


def init_db(app):
    """Create the SQLite database and tables if they do not already exist."""
    database_path = app.config["DATABASE_PATH"]
    if not os.path.isabs(database_path):
        project_root = os.path.abspath(os.path.join(app.root_path, os.pardir))
        database_path = os.path.join(project_root, database_path)
    app.config["DATABASE_PATH"] = database_path

    # Ensure the local instance directory exists before SQLite opens the file.
    os.makedirs(os.path.dirname(database_path), exist_ok=True)

    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(SCHEMA_SQL)

        # Existing local databases from the previous milestone need this column.
        if not _column_exists(connection, "users", "password_hash"):
            connection.execute(
                "ALTER TABLE users ADD COLUMN password_hash TEXT NOT NULL DEFAULT ''"
            )

        # Seed a standing "attacker" account so the CSRF lab always has a
        # valid transfer target, independent of who has registered so far.
        connection.execute(
            "INSERT OR IGNORE INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            ("attacker", "attacker@lab.local", generate_password_hash("attacker-lab-only")),
        )

        employee_count = connection.execute(
            "SELECT COUNT(*) FROM employees"
        ).fetchone()[0]
        if employee_count == 0:
            connection.executemany(
                """
                INSERT INTO employees
                    (full_name, email, department, job_title, location, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                SAMPLE_EMPLOYEES,
            )

        announcement_count = connection.execute(
            "SELECT COUNT(*) FROM announcements"
        ).fetchone()[0]
        if announcement_count == 0:
            connection.executemany(
                """
                INSERT INTO announcements (title, body, author, created_at)
                VALUES (?, ?, ?, ?)
                """,
                SAMPLE_ANNOUNCEMENTS,
            )

    app.teardown_appcontext(close_db)
