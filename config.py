class Config:
    """Base configuration for the local educational Flask app."""

    # Keep a simple default configuration for now.
    # More settings can be added here as the project grows.
    DEBUG = True

    # Development-only secret key used by Flask sessions.
    # Replace this before using the app outside local learning.
    SECRET_KEY = "dev-only-change-this-secret-key"

    # SQLite database file used by the local educational platform.
    # The instance folder is ignored by Git, so local data stays out of commits.
    DATABASE_PATH = "instance/owasp_platform.sqlite3"
