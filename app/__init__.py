from flask import Flask

from app.auth import auth_bp
from app.csrf import csrf_bp
from app.database import init_db
from app.detection import detection_bp, register_detection
from app.modules import modules_bp
from app.portal import portal_bp
from app.ssrf import internal_bp, ssrf_bp
from app.xss import xss_bp
from config import Config
from app.routes import main_bp


def create_app(config_class=Config):
    """Create and configure the Flask application.

    The application factory pattern keeps setup code in one place and makes the
    project easier to expand later with tests, Blueprints, and configuration.
    """
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )
    app.config.from_object(config_class)

    # Create the local SQLite database and required tables on startup.
    init_db(app)

    # Register the main Blueprint that contains the current public routes.
    app.register_blueprint(main_bp)

    # Register authentication routes separately from the public homepage.
    app.register_blueprint(auth_bp)

    # Register employee portal pages used by signed-in users.
    app.register_blueprint(portal_bp)

    # Keep existing module routes registered for future controlled testing.
    app.register_blueprint(modules_bp)

    # Attack lab blueprints - each pairs a vulnerable implementation with a
    # secure one under /modules/<attack-name>.
    app.register_blueprint(xss_bp)
    app.register_blueprint(csrf_bp)
    app.register_blueprint(ssrf_bp)
    app.register_blueprint(internal_bp)

    # Statistics-based threat detection: dashboard blueprint plus the
    # before_request hook that logs and flags every incoming request.
    app.register_blueprint(detection_bp)
    register_detection(app)

    return app
