from flask import Blueprint, render_template


# A Blueprint groups public routes together.
main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    """Render the public Employee Portal landing page."""
    return render_template("index.html")
