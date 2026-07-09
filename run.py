from app import create_app


# Create the Flask application using the application factory.
# This file is the entry point used to run the project locally.
app = create_app()


if __name__ == "__main__":
    # host="0.0.0.0" makes the app reachable at this machine's LAN IP, not
    # just 127.0.0.1. That matters for the SSRF lab: /internal/secret-report
    # only allows callers whose address is localhost, so browsing to it
    # directly via the LAN IP is a genuinely different, blockable caller from
    # the Flask server fetching it from itself over 127.0.0.1. See the
    # step-by-step guide for how to use this.
    app.run(debug=True, host="0.0.0.0")
