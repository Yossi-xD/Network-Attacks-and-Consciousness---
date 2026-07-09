from app import create_app


# Create the Flask application using the application factory.
# This file is the entry point used to run the project locally.
app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
