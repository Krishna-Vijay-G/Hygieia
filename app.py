import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")

# Configure database
database_url = os.environ.get("DATABASE_URL")
if not database_url:
    # Fallback to SQLite for development
    database_url = "sqlite:///hygieia.db"
    print(f"Using default SQLite database: {database_url}")

app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Initialize the app with the extension
db.init_app(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Import routes after app creation
from routes import *  # noqa: F401, F403

# Create database tables
with app.app_context():
    # Import models to ensure they're registered
    from models import DiagnosticResult, ChatConversation, ChatMessage  # noqa: F401
    db.create_all()

# Flask application instance - run via gunicorn in production
