"""
Hygieia Backend API
Medical Diagnostic Platform - Backend Service
"""

from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os
from dotenv import load_dotenv

load_dotenv()

db = SQLAlchemy()
jwt = JWTManager()
migrate = Migrate()

def create_app(config_name=None):
    """Application factory pattern"""
    app = Flask(__name__)
    
    # Import and apply configuration
    from config import Config
    app.config.from_object(Config)
    
    # Initialize extensions
    db.init_app(app)
    jwt.init_app(app)
    migrate.init_app(app, db)
    
    # CORS configuration - allow local network origins
    frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:3000')
    CORS(app, resources={
        r"/api/*": {
            "origins": Config.ALLOWED_ORIGINS + [
                frontend_url,
                "http://localhost:3000",
                "http://localhost:5173",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:5173",
                r"http://192\.168\..*",  # Allow all 192.168.x.x IPs
                r"http://10\..*",  # Allow all 10.x.x.x IPs
            ],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "supports_credentials": True
        },
        r"/uploads/*": {
            "origins": Config.ALLOWED_ORIGINS + [
                frontend_url,
                "http://localhost:3000",
                "http://localhost:5173",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:5173",
                r"http://192\.168\..*",  # Allow all 192.168.x.x IPs
                r"http://10\..*",  # Allow all 10.x.x.x IPs
            ],
            "methods": ["GET"],
            "allow_headers": ["Content-Type"],
        }
    })
    
    # Register blueprints
    from app.routes.auth import auth_bp
    from app.routes.analysis import analysis_bp
    from app.routes.blockchain import blockchain_bp
    from app.routes.users import users_bp
    from app.routes.benchmark import benchmark_bp
    from app.routes.chat import chat_bp
    
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(analysis_bp, url_prefix='/api/analysis')
    app.register_blueprint(blockchain_bp, url_prefix='/api/blockchain')
    app.register_blueprint(users_bp, url_prefix='/admin/users')
    app.register_blueprint(benchmark_bp, url_prefix='/api/benchmark')
    app.register_blueprint(chat_bp, url_prefix='/api/chat')

    # Backwards-compatible API aliases: some clients call /api/users/*
    # The canonical admin routes live under /admin/users; provide thin wrappers
    # so /api/users/* behaves the same (keeps same authorization checks).
    from app.routes.users import (
        get_all_users, get_user, get_user_analyses,
        toggle_admin, toggle_active, delete_user, get_user_stats
    )

    @app.route('/api/users', methods=['GET'])
    def api_get_all_users():
        return get_all_users()

    @app.route('/api/users/<user_id>', methods=['GET'])
    def api_get_user(user_id):
        return get_user(user_id)

    @app.route('/api/users/<user_id>/analyses', methods=['GET'])
    def api_get_user_analyses(user_id):
        return get_user_analyses(user_id)

    @app.route('/api/users/<user_id>/toggle-admin', methods=['POST'])
    def api_toggle_admin(user_id):
        return toggle_admin(user_id)

    @app.route('/api/users/<user_id>/toggle-active', methods=['POST'])
    def api_toggle_active(user_id):
        return toggle_active(user_id)

    @app.route('/api/users/<user_id>', methods=['DELETE'])
    def api_delete_user(user_id):
        return delete_user(user_id)

    @app.route('/api/users/stats', methods=['GET'])
    def api_get_user_stats():
        return get_user_stats()
    
    # Serve uploaded files
    from flask import send_from_directory
    @app.route('/uploads/<path:filename>')
    def serve_upload(filename):
        """Serve uploaded files from the uploads folder (supports nested paths)"""
        try:
            return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
        except FileNotFoundError:
            return {'error': 'File not found'}, 404
    
    # Health check endpoint
    @app.route('/api/health')
    def health_check():
        return {'status': 'healthy', 'service': 'hygieia-api', 'version': '2.0.0'}
    
    return app
