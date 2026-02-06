"""
Hygieia Backend Configuration
=============================
DuckDNS deployment configuration for free laptop hosting
"""

import os
import socket
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()


def get_local_ip():
    """Get the local IP address of the machine"""
    try:
        # Create a socket to get the local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # Connect to Google DNS
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        # Fallback to localhost if detection fails
        return "127.0.0.1"


class Config:
    """Base configuration class"""
    
    # =============================================================================
    # SERVER CONFIGURATION - LOCAL NETWORK ONLY
    # =============================================================================
    HOST = "0.0.0.0"  # Listen on all interfaces for local network access
    PORT = 5000
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    
    # Automatically detect local network IP, but prefer 192.168.x.x addresses
    LOCAL_IP = os.getenv('LOCAL_IP', get_local_ip())
    # If detected IP is not a 192.168.x.x address, try to find one
    if not LOCAL_IP.startswith('192.168.'):
        try:
            import subprocess
            result = subprocess.run(['ipconfig'], capture_output=True, text=True, shell=True)
            for line in result.stdout.split('\n'):
                if 'IPv4 Address' in line and '192.168.' in line:
                    LOCAL_IP = line.split(':')[1].strip()
                    break
        except:
            pass  # Keep the auto-detected IP if manual detection fails
    
    PUBLIC_API_BASE = f"http://{LOCAL_IP}:{PORT}"
    
    # =============================================================================
    # CORS ALLOWED ORIGINS - LOCAL NETWORK ONLY
    # =============================================================================
    ALLOWED_ORIGINS = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        f"http://{LOCAL_IP}:3000",  # Frontend on local network
        f"http://{LOCAL_IP}:5173",
        "*",  # Allow all origins for local network (development only)
    ]
    
    # =============================================================================
    # FLASK CONFIGURATION
    # =============================================================================
    SECRET_KEY = os.getenv("SECRET_KEY", "hygieia-secret-key-change-in-production")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "jwt-secret-key-change-in-production")
    JWT_ACCESS_TOKEN_EXPIRES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRES", 86400))  # 24 hours
    JWT_REFRESH_TOKEN_EXPIRES = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRES", 2592000))  # 30 days
    
    # =============================================================================
    # DATABASE CONFIGURATION
    # =============================================================================
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    # SQLAlchemy configuration
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_recycle': 300,
        'pool_pre_ping': True
    }
    
    # Set database URI
    if DATABASE_URL:
        # Handle Supabase PostgreSQL
        if DATABASE_URL.startswith('postgres://'):
            SQLALCHEMY_DATABASE_URI = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
        else:
            SQLALCHEMY_DATABASE_URI = DATABASE_URL
    else:
        # Fallback to SQLite for local development
        SQLALCHEMY_DATABASE_URI = 'sqlite:///hygieia.db'
    
    # =============================================================================
    # FILE UPLOAD CONFIGURATION
    # =============================================================================
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
    
    # Ensure upload directories exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(UPLOAD_FOLDER, 'avatars'), exist_ok=True)
    os.makedirs(os.path.join(UPLOAD_FOLDER, 'avatars_archive'), exist_ok=True)
    
    # =============================================================================
    # AI CHAT CONFIGURATION (Dr. Hygieia)
    # =============================================================================
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-exp')


# =============================================================================
# PRINT CONFIGURATION ON STARTUP
# =============================================================================
def print_config():
    print("\n" + "=" * 70)
    print("HYGIEIA BACKEND - LOCAL NETWORK")
    print("=" * 70)
    print(f"  Local Access:     http://localhost:{Config.PORT}")
    print(f"  Network Access:   http://{Config.LOCAL_IP}:{Config.PORT}")
    print(f"  API Endpoint:     {Config.PUBLIC_API_BASE}/api")
    print(f"  Debug Mode:       {Config.DEBUG}")
    print(f"  Database:         {'Configured' if Config.DATABASE_URL else 'SQLite (local)'}")
    print(f"  Upload Path:      {Config.UPLOAD_FOLDER}")
    print(f"  AI Chat (Dr. Hygieia): {'Google Gemini' if Config.GOOGLE_API_KEY else 'Disabled (set GOOGLE_API_KEY)'}")
    print(f"  Model:            {Config.GEMINI_MODEL}")
    print("=" * 70)
    print("\n  Access from this PC:      http://localhost:5000")
    print(f"  Access from network:      http://{Config.LOCAL_IP}:5000")
    print(f"  Frontend should use:      http://{Config.LOCAL_IP}:5000/api")
    print("=" * 70 + "\n")
