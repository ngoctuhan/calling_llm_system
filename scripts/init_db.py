#!/usr/bin/env python
import os
import sys
import logging
import subprocess
from pathlib import Path

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings
from app.db.session import engine, Base
from app.db.models import User
from app.security.password import get_password_hash
from sqlalchemy.orm import Session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_db() -> None:
    """Initialize the database with tables and a superuser"""
    try:
        # Create tables
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created.")
        
        # Create initial superuser if none exists
        with Session(engine) as db:
            # Check if superuser exists
            user = db.query(User).filter(User.is_superuser == True).first()
            if not user:
                logger.info("Creating initial superuser...")
                superuser = User(
                    username="admin",
                    email="admin@example.com",
                    hashed_password=get_password_hash("admin"),  # Change this in production!
                    full_name="System Administrator",
                    is_active=True,
                    is_superuser=True
                )
                db.add(superuser)
                db.commit()
                logger.info("Superuser created.")
            else:
                logger.info("Superuser already exists.")
                
        # Run Alembic migrations
        logger.info("Running database migrations...")
        alembic_ini = Path(__file__).resolve().parent.parent / "alembic.ini"
        result = subprocess.run(
            ["alembic", "--config", str(alembic_ini), "upgrade", "head"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("Database migrations completed.")
            logger.info(result.stdout)
        else:
            logger.error("Error running migrations:")
            logger.error(result.stderr)
            
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

if __name__ == "__main__":
    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialization completed.")
 