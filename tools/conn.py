import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base


DATABASE_URL = "postgresql://kurti:kurti123@localhost:5432/kurti_db" #os.getenv("PG_DATABASE_URL")

# Create engine
engine = create_engine(DATABASE_URL, echo=False)

# Session factory
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# Base class for models
Base = declarative_base()


def init_db():
    """
    Call this once at startup to create tables.
    """
    Base.metadata.create_all(bind=engine)