from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Replace with your actual MySQL credentials
# Format: mysql+pymysql://<username>:<password>@<host>:<port>/<database_name>
SQLALCHEMY_DATABASE_URL = "mysql+pymysql://root:AD21@localhost:3306/aquaxai"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()