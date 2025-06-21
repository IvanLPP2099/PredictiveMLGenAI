from contextlib import contextmanager
import MySQLdb
import os
from dotenv import load_dotenv


"""
load_dotenv()
"""

@contextmanager
def get_db_connection():
    connection = MySQLdb.connect(
        host=os.environ["DB_HOST"],
        port=int(os.environ["DB_PORT"]),
        user=os.environ["DB_USER"],
        passwd=os.environ["DB_PASSWORD"],
        db=os.environ["DB_NAME"]
    )
    
    try:
        yield connection
    finally:
        connection.close()
