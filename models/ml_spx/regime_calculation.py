import pandas as pd
import numpy as np
from sqlalchemy import create_engine

server = 'localhost'
port = '5432'
database = 'avalon'
username = 'admin'
password = 'password!'

conn_str = f'postgresql+psycopg2://{username}:{password}@{server}:{port}/{database}'
engine = create_engine(conn_str, future=True)   

model_data = pd.read_sql_query("SELECT * FROM ml_spx_data", engine)

spx = model_data["^GSPC"]

print(spx)