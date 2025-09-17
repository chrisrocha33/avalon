import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# Use centralized config
from config import Config
engine = create_engine(Config.DATABASE['connection_string'], future=True)   

model_data = pd.read_sql_query("SELECT * FROM ml_spx_data", engine)

spx = model_data["^GSPC"]

print(spx)