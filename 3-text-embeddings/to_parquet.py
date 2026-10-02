import os

import pandas as pd

script_dir = (
    os.path.dirname(os.path.abspath(__file__))
    if "__file__" in globals()
    else os.getcwd()
)

df = pd.read_csv(os.path.join(script_dir, "data", "AI_vs_Human.csv"))
df.to_parquet(os.path.join(script_dir, "data", "AI_vs_Human.parquet"))
