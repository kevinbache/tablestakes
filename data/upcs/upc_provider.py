import pandas as pd

df = pd.read_csv('upc_corpus.csv').sample(frac=0.1)
print(len(df))
print(df.memory_usage(deep=True))
