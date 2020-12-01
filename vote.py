import pandas as pd
import numpy as np


files = ['1.csv', '2.csv', '3.csv', '4.csv']
weights = [1, 1, 1, 1]

results = np.zeros((800, 49))
for file, w in zip(files, weights):
    print(w)
    df = pd.read_csv(file, header=None).values
    for x, y in df:
        # print(x, y)
        results[x, y] += w
        # break

print(results[0])

submit = {
    'name': np.arange(800).tolist(),
    'pred': np.argmax(results, axis=1).tolist()
    }

for k, v in submit.items():
    print(k, v)

df = pd.DataFrame(submit)
df.to_csv('vote.csv', header=False, index=False)



