import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


df = pd.read_csv('train.csv').values
print(df[:5])

class_cnt = {}


for idx, label in df:
    print(idx, label)
    if label not in class_cnt:
        class_cnt[label] = []
    class_cnt[label].append(idx)


for k, v in class_cnt.items():
    print(k, len(v))


big_x = []
big_y = []
small_x = []
small_y = []
for k, v in class_cnt.items():
    if len(v) < 30:
        small_x.extend(v)
        small_y.extend(np.ones(len(v), dtype=np.int16) * k)
    else:
        big_x.extend(v)
        big_y.extend(np.ones(len(v), dtype=np.int16) * k)

print(big_x)
print(big_y)

train_x, test_x, train_y, test_y = train_test_split(big_x, big_y, random_state=999, test_size=0.2)
train_x.extend(small_x)
train_y.extend(small_y)

with open('train.txt', 'w')as f:
    for fn, label in zip(train_x, train_y):
        f.write('./train/{}.jpg,{}\n'.format(fn, label))

with open('val.txt', 'w')as f:
    for fn, label in zip(test_x, test_y):
        f.write('./train/{}.jpg,{}\n'.format(fn, label))




