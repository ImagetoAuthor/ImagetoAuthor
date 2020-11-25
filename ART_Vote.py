import pandas as pd
import numpy as np

file1 = pd.read_csv('D:/competition/ART/47_94.csv',header=None)
file2 = pd.read_csv('D:/competition/ART/46_93.875.csv',header=None)
file3 = pd.read_csv('D:/competition/ART/33_93.125.csv',header=None)
file4 = pd.read_csv('D:/competition/ART/39_92.625.csv',header=None)
file5 = pd.read_csv('D:/competition/ART/40_92.75.csv',header=None)

file_sum = pd.concat([file1,file2[1],file3[1],file4[1],file5[1]],axis=1)

pre = []
for i in range(len(file_sum)):
    pre.append(file_sum.loc[i].value_counts().index[0])
print(pre)

test_csv = pd.DataFrame()
test_csv['num'] = list(range(800))
test_csv['label'] = pre

test_csv.to_csv('D:/competition/ART/result.csv', index=None,header=False)
