import re
from io import StringIO

import numpy as np
import pandas as pd
import torch

FILENAME_TRAIN = '../../train.csv'
FILENAME_TEST = '../../test.csv'

with open(FILENAME_TRAIN) as fp:
    # read a list of lines into data
    data = fp.readlines()

data[4790] = data[4790].replace("/", "")
data[14716] = data[14716].replace("/", "")
data[14961] = data[14961].replace("/", "")
data[29895] = data[29895].replace("/", "")

# and write everything back
with open(FILENAME_TRAIN, 'w') as file:
    file.writelines(data)

with open(FILENAME_TRAIN) as file:
    lines = [re.sub(r'([^,])"(\s*[^\n])', r'\1/"\2', line) for line in file]
    df_train_origin = pd.read_csv(StringIO(''.join(lines)), escapechar="/")
df_train_origin = df_train_origin.drop(columns='Caption').join(df_train_origin['Caption'].str.replace('\"', ''))
print(df_train_origin.shape)

label_list = []

for i in df_train_origin['Labels']:
    label_list.extend(i.split())
label_list = sorted([int(i) for i in label_list])
label_list = [str(i) for i in label_list]

# create a label set
label_set = set(label_list)
label_set = sorted([int(i) for i in label_set])

# count label frequency
count_dic = {}
for i in label_set:
    count_dic[str(i)] = 0
for i in label_list:
    if i in count_dic:
        count_dic[i] += 1
print(count_dic)

label_counts = []
for i in label_set:
    label_counts.append(count_dic[str(i)])
print(label_counts)

total_samples = 30000
total_labels = sum(label_counts)
print(total_labels)

class_weight = [0 for _ in range(18)]
pos_weight = [0 for _ in range(18)]

for label_idx, count in enumerate(label_counts):
    if count != 0:
        class_weight[label_idx] = 1 - count / total_labels
        pos_weight[label_idx] = total_samples / count - 1

print('class_weight:', class_weight)
print('pos_weight:', pos_weight)
print(torch.tensor(class_weight))


def calculate_pos_weights(class_counts):
    pos_weights = np.ones_like(class_counts)
    neg_counts = [30000-pos_count for pos_count in class_counts]
    for cdx, (pos_count, neg_count) in enumerate(zip(class_counts,  neg_counts)):
        pos_weights[cdx] = neg_count / (pos_count + 1e-5)

    return torch.as_tensor(pos_weights, dtype=torch.float)


print(calculate_pos_weights(label_counts))
print()
