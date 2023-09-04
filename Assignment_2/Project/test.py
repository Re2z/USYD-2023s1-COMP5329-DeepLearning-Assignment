import os
import re
from io import StringIO

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd

# predicted = torch.tensor([])
# output = torch.tensor([[0, 0.1, 0.2, 0.3, 0.1, 0.4, 0.5, 0.1, 0.2, 0], [1, 1.1, 1.2, 1.3, 1.1, 1.4, 1.5, 1.1, 1.2, 1]])
# predicted = torch.cat((predicted, output), 0)
# predicted = torch.cat([predicted, torch.tensor(
#     [[0, 0.1, 0.2, 0.3, 0.1, 0.4, 0.5, 0.1, 0.2, 0], [1, 1.1, 1.2, 1.3, 1.1, 1.4, 1.5, 1.1, 1.2, 2]])], 0)
#
# print(predicted > 0.4)
#
# id = ['1.jpg', '2.jpg', '3.jpg', '4.jpg']
# a = []
#
# for i, line in enumerate(predicted):
#     prediction = []
#     print(line)
#     line = line > 0.4
#     print(line)
#     for idx in range(len(line)):
#         if line[idx]:
#             prediction.append(idx + 1)
#     label = ' '.join(str(e) for e in prediction)
#     a.append(label)
# print(id)
# print(a)
# df = pd.DataFrame({'ImageID': id, 'Label': a})
# print(df)
# df.to_csv("result.csv", index=False)
#
#

# import torchvision.models as models
#
# efficient_net = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
#
# def calculate_model_size(model):
#     total_size = 0
#     embed_size = 0
#     for name, param in model.named_parameters():
#         param_size = np.prod(list(param.shape)) * 4
#         if "embeddings" in name:
#             embed_size += param_size
#         else:
#             total_size += param_size
#     total_size /= 1e6
#     embed_size /= 1e6
#     print(f"Model size: {total_size} MB; Word Embedding Size: {embed_size} MB")
#     if total_size > 100:
#         raise ValueError("Model too large!")
#
#
# calculate_model_size(efficient_net)


# class A2Dataset(Dataset):
#     def __init__(self, data_path, type, transforms=None):
#         # data_path: the root path of dataset
#         # transforms: img tranfroms
#         self.transforms = transforms
#         # self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
#         self.type = type
#         self.data_path = data_path
#         if self.type == 'train':
#             file_path = os.path.join(data_path, 'train_split.csv')
#         elif self.type == 'test':
#             file_path = os.path.join(data_path, 'test.csv')
#         elif self.type == 'val':
#             file_path = os.path.join(data_path, 'val_split.csv')
#         elif self.type == 'whole':
#             file_path = os.path.join(data_path, 'train.csv')
#         with open(file_path) as file:
#             lines = [re.sub(r'([^,])"(\s*[^\n])', r'\1/"\2', line) for line in file]
#             df = pd.read_csv(StringIO(''.join(lines)), escapechar="/")
#         self.samples = df
#
#         if self.type != "test":
#             label_set = []
#             for i in self.samples["Labels"]:
#                 label_set.extend(i.split())
#             label_set = set(label_set)
#             label_set = sorted([int(i) for i in label_set])
#
#             self.imgs = []
#             self.labels = []
#             for index, row in self.samples.iterrows():
#                 self.imgs.append(row['ImageID'])
#                 self.labels.append(row['Labels'])
#
#             self.classes = label_set
#             self.labels = self.samples['Labels']
#             self.caption = self.samples['Caption']
#             self.img_id = self.samples['ImageID']
#
#             for item_id in range(len(self.labels)):
#                 item = self.labels[item_id].split()
#                 vector = [1 if str(cls) in item else 0 for cls in self.classes]
#                 self.labels[item_id] = np.array(vector, dtype=float)
#
#         else:
#             self.imgs = []
#             for index, row in self.samples.iterrows():
#                 self.imgs.append(row['ImageID'])
#
#             self.caption = self.samples['Caption']
#             self.img_id = self.samples['ImageID']
#             self.caption = self.samples['Caption']
#
#         print("Loading {} set with {} samples".format(self.type, len(self.imgs)))
#
#     def __getitem__(self, item):
#         # process caption
#         caption = self.caption[item]
#
#         # caption = self.tokenizer.encode_plus(
#         #     caption,
#         #     None,
#         #     add_special_tokens=True,
#         #     max_length=28,
#         #     pad_to_max_length=True,
#         #     return_token_type_ids=True,
#         #     truncation=True
#         # )
#         # ids = caption['input_ids']
#         # mask = caption['attention_mask']
#         # token_type_ids = caption["token_type_ids"]
#
#         # process img and target
#         if self.type != "test":
#             target = torch.tensor(self.labels[item], dtype=torch.float)
#             img_path = os.path.join(self.data_path, 'data', self.samples['ImageID'][item])
#             img = Image.open(img_path)
#             if self.transforms is not None:
#                 img = self.transforms(img)
#
#             out_data = {
#                 # 'ids': torch.tensor(ids, dtype=torch.long),
#                 # 'mask': torch.tensor(mask, dtype=torch.long),
#                 # 'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
#                 'caption': caption,
#                 'img': img,
#                 'target': target
#             }
#             print(target)
#             return out_data
#         else:
#             img_path = os.path.join(self.data_path, 'data', self.samples['ImageID'][item])
#             img = Image.open(img_path)
#             if self.transforms is not None:
#                 img = self.transforms(img)
#
#             out_data = {
#                 # 'ids': torch.tensor(ids, dtype=torch.long),
#                 # 'mask': torch.tensor(mask, dtype=torch.long),
#                 # 'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
#                 'caption': caption,
#                 'img': img
#             }
#             return out_data
#
#     def __len__(self):
#         return len(self.imgs)
#
#
# a2 = A2Dataset("../", "whole")
# a2.__getitem__(1)

a = torch.tensor(([0.33, 0.1, 0.5, 0.2, 0.55, 0.9, 0.2, 0.111]))
b = [0.33, 0.1, 0.5, 0.2, 0.55, 0.9, 0.2, 0.111]


# def out_to_file(ID, prediction):
#     label = []
#     for i, pred in enumerate(prediction):
#         prediction = []
#         pred = pred > 0.5
#         for idx in range(len(pred)):
#             if pred[idx]:
#                 if idx > 10:
#                     prediction.append(idx + 2)
#                 else:
#                     prediction.append(idx + 1)
#         result = ' '.join(str(e) for e in prediction)
#         label.append(result)
#     for i in range(len(ID)):
#         ID[i] = str(ID[i])
#         print(ID[i])
#         ID[i] = ID[i] + '.jpg'
#     df = pd.DataFrame({'ImageID': ID, 'Label': label})
#     df.to_csv("result.csv", index=False)
#
#
# out_to_file(b, a)

history = {'MLP_loss': [0.8702,0.6806,0.6038,0.5685,0.5434,0.5346,0.5331,0.5109,0.5125,0.5181], 'MLP_accuracy': [0.7308,0.7893,0.8075,0.8129,0.8189,0.8209,0.8218,0.8276,0.8301,0.8245],
       'MLP_precision': [0.8296,0.8459,0.8599,0.8573,0.8618,0.8614,0.8524,0.8598,0.8574,0.8544],'MLP_recall': [0.6336,0.7352,0.7594,0.7722,0.7778,0.7793,0.7970,0.7994,0.8067,0.8000]
       'ResNet_loss': [0.7446,0.8409,0.4793,0.4542,0.4218,0.4156,0.4298,0.3893,0.4119,0.3978], 'ResNet_accuracy': [0.7620,0.7283,0.8384,0.8433,0.8515,0.8514,0.8492,0.8606,0.8583,0.8573],
       'ResNet_precision': [0.8106,0.7872,0.8647,0.8710,0.8777,0.8732,0.8693,0.8783,0.8741,0.8745],'ResNet_recall': [0.7185,0.6722,0.8127,0.8186,0.8256,0.8292,0.8321,0.8454,0.8440,0.8396]}

