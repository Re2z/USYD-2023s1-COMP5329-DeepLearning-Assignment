"""
This class use the skimage to read the image.
"""
import os
import re
import nltk
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import skimage.io as io
import gensim.downloader
from tqdm import tqdm
from io import StringIO
from torchvision import transforms
from collections import Counter
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import f1_score, precision_score, recall_score

nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Concluded 19 labels - one-hot encoded.
TRAIN_CSV = "../train.csv"
TEST_CSV = "../test.csv"
IMAGE_DIR = "../data"
TRAIN_VAL_PROP = 0.67
BATCH_SIZE = 32
SEED = 2023
LR = 0.0001
MAX_EPOCH = 100
THRESHOLD = 0.5


class DataLoad(torch.utils.data.Dataset):
    def __init__(self, file, image, transform=None, text_csv=None):
        self.image_dir = image
        self.text_csv = text_csv
        self.transform = transform
        self.classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19]

        with open(file) as file:
            lines = [re.sub(r'([^,])"(\s*[^\n])', r'\1/"\2', line) for line in file]
            dataframe = pd.read_csv(StringIO(''.join(lines)), escapechar="/")
            self.dataframe = dataframe.drop(columns='Caption').join(dataframe['Caption'].str.replace('\"', ''))

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item.to_list()

        # Combine the image path with the image file name
        img_path = os.path.join(self.image_dir, self.dataframe.iloc[item, 0])
        img = io.imread(img_path)
        img_id = self.dataframe.iloc[item, 0]

        if not self.text_csv:
            img_caption = self.dataframe.iloc[item, 2]
            img_label = self.dataframe.iloc[item, 1].split(' ')
            img_label = [int(x) for x in img_label]
            for i in range(len(img_label)):
                img_label[i] = [1 if cls == img_label[i] else 0 for cls in self.classes]
            img_label = sum(torch.tensor(img_label, dtype=torch.float))

            if self.transform:
                img = self.transform(img)

            sample = {'img': img, 'label': img_label, 'id': img_id, 'caption': img_caption}
        else:
            img_caption = self.dataframe.iloc[item, 1]
            if self.transform:
                img = self.transform(img)
            sample = {'img': img, 'id': img_id, 'caption': img_caption}

        return sample


# Image Pre-Processing
transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.5, 1.5),
                                shear=None),
        # transforms.RandomResizedCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        # transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

dataset = DataLoad(file=TRAIN_CSV, image=IMAGE_DIR, transform=transforms['train'], text_csv=False)
test_dataset = DataLoad(file=TEST_CSV, image=IMAGE_DIR, transform=transforms['val'], text_csv=True)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(TRAIN_VAL_PROP * len(dataset)),
                                                                     len(dataset) - (
                                                                         int(TRAIN_VAL_PROP * len(dataset)))])

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)


# def caption_extract(caption):
#     caption_in_word = []
#     for sentences in caption:
#         sentence_in_word = []
#         sentences = sentences.lower()
#         sentences = re.sub(r'[^A-Za-z]+', ' ', sentences)
#         for word in sentences.split():
#             if word not in set(stopwords.words('english')):
#                 sentence_in_word.append(word)
#
#         lemma_word = [WordNetLemmatizer().lemmatize(item) for item in sentence_in_word]
#
#         caption_in_word.append(lemma_word)
#     return caption_in_word

def caption_extract(caption):  # extract the main word in caption
    caption_in_word = []
    for sentences in caption:
        sentence_in_word = []
        sentences = sentences.lower()  # lowercase
        sentences = re.sub(r'[^A-Za-z]+', ' ', sentences)  # remove all symbols and digits
        for word in sentences.split():
            if word not in set(stopwords.words('english')):  # remove stopword
                sentence_in_word.append(word)

        lemma_word = [WordNetLemmatizer().lemmatize(item) for item in sentence_in_word]  # lammatisation

        # token = [word_tokenize(word) for word in text_le]
        caption_in_word.append(lemma_word)
    return caption_in_word


with open("../train.csv") as file:
    lines = [re.sub(r'([^,])"(\s*[^\n])', r'\1/"\2', line) for line in file]
    train_caption = pd.read_csv(StringIO(''.join(lines)), escapechar="/")
    train_caption = train_caption.drop(columns='Caption').join(train_caption['Caption'].str.replace('\"', ''))
with open("../test.csv") as file:
    lines = [re.sub(r'([^,])"(\s*[^\n])', r'\1/"\2', line) for line in file]
    test_caption = pd.read_csv(StringIO(''.join(lines)), escapechar="/")
    test_caption = test_caption.drop(columns='Caption').join(test_caption['Caption'].str.replace('\"', ''))

whole_caption = pd.concat([train_caption['Caption'], test_caption['Caption']], axis=0, ignore_index=True)
caption_processed = caption_extract(whole_caption)
max_seq_len = max(len(s) for s in caption_processed)

# Download the Word2Vec model
embedding_model = gensim.downloader.load('word2vec-google-news-300')
# vector size of each word
embedding_dim = embedding_model.vector_size

# Add the word appeared in the caption extraction to the word list
word_list = []
for sentence in caption_processed:
    for word in sentence:
        word_list.append(word)

# Select the word with more than 5 appearance in the caption to the vocabulary set
vocab_set = set()
counter = Counter(word_list)
for i in counter:
    if counter[i] >= 5:
        vocab_set.add(i)

# Add the PAD and UNKNOWN to the vocabulary set, each word in the vocabulary list only exist once
vocab_set.add('[PAD]')
vocab_set.add('[UNKOWN]')
vocab_list = list(vocab_set)
vocab_list.sort()

# word dictionary
vocab_dic = {}
emb_table = []

# Creating the word model based on the vocabulary
for i, word in enumerate(vocab_list):
    vocab_dic[word] = i
    if word in embedding_model:
        emb_table.append(embedding_model[word])
    else:
        emb_table.append([0] * embedding_dim)
emb_table = np.array(emb_table)


# def create_word_model(caption):
#     # Add the words appeared in the caption extraction to the word list
#     word_list = []
#     for sentence in caption:
#         for word in sentence:
#             word_list.append(word)
#
#     # Select the words with more than 5 appearances in the caption to the vocabulary set
#     vocab_set = set()
#     counter = Counter(word_list)
#     for word, count in counter.items():
#         if count >= 5:
#             vocab_set.add(word)
#
#     # Add the PAD and UNKNOWN to the vocabulary set, each word in the vocabulary list only exists once
#     vocab_set.add('[PAD]')
#     vocab_set.add('[UNKNOWN]')
#     vocab_list = sorted(list(vocab_set))
#
#     # Word dictionary
#     vocab_dict = {}
#     emb_table = []
#
#     # Create the word model based on the vocabulary
#     for i, word in enumerate(vocab_list):
#         vocab_dict[word] = i
#         if word in embedding_model:
#             emb_table.append(embedding_model[word])
#         else:
#             emb_table.append([0] * embedding_dim)
#     emb_table = np.array(emb_table)
#
#     return vocab_dict, emb_table

def tokenizer(caption):
    tokenize = []
    caption = caption_extract(caption)
    for item in caption:
        temp = [vocab_dic[word] if word in vocab_dic else vocab_dic['[UNKOWN]'] for word in item]
        if len(temp) < max_seq_len:
            temp += [vocab_dic['[PAD]']] * (max_seq_len - len(temp))
        else:
            temp = temp[:max_seq_len]
        tokenize.append(temp)
    return tokenize


class CnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        efficient_net = EfficientNet.from_pretrained('efficientnet-b4')
        efficient_net._fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=efficient_net._fc.in_features, out_features=18)
        )
        self.efficient_model = efficient_net
        self.sig = nn.Sigmoid()

        weights = torch.load('efficientnet-b4-6ed6700e.pth')
        self.load_state_dict(state_dict=weights, strict=False)

    def forward(self, x):
        out = self.sig(self.efficient_model(x))
        return out


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.emb = nn.Embedding(num_embeddings=emb_table.shape[0], embedding_dim=emb_table.shape[1])
        # Initialize the Embedding layer with the lookup table we created
        self.emb.weight.data.copy_(torch.from_numpy(emb_table))
        # make this lookup table untrainable
        self.emb.weight.requires_grad = False
        self.lstm = nn.LSTM(input_size=emb_table.shape[1], hidden_size=300, num_layers=2, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(in_features=300 * 2, out_features=18)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = self.emb(x)
        output, (h, c) = self.lstm(x)
        x = torch.cat((h[0, :, :], h[1, :, :]), 1)
        out = self.linear(x)
        # x = self.emb(x)
        # x, _ = self.lstm(x)
        # out = self.linear(x[:, -1, :])
        return self.sigm(out)


writer = SummaryWriter("logs")
device = "cuda" if torch.cuda.is_available() else "cpu"

classifier = nn.Sequential(
    nn.Linear(in_features=36, out_features=2392),
    nn.ReLU(),
    nn.Linear(in_features=2392, out_features=18),
    nn.Sigmoid()
)

cnn_model = CnnModel().to(device)
nlp_model = LSTM().to(device)
classifier = classifier.to(device)

# AdamW,BCELoss 86     Adam, MSE 86,  AdamW, L1 86
optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
loss_function = nn.MSELoss()


def get_acc(output, label):
    output = output.cpu().detach().numpy()
    label = label.cpu().numpy()
    predicted_labels = []
    for preds in output:
        if sum(preds > THRESHOLD) == 0:
            temp = np.zeros((18))
            temp[np.argmax(preds[1])] = 1
            predicted_labels.append(temp)
        else:
            predicted_labels.append(np.array(preds > THRESHOLD, dtype=float))
    pred = np.array(predicted_labels, dtype=float)

    precision = precision_score(y_true=pred, y_pred=label, average='weighted')
    recall = recall_score(y_true=pred, y_pred=label, average='weighted')
    f1 = f1_score(y_true=pred, y_pred=label, average='weighted')

    return precision, recall, f1


def train(cnn_model, nlp_model, dataloader):
    train_loss = 0
    train_precision = 0
    train_recall = 0
    train_f1 = 0
    batch_size = 0
    batch_num = 0
    cnn_model.train()
    nlp_model.train()
    # batch item: img, label, id, caption
    for batch in tqdm(dataloader):
        # Caption: input_ids, token_type_ids, attention_mask
        img, label, caption = batch["img"].to(device), batch["label"].to(device), torch.from_numpy(np.array(tokenizer(
            batch["caption"]))).to(device)
        img_output = cnn_model(img)
        nlp_output = nlp_model(caption)
        output = classifier(torch.cat((img_output, nlp_output), 1))

        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()

        precision, recall, f1 = get_acc(output, label)
        batch_num += 1
        batch_size += 1
        train_loss += loss.item() * img.size(0)
        train_precision += precision
        train_recall += recall
        train_f1 += f1

        print(
            '\rBatch[{}/{}] - loss: {:.6f}  precision: {:.4f}%  recall: {:.4f}%  F1_score: {:.4f}%'.format(batch_size,
                                                                                                           len(dataloader),
                                                                                                           loss.item() * img.size(
                                                                                                               0),
                                                                                                           precision,
                                                                                                           recall, f1))
    scheduler.step()
    return train_loss, train_precision, train_recall, train_f1, batch_num


def evaluate(cnn_model, nlp_model, dataloader):
    val_loss = 0
    val_precision = 0
    val_recall = 0
    val_f1 = 0
    batch_num = 0
    cnn_model.eval()
    nlp_model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            img, label, caption = batch["img"].to(device), batch["label"].to(device), torch.from_numpy(
                np.array(tokenizer(
                    batch["caption"]))).to(device)
            img_output = cnn_model(img)
            nlp_output = nlp_model(caption)
            output = classifier(torch.cat((img_output, nlp_output), 1))
            loss = loss_function(output, label)

            precision, recall, f1 = get_acc(output, label)

            batch_num += 1
            val_loss += loss.item() * img.size(0)
            val_precision += precision
            val_recall += recall
            val_f1 += f1

    return val_loss, val_precision, val_recall, val_f1, batch_num


def fit(cnn_model, nlp_model, train_dataloader, val_dataloader, epochs):
    start_time = time.time()
    best_f1 = 0
    for epoch in range(epochs):
        train_loss, train_precision, train_recall, train_f1, train_batch_num = train(cnn_model, nlp_model,
                                                                                     train_dataloader)
        val_loss, val_precision, val_recall, val_f1, val_batch_num = evaluate(cnn_model, nlp_model, val_dataloader)
        print('Epoch [{}/{}]'.format(epoch, epochs))
        print('Train - loss: {:.6f}  precision: {:.4f}%  recall: {:.4f}%  F1_score: {:.4f}%'.format(
            train_loss / len(train_dataset), train_precision / train_batch_num, train_recall / train_batch_num,
            train_f1 / train_batch_num))
        print('Val - loss: {:.6f}  precision: {:.4f}%  recall: {:.4f}%  F1_score: {:.4f}%'.format(
            val_loss / len(val_dataset), val_precision / val_batch_num, val_recall / val_batch_num,
            val_f1 / val_batch_num))

        if val_f1 / val_batch_num > best_f1:
            best_f1 = val_f1 / val_batch_num
            print('Saving best pre_model, f1: {:.4f}%\n'.format(best_f1))
            classifier.eval()
            torch.save(classifier, 'best_steps_{}.pt'.format(epoch))
            classifier.train()

    duration = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))


def predict(cnn_model, nlp_model, dataloader):
    id = []
    pred = torch.tensor([])
    cnn_model.eval()
    nlp_model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            img, caption = batch["img"].to(device), torch.from_numpy(
                np.array(tokenizer(batch["caption"]))).to(device)
            img_output = cnn_model(img)
            nlp_output = nlp_model(caption)
            output = classifier(torch.cat((img_output, nlp_output), 1))

            id.extend(batch["id"])
            pred = torch.cat([pred, output.cpu()], dim=0)
    out_to_file(id, pred)
    return id, pred


def out_to_file(ID, prediction):
    label = []
    for i, pred in enumerate(prediction):
        prediction = []
        pred = pred > THRESHOLD
        for idx in range(len(pred)):
            if pred[idx]:
                if idx > 10:
                    prediction.append(idx + 2)
                else:
                    prediction.append(idx + 1)
        result = ' '.join(str(e) for e in prediction)
        label.append(result)
    for i in range(len(ID)):
        ID[i] = str(ID[i])
        ID[i] = ID[i] + '.jpg'
    df = pd.DataFrame({'ImageID': id, 'Label': label})
    df.to_csv("result.csv", index=False)


fit(cnn_model, nlp_model, train_loader, val_loader, 5)
