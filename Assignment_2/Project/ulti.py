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

device = "cuda" if torch.cuda.is_available() else "cpu"


class DataLoad(torch.utils.data.Dataset):
    """
    dataloader
    """

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
            # One-hot
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


def caption_extract(caption):
    """
    For each caption, extract the main feature word

    :param caption:
    :return:
    """
    caption_in_word = []
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    for sentence in caption:
        sentence = sentence.lower()
        sentence = re.sub(r'[^A-Za-z]+', ' ', sentence)
        words = sentence.split()

        filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        caption_in_word.append(filtered_words)

    return caption_in_word


def process_captions(train_file_path, test_file_path):
    """
    Collect all captions in data file and get the maximum length of them

    :param train_file_path:
    :param test_file_path:
    :return:
    """
    with open(train_file_path) as train_file:
        train_lines = [re.sub(r'([^,])"(\s*[^\n])', r'\1/"\2', line) for line in train_file]
        train_caption = pd.read_csv(StringIO(''.join(train_lines)), escapechar="/")
        train_caption = train_caption.drop(columns='Caption').join(train_caption['Caption'].str.replace('\"', ''))

    with open(test_file_path) as test_file:
        test_lines = [re.sub(r'([^,])"(\s*[^\n])', r'\1/"\2', line) for line in test_file]
        test_caption = pd.read_csv(StringIO(''.join(test_lines)), escapechar="/")
        test_caption = test_caption.drop(columns='Caption').join(test_caption['Caption'].str.replace('\"', ''))

    caption = caption_extract(
        pd.concat([train_caption['Caption'], test_caption['Caption']], axis=0, ignore_index=True))

    max_len = max(len(s) for s in caption)

    return caption, max_len


# Download the Word2Vec model
embedding_model = gensim.downloader.load('word2vec-google-news-300')
# vector size of each word
embedding_dim = embedding_model.vector_size


def create_word_model(caption):
    """
    Creating the own model based on the 'word2vec-google-news-300'

    :param caption:
    :return:
    """
    # Extract words from the caption
    word_list = [word for sentence in caption for word in sentence]

    # Count word occurrences in the caption
    word_counter = Counter(word_list)

    # Select the words with more than 5 appearances for the vocabulary set
    vocab_set = {word for word, count in word_counter.items() if count >= 5}

    # Add the PAD and UNKNOWN to the vocabulary set
    vocab_set.update(['[PAD]', '[UNKNOWN]'])

    # Sort the vocabulary set
    vocab_list = sorted(vocab_set)

    # Create the word dictionary and embedding table based on the vocabulary
    vocab_dict = {}
    emb_table = []
    for i, word in enumerate(vocab_list):
        vocab_dict[word] = i
        if word in embedding_model:
            emb_table.append(embedding_model[word])
        else:
            emb_table.append([0] * embedding_dim)
    emb_table = np.array(emb_table)

    return vocab_dict, emb_table


whole_caption, max_caption_len = process_captions(TRAIN_CSV, TEST_CSV)
vocab_dict, emb_table = create_word_model(whole_caption)


def tokenizer(caption):
    """
    Caption tokenizer for each input

    :param caption:
    :return:
    """
    tokenize = []
    caption = caption_extract(caption)
    for item in caption:
        temp = [vocab_dict[word] if word in vocab_dict else vocab_dict['[UNKNOWN]'] for word in item]
        if len(temp) < max_caption_len:
            temp += [vocab_dict['[PAD]']] * (max_caption_len - len(temp))
        else:
            temp = temp[:max_caption_len]
        tokenize.append(temp)
    return tokenize


class CnnModel(nn.Module):
    """
    CNN model
    """

    def __init__(self):
        super().__init__()
        efficient_net = EfficientNet.from_pretrained('efficientnet-b4')
        efficient_net._fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=efficient_net._fc.in_features, out_features=18)
        )
        self.efficient_model = efficient_net
        self.sig = nn.Sigmoid()

        weights = torch.load('/Users/renenze/.cache/torch/hub/checkpoints/efficientnet-b4-6ed6700e.pth')
        self.load_state_dict(state_dict=weights, strict=False)

    def forward(self, x):
        out = self.sig(self.efficient_model(x))
        return out


class NlpModel(nn.Module):
    """
    NLP model
    """

    def __init__(self):
        super(NlpModel, self).__init__()
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
        return self.sigm(out)


class Combine(nn.Module):
    """
    Combine model for the output of the CNN and NLP models
    """

    def __init__(self):
        super(Combine, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=36, out_features=252),
            nn.ReLU(),
            nn.Linear(in_features=252, out_features=18),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(x)


def get_acc(output, label):
    """
    Calculate the score

    :param output:
    :param label:
    :return:
    """
    output = output.cpu().detach().numpy()
    label = label.cpu().numpy()
    predicted_labels = []
    for preds in output:
        if sum(preds > THRESHOLD) == 0:
            temp = np.zeros(18)
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
    """
    Train epoch

    :param cnn_model:
    :param nlp_model:
    :param dataloader:
    :return:
    """
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
        output = com_model(torch.cat((img_output, nlp_output), 1))

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
    """
    Evaluate

    :param cnn_model:
    :param nlp_model:
    :param dataloader:
    :return:
    """
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
            output = com_model(torch.cat((img_output, nlp_output), 1))
            loss = loss_function(output, label)

            precision, recall, f1 = get_acc(output, label)

            batch_num += 1
            val_loss += loss.item() * img.size(0)
            val_precision += precision
            val_recall += recall
            val_f1 += f1

    return val_loss, val_precision, val_recall, val_f1, batch_num


def fit(cnn_model, nlp_model, train_dataloader, val_dataloader, epochs):
    """
    Utilize

    :param cnn_model:
    :param nlp_model:
    :param train_dataloader:
    :param val_dataloader:
    :param epochs:
    :return:
    """
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
            com_model.eval()
            torch.save(com_model, 'best_steps_{}.pt'.format(epoch))
            com_model.train()

    duration = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))


def predict(cnn_model, nlp_model, dataloader):
    """
    Inference of the test dataset

    :param cnn_model:
    :param nlp_model:
    :param dataloader:
    :return:
    """
    img_id = []
    pred = torch.tensor([])
    cnn_model.eval()
    nlp_model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            img, caption = batch["img"].to(device), torch.from_numpy(
                np.array(tokenizer(batch["caption"]))).to(device)
            img_output = cnn_model(img)
            nlp_output = nlp_model(caption)
            output = com_model(torch.cat((img_output, nlp_output), 1))

            img_id.extend(batch["id"])
            pred = torch.cat([pred, output.cpu()], dim=0)
    out_to_file(img_id, pred)
    return img_id, pred


def out_to_file(img_id, prediction):
    """
    Output the result to the csv file

    :param img_id:
    :param prediction:
    :return:
    """
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
    for n in range(len(img_id)):
        img_id[n] = str(img_id[n])
        img_id[n] = img_id[n] + '.jpg'
    df = pd.DataFrame({'ImageID': id, 'Label': label})
    df.to_csv("result.csv", index=False)


# Image Pre-Processing
transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.5, 1.5),
                                shear=None),
        transforms.RandomResizedCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
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

cnn_model = CnnModel().to(device)
nlp_model = NlpModel().to(device)
com_model = Combine().to(device)

# AdamW,BCELoss: 86    Adam, MSE: 86,  AdamW, L1: 86
optimizer = torch.optim.Adam(com_model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
loss_function = nn.MSELoss()

# Train
fit(cnn_model, nlp_model, train_loader, val_loader, 5)
