import os
import re
import time
import torch
import gensim.downloader
import numpy as np
import pandas as pd
import torch.nn as nn
import skimage.io as io
import matplotlib.pyplot as plt
from tqdm import tqdm
from io import StringIO
from torchvision import transforms, models
from collections import Counter
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings

warnings.filterwarnings("ignore")  # Ignore the warning of the sklearn version

import nltk

nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# File path and the Global parameters
TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"
IMAGE_DIR = "data"
TRAIN_VAL_PROP = 0.8
BATCH_SIZE = 32
SEED = 2023
LR = 0.001
MAX_EPOCH = 7
THRESHOLD = 0.3
NUM_CLASS = 18
EMBEDDING_MODEL = gensim.downloader.load('glove-wiki-gigaword-50')  # Download the glove-wiki model
device = "cuda" if torch.cuda.is_available() else "cpu"  # Set the training device of CUDA

# Correct the error data item in the train data file
with open(TRAIN_CSV) as fp:
    data = fp.readlines()

data[4790] = data[4790].replace("/", "")
data[14716] = data[14716].replace("/", "")
data[14961] = data[14961].replace("/", "")
data[29895] = data[29895].replace("/", "")

# Write back
with open(TRAIN_CSV, 'w') as file:
    file.writelines(data)


def caption_extract(caption):
    """
    This function takes a caption as input and extracts the main feature words from it.
    It also performs preprocessing by removing non-meaningful letters and numbers from the caption.
    Delete the words in error.

    :param caption: input caption, which can be either a string or a list of strings
    :return: a list of extracted main feature words from each caption
    """
    caption_in_word = []
    # Set of stopwords
    stop_words = set(stopwords.words('english'))

    # Lemmatizer object
    lemmatizer = WordNetLemmatizer()

    for sentence in caption:
        # Convert the caption to lowercase and remove non-alphabetic characters and replace with space
        sentence = sentence.lower()
        sentence = re.sub(r'[^A-Za-z]+', ' ', sentence)
        # Split the caption into a list of words
        words = sentence.split()

        # Remove stopwords and perform lemmatization
        filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

        # Set of word errors
        word_error = {'baeball', 'basball', 'blackandwhite', 'bluewhite', 'checkerd', 'firehydrant',
                      'firsbee', 'fourwheeler', 'frisbe', 'frizbee', 'kiteboards', 'krispee',
                      'midswing', 'parasailers', 'skiboard', 'skii', 'skiies', 'surfboarder',
                      'surfboarding', 'tball', 'umbrells', 'windsurfs', 'deckered', 'rared',
                      'snowcovered'}

        # Filter out words with errors
        correct_word = [word for word in filtered_words if not word in word_error]
        caption_in_word.append(correct_word)

    return caption_in_word


def process_captions(train_file_path, test_file_path):
    """
    Processes captions from train and test files, performs preprocessing, and extracts main feature words.

    :param train_file_path: file path of the train captions file
    :param test_file_path:  path of the test captions file
    :return: processed captions and the maximum length among all captions
    """
    # Processing train captions
    with open(train_file_path) as train_file:
        # Preprocessing step to handle quotes inside the captions
        train_lines = [re.sub(r'([^,])"(\s*[^\n])', r'\1/"\2', line) for line in train_file]
        train_caption = pd.read_csv(StringIO(''.join(train_lines)), escapechar="/")
        train_caption = train_caption.drop(columns='Caption').join(train_caption['Caption'].str.replace('\"', ''))

    # Processing test captions
    with open(test_file_path) as test_file:
        # Preprocessing step to handle quotes inside the captions
        test_lines = [re.sub(r'([^,])"(\s*[^\n])', r'\1/"\2', line) for line in test_file]
        test_caption = pd.read_csv(StringIO(''.join(test_lines)), escapechar="/")
        test_caption = test_caption.drop(columns='Caption').join(test_caption['Caption'].str.replace('\"', ''))

    # Extracting main feature words from concatenated train and test captions
    caption = caption_extract(pd.concat([train_caption['Caption'], test_caption['Caption']], axis=0, ignore_index=True))

    # Finding the maximum length among all captions
    max_len = max(len(s) for s in caption)

    return caption, max_len


whole_caption, max_caption_len = process_captions(TRAIN_CSV, TEST_CSV)
# vector size of each word
embedding_dim = EMBEDDING_MODEL.vector_size


def create_word_model(caption):
    """
    Creates a word model by extracting words from captions basing on the pre-trained word embedding model.
    Building a vocabulary set and creating word embeddings.

    :param caption: list of captions, where each caption is a list of words
    :return: vocabulary dictionary and word embedding table
    """
    # Extract words from the caption
    word_list = [word for sentence in caption for word in sentence]

    # Count word occurrences in the caption
    word_counter = Counter(word_list)

    # Select the words with more than 1 appearance for the vocabulary set
    vocab_set = {word for word, count in word_counter.items() if count > 1}

    # Add the PAD and UNKNOWN to the vocabulary set
    vocab_set.update(['[PAD]', '[UNKNOWN]'])

    # Sort the vocabulary set
    vocab_list = sorted(vocab_set)

    # Create the word dictionary and embedding table based on the pre-trained word embedding model
    vocab_dictionary = {}
    embedding_table = []
    for i, word in enumerate(vocab_list):
        vocab_dictionary[word] = i
        # If the word is in the pre-trained word embedding model, add word embedding to the table
        if word in EMBEDDING_MODEL:
            embedding_table.append(EMBEDDING_MODEL[word])
        else:
            embedding_table.append([0] * embedding_dim)
    embedding_table = np.array(embedding_table)

    return vocab_dictionary, embedding_table


vocab_dict, emb_table = create_word_model(whole_caption)


def tokenizer(caption):
    """
    Tokenizes captions by converting words into corresponding indices based on the vocabulary dictionary.

    :param caption: list of captions, where each caption is a list of words
    :return: list of tokenized captions, where each caption is a list of word indices
    """
    tokenize = []
    # Extract main feature words from captions
    caption = caption_extract(caption)

    for item in caption:
        # Convert each word to its corresponding index in the vocabulary dictionary
        # If a word is not present in the dictionary, use the index for the [UNKNOWN] token
        temp = [vocab_dict[word] if word in vocab_dict else vocab_dict['[UNKNOWN]'] for word in item]
        if len(temp) < max_caption_len:
            # If the caption is shorter than the maximum length
            # Pad the caption with [PAD] tokens to make it of maximum length
            temp += [vocab_dict['[PAD]']] * (max_caption_len - len(temp))
        else:
            # Truncate the caption if it exceeds the maximum length
            temp = temp[:max_caption_len]
        tokenize.append(temp)

    return tokenize


class DataLoad(torch.utils.data.Dataset):
    """
    Custom dataset class for loading data, including images and captions.

    Args:
        data_file (str): path to the data file
        image (str): directory containing the images
        transform (object): optional image transformation to be applied
        text_csv (bool): whether the data file includes caption text in a separate CSV file
    """

    def __init__(self, data_file, image, transform=None, text_csv=None):
        """
        Initialize the dataset.

        :param data_file: path to the data file
        :param image: directory containing the images
        :param transform: optional image transformation to be applied
        :param text_csv: whether the data file includes caption text in a separate CSV file
        """
        self.image_dir = image
        self.text_csv = text_csv
        self.transform = transform
        self.classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19]  # List of classes

        with open(data_file) as data_file:
            # Remove the 'Caption' column and join the 'Caption' values after removing quotes
            lines = [re.sub(r'([^,])"(\s*[^\n])', r'\1/"\2', line) for line in data_file]
            dataframe = pd.read_csv(StringIO(''.join(lines)), escapechar="/")
            self.dataframe = dataframe.drop(columns='Caption').join(dataframe['Caption'].str.replace('\"', ''))

    def __len__(self):
        """
        Get the total number of samples in the dataset.

        :return: total number of samples
        """
        return self.dataframe.shape[0]

    def __getitem__(self, item):
        """
        Get a specific sample from the dataset.

        :param item: index of the sample
        :return: dictionary containing the image, label, image ID, and caption
        """
        if torch.is_tensor(item):
            item.to_list()

        # Combine the image path with the image file name
        img_path = os.path.join(self.image_dir, self.dataframe.iloc[item, 0])
        img = io.imread(img_path)
        img_id = self.dataframe.iloc[item, 0]

        if not self.text_csv:
            img_caption = self.dataframe.iloc[item, 2]
            # Get the image labels and split them
            img_label = self.dataframe.iloc[item, 1].split(' ')
            # Convert labels to integers
            img_label = [int(x) for x in img_label]

            for i in range(len(img_label)):
                # One-hot encode and sum the image labels
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
                                shear=None,
                                fill=tuple(np.array(np.array([0.485, 0.456, 0.406]) * 255).astype(int).tolist())),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

dataset = DataLoad(data_file=TRAIN_CSV, image=IMAGE_DIR, transform=transforms['train'], text_csv=False)
test_dataset = DataLoad(data_file=TEST_CSV, image=IMAGE_DIR, transform=transforms['val'], text_csv=True)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(TRAIN_VAL_PROP * len(dataset)),
                                                                     len(dataset) - (
                                                                         int(TRAIN_VAL_PROP * len(dataset)))])

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=8, shuffle=False, num_workers=0)


class Model(nn.Module):
    """
    The model for the image and natural language classification.
    The image classifier uses the pre-trained EFFICIENTNET_B1 with the pre-trained weights.
    The natural language classifier uses the LSTM as the model process.

    Args:
        emb_table (np.ndarray): pretrained word embedding table

    Attributes:
        efficient_net (nn.Module): EfficientNet-B1 model as the feature extractor
        emb (nn.Embedding): embedding layer, loads the pre-trained embedding table
        lstm (nn.LSTM): LSTM layer for processing text sequences
        linear (nn.Linear): linear layer for linear transformation of LSTM output
        classifier (nn.Linear): classifier layer for the final classification task
    """

    def __init__(self):
        super(Model, self).__init__()
        # Use EfficientNet-B1 model as the feature extractor
        weights = models.EfficientNet_B1_Weights.DEFAULT
        self.efficient_net = models.efficientnet_b1(weights=weights)
        # Modify the classifier layer to have an output dimension of 64
        self.efficient_net.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=self.efficient_net.classifier[1].in_features, out_features=64)
        )
        # Freeze the parameters of EfficientNet and only train the classifier layer
        for param in self.efficient_net.parameters():
            param.requires_grad = False
        for param in self.efficient_net.classifier.parameters():
            param.requires_grad = True

        # Create an embedding layer and load the pre-trained embedding table
        self.emb = nn.Embedding(num_embeddings=emb_table.shape[0], embedding_dim=emb_table.shape[1])
        self.emb.weight.data.copy_(torch.from_numpy(emb_table))
        self.emb.weight.requires_grad = False
        # Create an LSTM layer
        self.lstm = nn.LSTM(input_size=emb_table.shape[1], hidden_size=emb_table.shape[1], num_layers=2,
                            batch_first=True, dropout=0.2)
        # Create a linear layer for linear transformation of LSTM output
        self.linear = nn.Linear(in_features=emb_table.shape[1] * 2, out_features=64)

        # Create a classifier layer with an output dimension of num classes
        self.classifier = nn.Linear(in_features=128, out_features=NUM_CLASS)

    def forward(self, image, caption):
        """
        Forward propagation of the model.

        :param image: input image tensor
        :param caption: input text sequence tensor
        :return: output tensor of the model
        """
        net_out = self.efficient_net(image)

        x = self.emb(caption)
        # Pass the embedded caption through the LSTM layer and get the final hidden states (h) and cell states (c)
        _, (h, c) = self.lstm(x)
        # Concatenate the final hidden states from both directions of the LSTM
        x = torch.cat((h[0, :, :], h[1, :, :]), 1)
        lstm_out = self.linear(x)

        # Concatenate image features and text features, and pass through the classifier layer for classification
        output = self.classifier(torch.cat((net_out, lstm_out), 1))

        return output


net_model = Model().to(device)


def model_size(model):
    """
    Calculate the size of the model with its word embeddings.

    :param model: initialized model
    :return: None
    """
    cnn_size = 0
    # Iterate over the named parameters of the model
    for name, param in model.named_parameters():
        # Calculate the size of the parameter in megabytes
        param_size = np.prod(list(param.shape)) * 4 / 1e6
        cnn_size += param_size

    # Calculate the embedding model size
    embedding_model_path = gensim.downloader.load('glove-wiki-gigaword-50', return_path=True)
    embedding_size = os.stat(embedding_model_path).st_size / (1024 * 1024)

    print("Model size: {:4f} MB; Word Embedding Size: {:4f} MB".format(cnn_size, embedding_size))
    print("Total size: {:4f} MB".format((cnn_size + embedding_size)))
    if (cnn_size + embedding_size) > 100:
        raise ValueError("Model too large!")


model_size(net_model)


def get_acc(output, label):
    """
    Calculate precision, recall, and F1 score for model predictions.

    :param output: predicted output from the model
    :param label: ground truth labels
    :return: micro precision score, recall score and f1 score
    """
    # Convert the predicted output and labels to numpy arrays
    output = output.cpu().detach().numpy()
    label = label.cpu().numpy()
    predicted_labels = []
    for preds in output:
        # If there is no sigmoid output larger than threshold, set the second element as the predict label
        if sum(preds > THRESHOLD) == 0:
            temp = np.zeros(18)
            temp[np.argmax(preds[1])] = 1
            predicted_labels.append(temp)
        # If there is at least one sigmoid output larger than threshold
        else:
            predicted_labels.append(np.array(preds > THRESHOLD, dtype=float))
    pred = np.array(predicted_labels, dtype=float)

    precision = precision_score(y_true=label, y_pred=pred, average='micro')
    recall = recall_score(y_true=label, y_pred=pred, average='micro')
    f1 = f1_score(y_true=label, y_pred=pred, average='micro')

    return 100 * precision, 100 * recall, 100 * f1


def train(model, dataloader, loss_fn, opt):
    """
    Train the model on the training dataset.

    :param model: model to be trained
    :param dataloader: DataLoader providing the training dataset
    :param loss_fn: loss function used for optimization
    :param opt: optimizer for updating the model's parameters
    :return: training loss, precision, recall, F1-score, and the number of processed batches
    """
    train_loss = 0
    train_precision = 0
    train_recall = 0
    train_f1 = 0
    batch_size = 0
    batch_num = 0
    # Set the model to train mode
    model.train()

    # batch item: img, label, id, caption
    for batch in tqdm(dataloader):
        # Caption: input_ids, token_type_ids, attention_mask
        img, label, caption = batch["img"].to(device), batch["label"].to(device), torch.from_numpy(
            np.array(tokenizer(batch["caption"]))).to(device)
        # Zero the gradients
        opt.zero_grad()
        output = model(img, caption)
        loss = loss_fn(output, label)
        # Backward pass and optimization step
        loss.backward()
        opt.step()

        precision, recall, f1 = get_acc(nn.Sigmoid()(output), label)
        batch_num += 1
        batch_size += 1
        # Accumulate the batch loss scaled by the batch size
        train_loss += loss.item() * img.size(0)
        train_precision += precision
        train_recall += recall
        train_f1 += f1
        print('\rBatch[{}/{}] - loss: {:.6f}  precision: {:.4f}%  recall: {:.4f}%  F1_score: {:.4f}%'.format(batch_size,
                                                                                                             len(dataloader),
                                                                                                             loss.item() * img.size(
                                                                                                                 0),
                                                                                                             precision,
                                                                                                             recall,
                                                                                                             f1))

    return train_loss, train_precision, train_recall, train_f1, batch_num


def evaluate(model, dataloader, loss_fn):
    """
    Evaluate the performance of the model on the validation set.

    :param model: model to evaluate
    :param dataloader: data loader for the validation set
    :param loss_fn: loss function used for evaluation
    :return: total validation loss, precision, recall, F1 score, and the number of batches
    """
    val_loss = 0
    val_precision = 0
    val_recall = 0
    val_f1 = 0
    batch_num = 0
    # Set the model to evaluation mode
    model.eval()

    # Disable gradient computation
    with torch.no_grad():
        for batch in tqdm(dataloader):
            img, label, caption = batch["img"].to(device), batch["label"].to(device), torch.from_numpy(
                np.array(tokenizer(batch["caption"]))).to(device)
            output = model(img, caption)
            loss = loss_fn(output, label)

            precision, recall, f1 = get_acc(nn.Sigmoid()(output), label)
            batch_num += 1
            val_loss += loss.item() * img.size(0)
            val_precision += precision
            val_recall += recall
            val_f1 += f1

    return val_loss, val_precision, val_recall, val_f1, batch_num


def fit(model, train_dataloader, val_dataloader, epochs, loss_fn, opt, shed):
    """
    Train and evaluate model.

    :param model: model to train
    :param train_dataloader: dataloader for the training data
    :param val_dataloader: dataloader for the validation data
    :param epochs: number of epochs to train
    :param loss_fn: loss function
    :param opt: optimizer
    :param shed: scheduler for adjusting learning rate
    :return: training and validation metrics
    """
    # Record the start time
    start_time = time.time()

    tr_loss_all = np.zeros(epochs)
    tr_precision_all = np.zeros(epochs)
    tr_recall_all = np.zeros(epochs)
    tr_f1_all = np.zeros(epochs)
    vl_loss_all = np.zeros(epochs)
    vl_precision_all = np.zeros(epochs)
    vl_recall_all = np.zeros(epochs)
    vl_f1_all = np.zeros(epochs)

    best_f1 = 0

    for epoch in range(epochs):
        train_loss, train_precision, train_recall, train_f1, train_batch_num = train(model, train_dataloader, loss_fn,
                                                                                     opt)
        val_loss, val_precision, val_recall, val_f1, val_batch_num = evaluate(model, val_dataloader, loss_fn)
        print('Epoch [{}/{}]'.format(epoch, epochs))
        print('Train - loss: {:.6f}  precision: {:.4f}%  recall: {:.4f}%  F1_score: {:.4f}%'.format(
            train_loss / len(train_dataset), train_precision / train_batch_num, train_recall / train_batch_num,
            train_f1 / train_batch_num))
        print('Val - loss: {:.6f}  precision: {:.4f}%  recall: {:.4f}%  F1_score: {:.4f}%'.format(
            val_loss / len(val_dataset), val_precision / val_batch_num, val_recall / val_batch_num,
            val_f1 / val_batch_num))
        print('-' * 19)

        # Store metrics for the current epoch
        tr_loss_all[epoch] = train_loss / len(train_dataset)
        tr_precision_all[epoch] = train_precision / train_batch_num
        tr_recall_all[epoch] = train_recall / train_batch_num
        tr_f1_all[epoch] = train_f1 / train_batch_num
        vl_loss_all[epoch] = val_loss / len(val_dataset)
        vl_precision_all[epoch] = train_precision / val_batch_num
        vl_recall_all[epoch] = val_recall / val_batch_num
        vl_f1_all[epoch] = val_f1 / val_batch_num

        # Adjust learning rate using the scheduler
        shed.step(train_loss / len(train_dataset))

        # Save the model with the best F1 score so far
        if val_f1 / val_batch_num > best_f1:
            best_f1 = val_f1 / val_batch_num
            print('Saving best pre_model, f1: {:.4f}%\n'.format(best_f1))
            torch.save(model.state_dict(), 'best_steps_{}.pth'.format(epoch))

    # Calculate the duration of training
    duration = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))

    return tr_loss_all, tr_precision_all, tr_recall_all, tr_f1_all, vl_loss_all, vl_precision_all, vl_recall_all, vl_f1_all


class FocalLoss(nn.Module):
    """
    Focal loss implementation for binary classification tasks.

    Args:
        gamma (float): parameter controls the shape of the focal loss function, default is 2
        alpha (float): balancing factor used to adjust the weight between positive and negative samples, default is 0.25
    """

    def __init__(self, gamma=2, alpha=0.25):
        """
        Initialize the FocalLoss module.

        :param gamma: parameter controls the shape of the focal loss function, default is 2
        :param alpha: balancing factor used to adjust the weight between positive and negative samples, default is 0.25
        """
        super().__init__()
        # Wraps focal loss around existing loss_fcn()
        self.loss_fcn = nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = 'sum'
        # required to apply FL to each element
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        """
        Compute the focal loss.

        :param pred: predicted logits
        :param true: true labels
        :return: focal loss
        """
        # Calculate the loss
        loss = self.loss_fcn(pred, true)
        # Calculate probability from logits
        pred_prob = torch.sigmoid(pred)
        # Calculate the modulating factor
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        # Calculate the alpha factor
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


optimizer = torch.optim.Adam(net_model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
    threshold=0.0001,
    threshold_mode="rel",
    cooldown=0,
    min_lr=0,
    eps=1e-08,
    verbose=False,
)
loss_function = nn.BCEWithLogitsLoss()
# loss_function = FocalLoss()


tr_loss, tr_precision, tr_recall, tr_f1, vl_loss, vl_precision, vl_recall, vl_f1 = fit(net_model, train_loader,
                                                                                       val_loader, MAX_EPOCH,
                                                                                       loss_function, optimizer,
                                                                                       scheduler)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(tr_loss, label='training_Loss')
plt.plot(vl_loss, label='validation_Loss')
plt.legend()
plt.show()

plt.xlabel("Epoch")
plt.ylabel("Precision")
plt.plot(tr_precision, label='training_Precision')
plt.plot(vl_precision, label='validation_Precision')
plt.legend()
plt.show()

plt.xlabel("Epoch")
plt.ylabel("Recall")
plt.plot(tr_recall, label='training_Recall')
plt.plot(vl_recall, label='validation_Recall')
plt.legend()
plt.show()

plt.xlabel("Epoch")
plt.ylabel("Micro-F1")
plt.plot(tr_f1, label='training_F1')
plt.plot(vl_f1, label='validation_F1')
plt.legend()
plt.show()


def predict(model, dataloader):
    """
    Perform predictions using the best trained model.

    :param model: trained model
    :param dataloader: data loader for prediction
    :return: list of image IDs, predicted outputs
    """
    img_id = []
    pred = torch.tensor([])
    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader):
            img, caption = batch["img"].to(device), torch.from_numpy(np.array(tokenizer(batch["caption"]))).to(device)
            # Perform forward pass and apply sigmoid activation
            output = nn.Sigmoid()(model(img, caption))
            img_id.extend(batch["id"])
            # Concatenate predicted outputs along the batch dimension
            pred = torch.cat([pred, output.cpu()], dim=0)
    # Save predictions to a file
    out_to_file(img_id, pred)

    return img_id, pred


def out_to_file(img_id, prediction):
    """
    Save predictions to a CSV file.

    :param img_id: list of image IDs
    :param prediction: predicted outputs
    :return: None
    """
    label = []
    for i, pred in enumerate(prediction):
        prediction = []
        # Apply thresholding to convert predictions to binary values
        pred = pred > THRESHOLD
        for idx in range(len(pred)):
            if pred[idx]:
                if idx > 10:
                    # Append the label index (+2) to the prediction list if the index is less than 12
                    prediction.append(idx + 2)
                else:
                    # Append the label index (+1) to the prediction list if the index is larger than 12
                    prediction.append(idx + 1)
        # Convert the prediction list to a string
        result = ' '.join(str(e) for e in prediction)
        label.append(result)
    for n in range(len(img_id)):
        img_id[n] = str(img_id[n])
        # Append the file extension to the image ID
        img_id[n] = img_id[n] + '.jpg'
    # Save the DataFrame to a CSV file
    df = pd.DataFrame({'ImageID': img_id, 'Label': label})
    df.to_csv("result.csv", index=False)
