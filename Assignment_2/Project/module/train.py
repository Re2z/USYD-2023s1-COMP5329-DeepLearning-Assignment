import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from Project.pre_model.efficient_model import CnnModel
from Project.pre_model.nlp_model import LSTM
from sklearn.metrics import f1_score, precision_score, recall_score

writer = SummaryWriter("logs")
device = "cuda" if torch.cuda.is_available() else "cpu"

LR = 0.0001
MAX_EPOCH = 100
THRESHOLD = 0.5

classifier = nn.Sequential(
    nn.Linear(in_features=36, out_features=2392),
    nn.ReLU(),
    nn.Linear(in_features=2392, out_features=18),
    nn.Sigmoid()
)

cnn_model = CnnModel().to(device)
nlp_model = LSTM().to(device)
classifier = classifier.to(device)

optimizer = torch.optim.AdamW(classifier.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
loss_function = nn.BCELoss()


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

        writer.add_scalar("train_loss", train_loss / len(train_dataset), epoch)
        writer.add_scalar("train_f1", train_f1 / train_batch_num, epoch)
        writer.add_scalar("val_loss", val_loss / len(val_dataset), epoch)
        writer.add_scalar("val_f1", val_f1 / val_batch_num, epoch)

        if val_f1 / val_batch_num > best_f1:
            best_f1 = val_f1 / val_batch_num
            print('Saving best pre_model, f1: {:.4f}%\n'.format(best_f1))
            classifier.eval()
            torch.save(classifier, 'best_steps_{}.pt'.format(epoch))
            classifier.train()

    duration = time.time() - start_time
    writer.close()
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
