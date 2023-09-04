import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.emb = nn.Embedding(num_embeddings=emb_table.shape[0], embedding_dim=emb_table.shape[1])
        # Initialize the Embedding layer with the lookup table we created
        self.emb.weight.data.copy_(torch.from_numpy(emb_table))
        # make this lookup table untrainable
        self.emb.weight.requires_grad = False
        self.lstm = nn.LSTM(input_size=emb_table.shape[1], hidden_size=300, num_layers=2, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(in_features=300*2, out_features=18)
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
