import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoTokenizer, AutoModelForSequenceClassification


tokenizers = AutoTokenizer.from_pretrained("prajjwal1/bert-mini")


class CombineModel(nn.Module):
    def __init__(self):
        super(CombineModel, self).__init__()
        self.efficient_net = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        self.efficient_net.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=self.efficient_net.classifier[1].in_features, out_features=19)
        )
        self.sig = nn.Sigmoid()

        self.nlp_net = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-mini")
        self.pre_classifier = nn.Linear(in_features=2, out_features=64)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(in_features=64, out_features=19)

        self.linear = nn.Linear(in_features=38, out_features=19)

    def forward(self, x, input_ids=None, attention_mask=None):
        cnn_out = self.sig(self.efficient_net(x))

        nlp_out = self.nlp_net(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = nlp_out[0]
        out = hidden_state[:, 0]
        out = self.pre_classifier(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.classifier(out)

        output = torch.cat((out, cnn_out), dim=1)
        return self.sig(self.linear(output))


