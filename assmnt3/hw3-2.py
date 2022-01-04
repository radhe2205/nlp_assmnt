import json
import os
import random

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def read_data(file_path):
    with open(file_path, "r") as f:
        data = f.read()
    all_chars = set()
    data = data
    for ch in data:
        all_chars.add(ch)

    chartoidx = {ch: i for i, ch in enumerate(all_chars)}
    idxtochar = {chartoidx[ch]: ch for ch in chartoidx}
    return data, chartoidx, idxtochar

class CharModel(nn.Module):
    def __init__(self, num_chars, chartoidx, idxtochar):
        super(CharModel, self).__init__()
        self.num_chars = num_chars
        self.chtoidx = chartoidx
        self.idxtoch = idxtochar
        self.num_layers = 5
        self.hidden_size = num_chars * 5
        self.rnn_layers = nn.LSTM(input_size=num_chars, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=num_chars),
            nn.Softmax(dim=-1)
        )

    def forward(self, samples): # N x seq_len x num_chars
        o, (h, c) = self.rnn_layers(samples) # N x seq_len x 2 * num_chars
        return self.linear_layer(o)

    def generate_code(self):
        self.eval()
        with torch.no_grad():
            first_char = "t"
            last_char = torch.zeros(1, self.num_chars).to(device)
            last_char[0, self.chtoidx[first_char]] = 1
            generated_txt = "It is a sleepy language and thou speak'st".lower()

            inp = torch.zeros(len(generated_txt), self.num_chars).to(device)
            for i, ch in enumerate(generated_txt[:-1]):
                inp[i, self.chtoidx[ch]] = 1.

            o, (h, c) = self.rnn_layers(inp.unsqueeze(0))
            h = h[:, -1, :].unsqueeze(1).contiguous()
            c = c[:, -1, :].unsqueeze(1).contiguous()

            for i in range(1000):
                o, (h, c) = self.rnn_layers(last_char.unsqueeze(0), (h, c))
                probs = self.linear_layer(o.view(-1).unsqueeze(0)).view(-1)

                probs = torch.softmax(probs * 4, dim=-1)

                sample = random.random()
                char_idx = 0
                total_prob = 0

                for j, prob in enumerate(probs):
                    if total_prob <= sample and total_prob + prob > sample:
                        char_idx = j
                        break
                    total_prob += prob
                generated_txt += self.idxtoch[char_idx]
                last_char = torch.zeros(1, self.num_chars).to(device)
                last_char[0, char_idx] = 1.
        return generated_txt


class CharDataset(nn.Module):
    def __init__(self, data, chartoidx, seq_len = 50):
        super(CharDataset, self).__init__()
        self.data = data
        self.seq_len = seq_len
        self.chartoidx = chartoidx
        self.num_chars = len(chartoidx.keys())

    def __len__(self):
        return int(len(self.data) / self.seq_len - 2)

    def __getitem__(self, item):
        givens = self.data[item * self.seq_len: item*self.seq_len + self.seq_len]
        targets = self.data[item * self.seq_len + 1: item*self.seq_len + self.seq_len + 1]
        given_vec = torch.zeros(self.seq_len, self.num_chars)
        target_vec = torch.zeros(self.seq_len).long()

        for i in range(self.seq_len):
            given_vec[i, self.chartoidx[givens[i]]] = 1.
            target_vec[i] = self.chartoidx[targets[i]]
        return given_vec, target_vec

def get_dataloader(train_ops):
    data, chtoidx, idxtoch = read_data(train_ops["data_path"])
    # return data, chtoidx, idxtoch, len(chtoidx.keys())
    if train_ops["load_vocab"]:
        chtoidx, idxtoch = load_vocab(train_ops["vocab_path"])
    save_vocab(chtoidx, train_ops["vocab_path"])

    dataset = CharDataset(data, chtoidx, train_ops["seq_len"])
    dataloader = DataLoader(dataset = dataset, batch_size=train_ops["batch_size"], shuffle=True)

    return dataloader, chtoidx, idxtoch, len(chtoidx.keys())

def save_vocab(vocab, path):
    with open(path, "w") as f:
        f.write(json.dumps(vocab))

def load_vocab(path):
    with open(path, "r") as f:
        vocab = f.read()
        vocab = json.loads(vocab)

    wordtoidx = {}
    idxtoword = {}
    for i, k in enumerate(vocab):
        wordtoidx[k] = i
        idxtoword[i] = k

    return wordtoidx, idxtoword

def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)
    return model

def load_model(model, file_path):
    if os.path.exists(file_path):
        model.load_state_dict(torch.load(file_path, map_location=device))
    return model

def train_model(train_options):
    dataloader, chtoidx, idxtoch, num_chars = get_dataloader(train_options)
    model = CharModel(num_chars, chtoidx, idxtoch)
    model.to(device)

    if train_options["load_model"]:
        load_model(model, train_options["model_path"])
        print(model.generate_code())

    if not train_options["train_model"]:
        return model

    min_loss = 1000
    optimizer = Adam(params=model.parameters(), lr=train_options["lr_rate"])

    loss_fn = nn.CrossEntropyLoss()

    for epoch_num in range(train_options["epochs"]):
        print("Epoch:" + str(epoch_num + 1))

        total_loss = 0
        total_correct = 0
        model.train()
        for X, Y in dataloader:
            X = X.to(device)
            Y_pred = model(X).cpu()
            loss = loss_fn(Y_pred.view(-1, num_chars), Y.view(-1))

            total_correct += (Y_pred.argmax(dim = -1) == Y).sum()

            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("total_correct: " + str(total_correct) + ", out of: " + str(dataloader.dataset.__len__() * train_options["seq_len"]))
        print("Loss:" + str(total_loss))
        print("------------------------")

        print(model.generate_code())

        if min_loss > total_loss:
            min_loss = total_loss
            print("saving model.")
            save_model(model, train_options["model_path"])

    return model

train_options = {
    "epochs": 100,
    "load_model": True,
    "load_vocab": True,
    "train_model": True,
    "save_model": False,
    "batch_size": 256,
    "lr_rate": 0.001,
    "seq_len": 200,
    "model_path": "saved_models/rnn_eff/rnn_shakes_upper.model",
    "vocab_path": "saved_models/rnn_eff/vocab.json",
    "data_path": "texts/poems.txt"
}

train_model(train_options)
