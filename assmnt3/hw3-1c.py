import json
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.optim import Adam
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def read_dataset(file_path):
    """
    File_path should be a string that represents the filepath
    where the movie dataset can be found

    This returns an array of strings and an array of labels
    """
    neg_data = []
    pos_data = []
    for root, dirs, files in os.walk(file_path + "/neg"):
        for file_name in files:
            fp = open(os.path.join(root, file_name), encoding="utf-8", errors="ignore")
            neg_data.append(fp.read())

    for root, dirs, files in os.walk(file_path + "/pos"):
        for file_name in files:
            fp = open(os.path.join(root, file_name), encoding="utf-8", errors="ignore")
            pos_data.append(fp.read())

    neg_labels = np.repeat(0, len(neg_data))
    pos_labels = np.repeat(1, len(pos_data))
    labels = np.concatenate([neg_labels, pos_labels])
    data = neg_data + pos_data
    return data, labels

def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)
    return model

def load_model(model, file_path):
    if os.path.exists(file_path):
        model.load_state_dict(torch.load(file_path))
    return model


class LinRegModel(nn.Module):
    def __init__(self, embedding_dim, total_embeddings, vec_strategy = "addition"):
        super(LinRegModel, self).__init__()
        self.vec_strategy = vec_strategy
        self.embedding_dim = embedding_dim
        self.layers = nn.Sequential(
            nn.Linear(in_features=(embedding_dim, embedding_dim * 180)[["addition", "concate"].index(vec_strategy)], out_features=1),
            nn.Sigmoid()
        )
        torch.nn.init.xavier_normal_(self.layers[0].weight.data)
        self.embeddings = nn.Embedding(total_embeddings, embedding_dim)
        nn.init.uniform_(self.embeddings.weight, -1, 1)

    def get_embeddings(self, samples):
        if self.vec_strategy == "addition":
            sample_vec = torch.zeros(len(samples), self.embedding_dim).to(device)
            for idx, sample in enumerate(samples):
                sample = sample[sample != -1]
                sample_vec[idx] = nn.functional.normalize(self.embeddings.weight[sample.type(torch.LongTensor)].sum(dim=0), dim = 0)
            return sample_vec
        return torch.zeros(len(samples), self.embedding_dim).to(device)

    def forward(self, samples):
        sample_vec = self.get_embeddings(samples)
        return self.layers(sample_vec).squeeze(1)

train_options = {
    "epochs": 40,
    "load_model": True,
    "load_vocab": True,
    "train_model": True,
    "save_model": True,
    "batch_size": 64,
    "lr_rate": 0.001,
    "tf_threshold": 1000,
    "embedding_dim": 50,
    "model_path": "saved_models/lin_reg_emb",
    "vocab_path": "saved_models/vocab.json"
}

train_data, train_labels = read_dataset("aclImdb/train")
test_data, test_labels = read_dataset("aclImdb/test")

def create_vocab(data):
    all_words = {}
    for line in data:
        # line = "".join([char if char.isalnum() else " " for char in line])
        line = line.lower()
        words = line.split()
        for word in words:
            if word not in all_words:
                all_words[word] = 0
            all_words[word] += 1

    all_uniq_words = set()

    for k in all_words:
        if all_words[k] > 1:
            all_uniq_words.add(k)

    wordtoidx = {word: i for i, word in enumerate(all_uniq_words)}
    wordtoidx["<unk>"] = len(all_uniq_words)
    idxtoword = {i: word for i, word in enumerate(all_uniq_words)}
    idxtoword[len(all_uniq_words)] = "<unk>"
    all_uniq_words.add("<unk>")
    return wordtoidx, idxtoword, len(all_uniq_words)

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

    return wordtoidx, idxtoword, len(wordtoidx.keys())

def get_vector(wordtoidx, data):
    data = data.lower()
    # data = "".join([char if char.isalnum() else " "  for char in data])
    words = data.split()
    emb_idx = torch.zeros(len(words)).type(torch.LongTensor)
    for i, word in enumerate(words):
        if word in wordtoidx:
            emb_idx[i] = wordtoidx[word]
        else:
            emb_idx[i] = wordtoidx["<unk>"]
    return emb_idx

def get_vector_from_dataset(dataset, wordtoidx):
    vectors = []
    for data in dataset:
        vectors.append(get_vector(wordtoidx, data))
    return vectors

def get_dataloader(batch_size = 512, load_path = None, load_vocab_from_file = False):
    if load_vocab_from_file:
        wordtoidx, idxtoword, total_words = load_vocab(load_path)
    else:
        wordtoidx, idxtoword, total_words = create_vocab(train_data)
        save_vocab(wordtoidx, load_path)

    train_vectors = get_vector_from_dataset(train_data, wordtoidx)
    test_vectors = get_vector_from_dataset(test_data, wordtoidx)

    print(len(train_vectors))

    train_dataset = ReviewDatasetEmb(wordtoidx, idxtoword, train_vectors, torch.from_numpy(train_labels).float())
    test_dataset = ReviewDatasetEmb(wordtoidx, idxtoword, test_vectors, torch.from_numpy(test_labels).float())

    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size)

    return train_loader, test_loader, wordtoidx, idxtoword, total_words

class ReviewDatasetEmb(nn.Module):
    def __init__(self, wordtoidx, idxtoword, data, labels):
        super(ReviewDatasetEmb, self).__init__()
        self.wordtoidx = wordtoidx
        self.idxtoword = idxtoword
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        vec = torch.cat((self.data[item], torch.empty(5000-len(self.data[item])).fill_(-1.)), dim=0)
        return vec, self.labels[item]

def check_val_accuracy(model, loader):
    total_correct = 0
    model.eval()
    fp = 0
    fn = 0
    tp = 0
    with torch.no_grad():
        for X, Y in loader:
            Y_pred = model(X.to(device)).cpu()
            Y_pred = Y_pred > 0.5
            total_correct += (Y_pred == Y).sum()

            tp += (Y_pred[Y == 1] == 1).sum()
            fp += (Y_pred[Y == 0] == 1).sum()
            fn += (Y_pred[Y == 1] == 0).sum()

    precision = tp / (fp + tp)
    recall = tp / (tp + fn)
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F1 Score: " + str(2 * precision * recall / (precision + recall)))

    return total_correct / loader.dataset.__len__()

def get_nearest_neighbors(embedding_w, vocab, word, n_neighbors = 6):
    wordtoidx, idxtoword = vocab
    if word not in wordtoidx:
        print("Word not found, changed to <>unk")
        word = "<unk>"
    idx = wordtoidx[word]
    word_emb = embedding_w[idx]

    cos_sim = (embedding_w * word_emb).sum(dim=1) / (torch.norm(embedding_w, dim=1) * torch.norm(word_emb))
    vals, indices = torch.topk(cos_sim, n_neighbors, largest=True)
    all_words = []
    for idx in indices:
        all_words.append(idxtoword[idx.item()])

    return all_words

def get_neighbors(word, embedding_w, wordtoidx, idxtoword, n_neighbors = 6):
    return get_nearest_neighbors(embedding_w, (wordtoidx, idxtoword), word, n_neighbors)

def load_embedding_n_vocab(model_path, vocab_path, embedding_dim = 50):
    wordtoidx, idxtoword, total_words = load_vocab(vocab_path)
    model = LinRegModel(embedding_dim, total_words)
    load_model(model, model_path)
    return model.embeddings, (wordtoidx, idxtoword)

def train_torch_model(train_options):
    train_loader, test_loader, wordtoidx, idxtoword, total_words = get_dataloader(train_options["batch_size"], train_options["vocab_path"], train_options["load_vocab"])

    print("Total words: " + str(total_words))

    model = LinRegModel(train_options["embedding_dim"], total_words)
    model.to(device)

    if train_options["load_model"]:
        load_model(model, train_options["model_path"])

    max_acc = check_val_accuracy(model, test_loader)
    print(max_acc)

    if not train_options["train_model"]:
        return model

    max_acc = check_val_accuracy(model, test_loader)
    min_loss = 1000
    optimizer = Adam(params= model.parameters(), lr=train_options["lr_rate"])

    loss_fn = nn.BCELoss()

    for epoch_num in range(train_options["epochs"]):
        print("Epoch:" + str(epoch_num+1))

        total_loss = 0
        for X,Y in train_loader:
            X = X.to(device)
            Y_pred = model(X).cpu()
            loss = loss_fn(Y_pred, Y)

            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Loss:" + str(total_loss))
        acc = check_val_accuracy(model, test_loader)
        print("Acc:" + str(acc))
        print("------------------------")

        if min_loss > total_loss:
            min_loss = total_loss
            print("saving model.")
            save_model(model, train_options["model_path"])

        print("amazing: " + str(get_neighbors("amazing", model.embeddings.weight, wordtoidx, idxtoword)))
        print("good: " + str(get_neighbors("good", model.embeddings.weight, wordtoidx, idxtoword)))
        print("worst: " + str(get_neighbors("worst", model.embeddings.weight, wordtoidx, idxtoword)))
        print("bad: " + str(get_neighbors("bad", model.embeddings.weight, wordtoidx, idxtoword)))
        print("bored: " + str(get_neighbors("bored", model.embeddings.weight, wordtoidx, idxtoword)))
        print("lack: " + str(get_neighbors("lack", model.embeddings.weight, wordtoidx, idxtoword)))

        print("------------------------")

    return model


train_options["train_model"] = True
train_options["load_vocab"] = True
train_options["load_model"] = True
train_options["epochs"] = 30

model = train_torch_model(train_options)

wordtoidx, idxtoword, total_words = load_vocab(train_options["vocab_path"])

print("amazing: " + str(get_neighbors("amazing", model.embeddings.weight, wordtoidx, idxtoword)))
print("good: " + str(get_neighbors("good", model.embeddings.weight, wordtoidx, idxtoword)))
print("worst: " + str(get_neighbors("worst", model.embeddings.weight, wordtoidx, idxtoword)))
print("bad: " + str(get_neighbors("bad", model.embeddings.weight, wordtoidx, idxtoword)))
print("bored: " + str(get_neighbors("bored", model.embeddings.weight, wordtoidx, idxtoword)))
print("lack: " + str(get_neighbors("lack", model.embeddings.weight, wordtoidx, idxtoword)))
