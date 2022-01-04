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

class LinRegModel(nn.Module):
    def __init__(self, in_dim):
        super(LinRegModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=1),
            nn.Sigmoid()
        )
        torch.nn.init.xavier_normal_(self.layers[0].weight.data)

    def forward(self, samples):
        return self.layers(samples).squeeze(1)

class ReviewDataset(nn.Module):
    def __init__(self, data, labels):
        super(ReviewDataset, self).__init__()
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, item):
        return self.data[item], self.labels[item]

train_data, train_labels = read_dataset("aclImdb/train")
test_data, test_labels = read_dataset("aclImdb/test")

def get_dataloaders(tf_idf_thresh, batch_size):
    vectorizer = TfidfVectorizer(ngram_range=(1,3), max_features=tf_idf_thresh)
    train_vector = torch.from_numpy(vectorizer.fit_transform(train_data).toarray()).float()
    test_vector = torch.from_numpy(vectorizer.transform(test_data).toarray()).float()

    train_dataset = ReviewDataset(train_vector, torch.from_numpy(train_labels).float())
    test_dataset = ReviewDataset(test_vector, torch.from_numpy(test_labels).float())
    train_loader =  DataLoader(dataset = train_dataset, shuffle = True, batch_size = batch_size)
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size)

    return train_loader, test_loader

train_options = {
    "epochs": 1000,
    "load_model": True,
    "save_model": True,
    "batch_size": 256,
    "lr_rate": 0.001,
    "train_model": True,
    "tf_threshold": 1000,
    "model_path": "saved_models/lin_reg_emb"
}

def check_val_accuracy(model, loader):
    total_correct = 0
    model.eval()
    fp = 0
    fn = 0
    tp = 0
    with torch.no_grad():
        for X, Y in loader:
            Y_pred = model(X.to(device)).cpu()
            total_correct += ((Y_pred > 0.5) == Y).sum()

            Y_pred_label = Y_pred > 0.5
            tp += (Y_pred_label[Y == 1] == 1).sum()
            fp += (Y_pred_label[Y == 0] == 1).sum()
            fn += (Y_pred_label[Y == 1] == 0).sum()

        precision = tp / (fp + tp)
        recall = tp / (tp + fn)
        print("Precision: " + str(precision))
        print("Recall: " + str(recall))
        print("F1 Score: " + str(2 * precision * recall / (precision + recall)))

    return total_correct / loader.dataset.__len__()

def train_torch_model(train_options):
    train_loader, test_loader = get_dataloaders(train_options["tf_threshold"], train_options["batch_size"])

    model = LinRegModel(train_options["tf_threshold"])
    model.to(device)
    max_acc = check_val_accuracy(model, test_loader)
    optimizer = Adam(params= model.parameters(), lr=train_options["lr_rate"])

    loss_fn = nn.BCELoss()
    no_max_count = 0
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
        if max_acc < acc:
            no_max_count = 0
            max_acc = acc
        else:
            no_max_count += 1
        if no_max_count > 30:
            break

    print("Test set scores")
    acc = check_val_accuracy(model, test_loader)

    print("Test Accuracy: " + str(acc))
    print("------------------------")
    print("Train set scores")
    acc = check_val_accuracy(model, train_loader)
    print("Train accuracy: " + str(acc))


train_torch_model(train_options)
