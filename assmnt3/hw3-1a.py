import os

import numpy as np


# CODE TAKEN FROM DEMO PROJECT -- START
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


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
            neg_data.append(fp.read().lower())

    for root, dirs, files in os.walk(file_path + "/pos"):
        for file_name in files:
            fp = open(os.path.join(root, file_name), encoding="utf-8", errors="ignore")
            pos_data.append(fp.read().lower())

    neg_labels = np.repeat(0, len(neg_data))
    pos_labels = np.repeat(1, len(pos_data))
    labels = np.concatenate([neg_labels, pos_labels])
    data = neg_data + pos_data
    return data, labels

# CODE TAKEN FROM DEMO PROJECT -- END

def get_scores(target, pred):
    tp = (target[pred == 1] == 1).sum()
    fp = (pred[target == 0] == 1).sum()
    fn = (pred[target == 1] == 0).sum()
    precision = tp / (fp + tp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)

    return precision, recall, f1_score

def train_test_classifier():
    train_data, train_labels = read_dataset("aclImdb/train")
    test_data, test_labels = read_dataset("aclImdb/test")

    print("Reading complete.")

    tf_idf_thresh = (1000, 5000, 10000)

    for thresh in tf_idf_thresh:
        vectorizer = TfidfVectorizer(ngram_range=(1,3), max_features=thresh)
        train_vector = vectorizer.fit_transform(train_data)
        test_vector = vectorizer.transform(test_data)

        classifier = LogisticRegression(random_state=42)
        classifier.fit(train_vector, train_labels)

        test_pred = classifier.predict(test_vector)

        print("Threshold:" + str(thresh))
        print("Accuracy: " + str((test_pred == test_labels).sum()/len(test_pred)))
        precision, recall, f1_score = get_scores(test_labels, test_pred)
        print("TEST: precision: " + str(precision) + ", recall: " + str(recall) + ", f1_score: " + str(f1_score))

        train_pred = classifier.predict(train_vector)
        precision, recall, f1_score = get_scores(train_labels, train_pred)
        print("Train: precision: " + str(precision) + ", recall: " + str(recall) + ", f1_score: " + str(f1_score))
        print("----------------------------")

train_test_classifier()
