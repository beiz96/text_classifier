import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import argparse
from model import CNNclass


def parse_args():
    parser = argparse.ArgumentParser(description="CNN for sentence classification")
    # input data path
    parser.add_argument('--train', type=str, default='./topicclass/topicclass_train.txt', help='Training dataset.')
    parser.add_argument('--val', type=str, default='./topicclass/topicclass_valid.txt', help='Validation dataset.')
    parser.add_argument('--test', type=str, default='./topicclass/topicclass_test.txt', help='Test dataset.')
    # embedding params
    parser.add_argument('--embed_dim', type=int, default=100, help='Dimension of word embedding.')
    parser.add_argument('--kernel_num', type=int, default=100, help='Number of kernels used for each filter size.')
    parser.add_argument('--class_num', type=int, default=16, help='Number of classes needed for prediction.')
    parser.add_argument('--kernel_sizes', nargs='?', default='[3,4,5]', help='Kernel sizes')
    # training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout prob.')
    return parser.parse_args()


def load_data(filename):
    mydict = {'tag': list(), 'words': list()}
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            if tag == 'media and darama':
                tag = 'media and drama'
            mydict['tag'].append(tag)
            mydict['words'].append(words)
    data = pd.DataFrame(mydict)  # data is in DataFrame format
    return data


class TextDataset(Dataset):
    def __init__(self, train_data, val_data, test_data, word2idx, idx2word, tag2idx, idx2tag):
        # get train, val, test data
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        # get tag-idx word-idx dicts
        self.tag2idx = tag2idx
        self.idx2tag = idx2tag
        self.word2idx = word2idx
        self.idx2word = idx2word
        # default self.data is train data
        self.data = self.train_data

    def is_train_val_test(self, flag):
        if flag == 'train':
            self.data = self.train_data
        elif flag == 'valid':
            self.data = self.val_data
        else:
            self.data = self.test_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, row_idx):
        row = self.data.iloc[row_idx]
        tag_in_sentence = self.tag2idx.get(row['tag'], -1)  # if no such tag found, then default is -1
        words_in_sentence = row['words'].split(" ")
        # use a list to contain each word index (from word2idx) in a sentence
        sentence_words_idx_list = []
        for w in words_in_sentence:
            sentence_words_idx_list.append(self.word2idx.get(w, 0))
        # since the max length for a sentence is around 60, we pad each sentence to 60 in len
        num_pad = 60 - len(sentence_words_idx_list)
        sentence_words_idx_list = sentence_words_idx_list + num_pad*[1]  # here, 1 is for <PAD> idx
        sentence_words_idx_list = torch.LongTensor(sentence_words_idx_list)
        # note: 'words_sentence' is a tensor of words's indexes for one sentence
        return {'words_sentence': sentence_words_idx_list, 'tag_sentence': tag_in_sentence}


def eval_acc(tag_predict, tag_batch):
    tag_best, tag_idx = tag_predict.max(dim=1)
    total_match = torch.eq(tag_idx, tag_batch).sum().item()
    tag_acc = total_match/len(tag_best)
    return tag_acc


def predict(filename, text_dataset, args, isVal):
    result = []
    # load the best_acc model
    bestCNN = CNNclass(args, train_emb)
    bestCNN = torch.nn.DataParallel(bestCNN)
    bestCNN.load_state_dict(torch.load('CNN.pth'))
    # evaluate
    datatype = 'val' if isVal else 'test'
    text_dataset.is_train_val_test(datatype)
    bestCNN.eval()
    for idx in range(len(text_dataset.data)):
        row = text_dataset.__getitem__(idx)
        sentence = row['words_sentence']
        sentence = torch.from_numpy(sentence).unsqueeze(dim=0)
        tag_prediction = bestCNN(sentence, softmax=True).argmax(dim=1).item()
        tag_pre = text_dataset.idx2tag[tag_prediction]
        result.append(tag_pre)
    # save the result file
    with open(filename, "w") as f:
        for res in result:
            f.write(res + "\n")


if __name__ == "__main__":
    args = parse_args()
    kernel_sizes = eval(args.kernel_sizes)
    is_cuda = torch.cuda.is_available()
    gpus = [0]

    # --- 1. load and process training data ---
    train_data = load_data(args.train)
    val_data = load_data(args.val)
    test_data = load_data(args.test)
    word2idx = {'<UNK>': 0, '<PAD>': 1}
    idx2word = {v: k for k, v in word2idx.items()}
    tag2idx = {}
    idx2tag = {}
    # build word2idx and idx2word dicts
    for sentence in train_data.words:
        words = sentence.split(" ")
        for word in words:
            if word not in word2idx.keys():
                idx = len(word2idx)
                word2idx[word] = idx
                idx2word[idx] = word
    # build tag2idx and idx2tag dicts
    for tag in train_data.tag:
        if tag not in tag2idx.keys():
            idx = len(tag2idx)
            tag2idx[tag] = idx
            idx2tag[idx] = tag

    # --- 2. get dataset to feed into network from TextDataset ---
    text_dataset = TextDataset(train_data, val_data, test_data, word2idx, idx2word, tag2idx, idx2tag)

    # --- 3. get pretrained word embeddings from glove.6B.100d.txt ---
    emb = []         # emb:        each word index    --> its embedding
    pretrained = {}  # pretrained: each word in glove --> its unique index
    with open("./glove.6B.100d.txt", "r") as f:
        for i, line in enumerate(f):
            line = line.split(" ")
            pretrained[line[0]] = i
            emb.append([float(number) for number in line[1:]])
    emb = np.array(emb)  # dim is: num_words_in_glove x num_embedding_dim

    # --- 4. get embed for each word in trainining dataset ---
    train_emb = []
    for word in text_dataset.word2idx.keys():  # for each word in train data
        if word in pretrained.keys():  # if it is in glove words
            idx = pretrained[word]
            train_emb.append(emb[idx])
        else:   # if the word does not have pretrained embedding from glove, just set to zero
            emb_unseen = torch.zeros(1, len(emb[0]))  # 1 x 100 dim zeros
            emb_unseen = emb_unseen.view(-1)  # to squeeze to 100 dim zeros
            train_emb.append(emb_unseen)
    # get train_emb: num_words_in_traindata x embedding_dim
    train_emb = np.stack(train_emb, axis=0)

    # --- 5. get CNNclass object ---
    CNN = CNNclass(args, train_emb)
    loss_func = nn.CrossEntropyLoss()
    if is_cuda:
        CNN = torch.nn.DataParallel(CNN, device_ids=gpus).cuda()
        loss_func = loss_func.cuda()
    optimizer = optim.Adam(CNN.parameters(), lr=args.lr)

    # --- 6. begin training ---
    train_acc = []
    val_acc = []
    max_acc = 0.0
    for epoch in range(args.epochs):
        # set to use train data
        text_dataset.is_train_val_test('train')
        # get batch data for training
        mydataloader = DataLoader(dataset=text_dataset, batch_size=args.batch_size, shuffle=True)
        # initialize average accuracy for train and val to zero
        train_avg_acc = 0.0
        val_avg_acc = 0.0
        # begin train
        CNN.train()
        # calculate loss
        for i, batch in enumerate(mydataloader):
            sentence_batch = batch['words_sentence'].cuda() if is_cuda else batch['words_sentence']
            tag_batch = batch['tag_sentence'].cuda() if is_cuda else batch['tag_sentence']

            optimizer.zero_grad()
            tag_predict = CNN(sentence_batch)
            loss = loss_func(tag_predict, tag_batch)
            loss_value = loss.item()
            train_avg_acc = (i*train_avg_acc + eval_acc(tag_predict, tag_batch)) / (1+i)
            loss.backward()
            optimizer.step()
            train_acc.append(train_avg_acc)
        print("At Epoch %d : train acc = %f" % (int(epoch), train_avg_acc))

        # --- on validation dataset ---
        text_dataset.is_train_val_test("valid")
        mydataloader = DataLoader(dataset=text_dataset, batch_size=args.batch_size, shuffle=True)
        # call eval()
        CNN.eval()

        for i, batch in enumerate(mydataloader):
            sentence_batch = batch['words_sentence'].cuda() if is_cuda else batch['words_sentence']
            tag_batch = batch['tag_sentence'].cuda() if is_cuda else batch['tag_sentence']
            tag_predict = CNN(sentence_batch)
            val_avg_acc = (val_avg_acc*i + eval_acc(tag_predict, tag_batch)) / (1+i)
        val_acc.append(val_avg_acc)
        print("At Epoch %d : val acc = %f" % (int(epoch), val_avg_acc))

        if val_avg_acc > max_acc:
            torch.save(CNN.state_dict(), 'CNN.pth')

    # --- 7. begin testing ---
    predict('dev_results.txt', text_dataset, args, isVal=True)
    predict('test_results.tx', text_dataset, args, isVal=False)
