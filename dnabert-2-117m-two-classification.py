#!/usr/bin/env python
# coding: utf-8

# code ref: https://transformers.run/intro/2021-12-17-transformers-note-4/
# 
# 对 NESG 数据做二分类任务
# - 0 1 2 为低表达数据，5 为高表达数据
# - 使用 DNABERT-2 做词嵌入

# In[1]:


is_debug = False

is_preprocessing_data = True
is_test_data = False
show_data_pic = False

use_web_ssi = True

# normal training: 0
# bayesian opt: 1
# grid search: 2
training_methods = 1

trials = 80
num_epoch = 50

log_file_tag = 1

if False:
    is_preprocessing_data = True
    is_test_data = True

    num_epoch = 3

    trials = 3

print('ok')

# In[2]:


import os
import datetime


class Logging:
    def __init__(self, log_file=None):
        self.log_file = log_file
        if self.log_file is None:
            self.log_file = f'data/log-{log_file_tag}.txt'

    def init(self):
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def msg(self, info, is_print=True):
        with open(self.log_file, 'a', encoding='utf-8') as w:
            w.write(f'{info}\n')
        if is_print:
            print(info)

    def datatime_now(self):
        self.msg(f'### {datetime.datetime.now()} ###')


logging = Logging()

logging.init()
# logging.datatime_now()


# ## Load dataset

# In[3]:


#
import random
import os
import numpy as np
import torch


def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


seed = 2023
seed_everything(seed=seed)

# In[4]:


if is_preprocessing_data:
    import pandas as pd

    # 原始数据 name, E, seq
    t7_e_coli_2016_csv = 'data/T7-E.coli-2016.csv'

    data = pd.read_csv(t7_e_coli_2016_csv, header=1, names=['name', 'label', 'nucle-seq'])

    print(data.shape)
    print(data.head())
    print(data.tail())

    # 筛选数据
    data['label'].loc[data['label'] == 1] = 0
    data['label'].loc[data['label'] == 2] = 0
    data['label'].loc[data['label'] == 5] = 1
    data = data[data['label'] < 2]

    if is_test_data:
        data = data[:10]

    print(data.shape)
    print(data.head())
    print(data.tail())

# In[5]:


if is_preprocessing_data:
    # 划分数据集
    from sklearn.model_selection import train_test_split

    train_dataset, test_dataset = train_test_split(data, test_size=0.2, random_state=seed)

    print(train_dataset.shape, test_dataset.shape)
    print(train_dataset.head())
    print(test_dataset.head())

    # Save the remaining data to a CSV file
    train_csv, test_csv = 'data/train.csv', 'data/test.csv'

    # 设置index=False参数以避免保存索引列
    train_dataset.to_csv(train_csv, index=False)
    test_dataset.to_csv(test_csv, index=False)

# In[6]:


if is_preprocessing_data and show_data_pic:
    import matplotlib.pyplot as plt


    def show_data(data_pd):
        label_count = {}
        for item in data_pd['label']:
            label_count[item] = label_count.get(item, 0) + 1

        x = [i for i in range(2)]
        y = [label_count[i] for i in x]
        print(x, y)

#     for item in [data, train_dataset, test_dataset]:
#         show_data(item)


# ## DNABERT-2 Embedding

# ### Dataset

# In[7]:


from torch.utils.data import Dataset
import csv


class SequenceDataset(Dataset):
    def __init__(self, data_file: str):

        super(SequenceDataset, self).__init__()

        # load data from the disk
        with open(data_file, "r") as f:
            data = list(csv.reader(f))[1:]

        if len(data[0]) == 3:
            self.texts = [d[2] for d in data]
            #             self.labels = [int(d[1]) * 0.2 for d in data]
            self.labels = [int(d[1]) for d in data]
        else:
            raise ValueError(f"Data format not supported. need 3 but get {len(data[0])}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]


if is_debug:
    train_csv, test_csv = 'data/train.csv', 'data/test.csv'

    train_dataset = SequenceDataset(train_csv)
    test_dataset = SequenceDataset(test_csv)

    print(train_dataset[0])
    print(test_dataset[0])

# ### DataLoader

# In[8]:


import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def collate_fn(batch_samples, tokenizer):
    batch_texts = []
    batch_labels = []

    for sample in batch_samples:
        batch_texts.append(sample[0])
        batch_labels.append(float(sample[1]))

    # return_attention_mask：是否返回 attention mask。Attention mask 用于表示哪些位置在文本中是有效的，哪些是 padding
    # return_token_type_ids：是否返回 token type ids。Token type ids 用于区分文本中不同句子的标识
    X = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
        return_token_type_ids=False
    )

    y = torch.tensor(batch_labels)

    return X, y


if is_debug:
    checkpoint = 'Huggingface-DNABERT-2-117M'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

    batch_size = 4
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=lambda x: collate_fn(x, tokenizer))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                 collate_fn=lambda x: collate_fn(x, tokenizer))

    print(len(train_dataloader))
    print(len(train_dataloader.dataset))

    batch_X, batch_y = next(iter(train_dataloader))
    batch_lens = [attention_mask.sum().item() for attention_mask in batch_X['attention_mask']]
    print('batch_X shape:', {k: v.shape for k, v in batch_X.items()})
    print('batch_y shape:', batch_y.shape)
    print(batch_X)
    print(batch_lens)
    print(batch_y)

# ## Training model

# ### Build model

# In[9]:


from torch import nn


# 多层感知机
class MLP(nn.Module):
    def __init__(self, param):
        super(MLP, self).__init__()

        input_size = param['input_size']
        hidden_sizes = param['hidden_sizes']  # 隐藏层大小列表
        output_size = param['output_size']  # MLP的输出大小
        dropout_prob = param['dropout_prob']  # 被丢弃的概率

        layers = []

        if len(hidden_sizes) != 0:
            # 最后一层不添加 BatchNorm1d, ReLU, Dropout
            sizes = [input_size] + hidden_sizes
            for i in range(len(sizes) - 1):
                layers.append(nn.Linear(sizes[i], sizes[i + 1]))
                layers.append(nn.BatchNorm1d(sizes[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=dropout_prob))

            layers.append(nn.Linear(hidden_sizes[-1], output_size))
        else:
            layers.append(nn.Dropout(p=dropout_prob))
            layers.append(nn.Linear(input_size, output_size))

        self.mlp = nn.Sequential(*layers)

        # 初试化参数
        for layer in self.mlp.children():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')

    def forward(self, x):
        return self.mlp(x)


# ### Train Test Loop

# In[10]:


from tqdm.auto import tqdm
import numpy as np

import torch
from sklearn.metrics import mean_squared_error, r2_score


def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, device, use_tqdm=False):
    if use_tqdm:
        progress_bar = tqdm(range(len(dataloader)))
        progress_bar.set_description(f'Train loss: {0:>4f}')

    loss_total, step_count = 0, 0
    correct = 0

    model.train()
    for batch_X, batch_y in dataloader:
        # get input and batch_len
        batch_lens = [attention_mask.sum().item() for attention_mask in batch_X['attention_mask']]

        batch_X = batch_X['input_ids'].to(device)
        batch_y = batch_y.type(torch.LongTensor)
        batch_y = batch_y.to(device)

        pred = model(batch_X, batch_lens)
        #         pred = model(batch_X)

        correct += (pred.argmax(1) == batch_y).type(torch.float).sum().item()

        # tensor[batch_size, 1] -> tensor[batch_size]
        #         pred = torch.squeeze(pred)

        loss = loss_fn(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        loss_total += loss.item()
        step_count += 1

        if use_tqdm:
            progress_bar.set_description(f'Train loss: {(loss_total / step_count):>4f}')
            progress_bar.update(1)

    loss = loss_total / step_count
    correct /= len(dataloader.dataset)

    return [loss, correct]


# In[11]:


def test_loop(dataloader, model, loss_fn, device, mode='Test', use_tqdm=False):
    assert mode in ['Valid', 'Test']

    if use_tqdm:
        progress_bar = tqdm(range(len(dataloader)))
        progress_bar.set_description(f'Test loss: {0:>4f}')

    loss_total, step_count = 0, 0
    correct = 0

    model.eval()
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            # get input and batch_len
            batch_lens = [attention_mask.sum().item() for attention_mask in batch_X['attention_mask']]

            batch_X = batch_X['input_ids'].to(device)
            batch_y = batch_y.type(torch.LongTensor)
            batch_y = batch_y.to(device)

            pred = model(batch_X, batch_lens)
            #             pred = model(batch_X)

            correct += (pred.argmax(1) == batch_y).type(torch.float).sum().item()

            # tensor[batch_size, 1] -> tensor[batch_size]
            #             pred = torch.squeeze(pred)

            loss = loss_fn(pred, batch_y)
            loss_total += loss.item()
            step_count += 1

            if use_tqdm:
                progress_bar.set_description(f'Test loss: {(loss_total / step_count):>4f}')
                progress_bar.update(1)

    loss = loss_total / step_count
    correct /= len(dataloader.dataset)

    return [loss, correct]


# ### Bayesian opt

# In[12]:


from torch import nn
from transformers import AutoModel, AutoTokenizer


class SequenceRegression(nn.Module):
    def __init__(self, mlp_parma, dna_bert_2_checkpoint, use_mean=True, dna_bert_2_requires_grad=False):
        super(SequenceRegression, self).__init__()

        self.use_mean = use_mean
        self.dna_bert_2 = AutoModel.from_pretrained(dna_bert_2_checkpoint, trust_remote_code=True)

        self.lstm = nn.LSTM(input_size=768, hidden_size=512, num_layers=1, batch_first=True, bias=True)

        self.attention = nn.Linear(512, 1)

        self.mlp = MLP(mlp_parma)

        for param in self.dna_bert_2.parameters():
            param.requires_grad = dna_bert_2_requires_grad

    def forward(self, x, lens):
        #     def forward(self, x):
        hidden_states = self.dna_bert_2(x)[0]

        if self.use_mean is True:
            # Generate per-sequence representations via averaging: get batch lens
            batch_representations = [hidden_states[index, 1: item - 1].mean(0) for index, item in enumerate(lens)]
            batch_representations = torch.stack(batch_representations, dim=0)
        else:
            batch_representations = hidden_states[:, 0, :]

        # MLP
        output = self.mlp(batch_representations)

        return output


# In[ ]:


# 1: bayesian opt
# training_methods = 1
if training_methods == 1 and not is_debug:
    import os
    import sys
    from transformers import get_scheduler
    import optuna

    from tqdm.auto import tqdm

    # from tqdm.autonotebook import tqdm
    # from tqdm.notebook import tqdm

    # 设置 Optuna 的日志级别为 WARNING
    optuna.logging.set_verbosity(optuna.logging.INFO)

    use_mean = False
    layers = -1

    if use_web_ssi:
        checkpoint = 'zhihan1996/DNABERT-2-117M'
    else:
        # checkpoint = 'Huggingface-DNABERT-2-117M'
        checkpoint = '/Users/yudd/code/python/jupyter/Huggingface-DNABERT-2-117M'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # hyper-parameters
    batch_size = 32
    epoch_num = num_epoch

    # dataset
    train_csv, test_csv = 'data/train.csv', 'data/test.csv'
    train_dataset, test_dataset = SequenceDataset(train_csv), SequenceDataset(test_csv)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, model_max_length=636, padding_side="right",
                                              trust_remote_code=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=lambda x: collate_fn(x, tokenizer))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                 collate_fn=lambda x: collate_fn(x, tokenizer))


    # bayesian opt
    def objective(_trial):
        # 对数均匀分布的下界。通常是超参数的最小值，最大值
        lr = _trial.suggest_float('lr', 1e-6, 1, log=True)
        dropout = _trial.suggest_float('dropout', 0.1, 0.6, log=True)

        if layers == 0:
            hidden_dim = []

        if layers == 1:
            h1 = int(_trial.suggest_float('hidden_layer_1', 32, 1024, log=True))
            hidden_dim = [h1]

        if layers == 2:
            h1 = int(_trial.suggest_float('hidden_layer_1', 32, 1024, log=True))
            h2 = int(_trial.suggest_float('hidden_layer_2', 16, h1, log=True))
            hidden_dim = [h1, h2]

        if layers == 3:
            h1 = int(_trial.suggest_float('hidden_layer_1', 32, 1024, log=True))
            h2 = int(_trial.suggest_float('hidden_layer_2', 16, h1, log=True))
            h3 = int(_trial.suggest_float('hidden_layer_3', 8, h2, log=True))
            hidden_dim = [h1, h2, h3]

        if layers == 4:
            h1 = int(_trial.suggest_float('hidden_layer_1', 32, 1024, log=True))
            h2 = int(_trial.suggest_float('hidden_layer_2', 16, h1, log=True))
            h3 = int(_trial.suggest_float('hidden_layer_3', 8, h2, log=True))
            h4 = int(_trial.suggest_float('hidden_layer_4', 4, h3, log=True))
            hidden_dim = [h1, h2, h3, h4]

        if layers == 5:
            h1 = int(_trial.suggest_float('hidden_layer_1', 32, 1024, log=True))
            h2 = int(_trial.suggest_float('hidden_layer_2', 16, h1, log=True))
            h3 = int(_trial.suggest_float('hidden_layer_3', 8, h2, log=True))
            h4 = int(_trial.suggest_float('hidden_layer_4', 8, h3, log=True))
            h5 = int(_trial.suggest_float('hidden_layer_5', 4, h4, log=True))
            hidden_dim = [h1, h2, h3, h4, h5]

        # hyper-parameters
        learning_rate = lr
        tag = f'Model-{batch_size}-{round(learning_rate, 6)}-{round(dropout, 4)}-{layers}-{use_mean}'
        logging.msg(f'Tag: {tag}')

        # path: model and train_result_data
        model_bin = f'data/{tag}.bin'
        train_result_data = f'data/Train-{tag}.csv'
        if os.path.exists(train_result_data):
            os.remove(train_result_data)

            # model
        mlp_parma = {
            'input_size': 768,
            'hidden_sizes': hidden_dim,
            'output_size': 2,
            'dropout_prob': dropout
        }

        model = SequenceRegression(mlp_parma, checkpoint, use_mean).to(device)
        # print(model)

        # loss fun
        loss_fn = nn.CrossEntropyLoss()

        # ransformers 库的优化器会随着训练过程逐步减小学习率
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        # lr
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=epoch_num * len(train_dataloader),
        )

        # train
        progress_bar = tqdm(range(epoch_num))
        progress_bar.set_description(f'Trian acc: {0:>4f}')

        limit_count_max = 10
        limit_count = limit_count_max

        bast_acc = -sys.maxsize - 1
        for t in range(epoch_num):
            limit_count -= 1

            #             logging.msg(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
            train_data = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, device, use_tqdm=True)
            test_data = test_loop(test_dataloader, model, loss_fn, device, mode='Test', use_tqdm=True)

            # save train data
            tt_data = train_data + test_data
            with open(train_result_data, 'a', encoding='utf-8') as w:
                strs = ''
                for item in tt_data:
                    strs += f'{item:.4f},'
                w.write(f'{strs[:-1]}\n')

                logging.msg(f'Epoch {t + 1}/{epoch_num}: {strs}')

            # save bast model
            test_acc = test_data[-1]
            if bast_acc < test_acc:
                bast_acc = test_acc
                #                 torch.save(model.state_dict(), model_bin)
                logging.msg(f'Bast acc: {bast_acc}')

                limit_count = limit_count_max

            progress_bar.set_description(f'Trian acc: {bast_acc:>4f}, {test_acc:>4f}')
            progress_bar.update(1)

            if limit_count == 0:
                break

        return bast_acc


    #     for _use_mean in [False, True]:
    #         for _layers in range(6):
    logging.datatime_now()

    #         use_mean = _use_mean
    #         layers = _layers
    use_mean = True
    layers = 3

    # start bayesian opt
    n_trials = trials
    logging.msg(f'use_mean: {use_mean}, layers: {layers}, trials: {n_trials}')

    # 取两个值之一：'minimize' 或 'maximize'。默认值是 'minimize'
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    logging.msg("Number of finished trials: ", len(study.trials))
    logging.msg("Best trial:")
    trial = study.best_trial
    logging.msg("  Value: ", trial.value)
    logging.msg("  Params: ")
    for key, value in trial.params.items():
        logging.msg("    {}: {}".format(key, value))

    logging.msg('Done Done Done')
    logging.msg('')

