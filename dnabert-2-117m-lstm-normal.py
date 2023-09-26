import argparse
import datetime
import random
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from torch import nn
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import csv
from transformers import get_scheduler
import optuna


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


def preprocessing_data():
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

    print(data.shape)
    print(data.head())
    print(data.tail())

    # 划分数据集
    train_dataset, test_dataset = train_test_split(data, test_size=0.2, random_state=2023)

    print(train_dataset.shape, test_dataset.shape)
    print(train_dataset.head())
    print(test_dataset.head())

    # Save the remaining data to a CSV file
    train_csv, test_csv = 'data/train.csv', 'data/test.csv'

    # 设置index=False参数以避免保存索引列
    train_dataset.to_csv(train_csv, index=False)
    test_dataset.to_csv(test_csv, index=False)


class Logging:
    """
    日志类
    """
    def __init__(self, log_file=None):
        self.log_file = log_file

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


def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, device, use_tqdm=False):
    if use_tqdm:
        progress_bar = tqdm(range(len(dataloader)))
        progress_bar.set_description(f'Train loss: {0:>4f}')

    loss_total, step_count = 0, 0
    correct = 0

    model.train()
    for batch_X, batch_y in dataloader:
        # get input and batch_len
        # batch_lens = [attention_mask.sum().item() for attention_mask in batch_X['attention_mask']]

        batch_X = batch_X['input_ids'].to(device)
        batch_y = batch_y.type(torch.LongTensor)
        batch_y = batch_y.to(device)

        # pred = model(batch_X, batch_lens)
        pred = model(batch_X)

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
            # batch_lens = [attention_mask.sum().item() for attention_mask in batch_X['attention_mask']]

            batch_X = batch_X['input_ids'].to(device)
            batch_y = batch_y.type(torch.LongTensor)
            batch_y = batch_y.to(device)

            # pred = model(batch_X, batch_lens)
            pred = model(batch_X)

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


class SequenceRegression(nn.Module):
    def __init__(self, dna_bert_2_checkpoint, lstm_unit_number, use_double_lstm, use_attention, dropout_prob, dna_bert_2_requires_grad=False):
        super(SequenceRegression, self).__init__()
        self.use_attention = use_attention

        self.dna_bert_2 = AutoModel.from_pretrained(dna_bert_2_checkpoint, trust_remote_code=True)

        self.lstm = nn.LSTM(input_size=768, hidden_size=lstm_unit_number, num_layers=1, batch_first=True, bias=True, bidirectional=use_double_lstm)

        if self.use_attention:
            self.attention = nn.Linear(lstm_unit_number * (int(use_double_lstm) + 1), 1)

        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(lstm_unit_number * (int(use_double_lstm) + 1), 2)

        for param in self.dna_bert_2.parameters():
            param.requires_grad = dna_bert_2_requires_grad

    def forward(self, x):
        hidden_states = self.dna_bert_2(x)[0]

        # LSTM
        lstm_output, _ = self.lstm(hidden_states)

        if self.use_attention:
            # Attention
            attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
            weighted_lstm_out = torch.sum(attention_weights * lstm_output, dim=1)
            lstm_representations = weighted_lstm_out
        else:
            lstm_representations = torch.mean(lstm_output, dim=1)

        # FC
        output = self.fc(lstm_representations)

        return output


def custom_objective(use_web_ssi, use_double_lstm, use_attention, logging):
    def objective(trial):
        # 对数均匀分布的下界。通常是超参数的最小值，最大值
        learning_rate = trial.suggest_float('lr', 1e-6, 1, log=True)
        dropout = trial.suggest_float('dropout', 0.000001, 0.4, log=True)
        lstm_unit_num = int(trial.suggest_float('lstm_unit_mum', 256, 1024, log=True))

        # hyper-parameters
        epoch_num = 80
        batch_size = 16
        tag = f'Model-{lstm_unit_num}-{batch_size}-{round(learning_rate, 3)}-{round(dropout, 3)}-{use_double_lstm}-{use_attention}'
        logging.msg(f'Tag: {tag}')

        # path: model and train_result_data
        train_result_data = f'data/Train-{tag}.csv'
        if os.path.exists(train_result_data):
            os.remove(train_result_data)

        # use_web_ssi = False
        if use_web_ssi:
            checkpoint = 'zhihan1996/DNABERT-2-117M'
        else:
            checkpoint = '/Users/yudd/code/python/jupyter/Huggingface-DNABERT-2-117M'

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # dataset
        train_csv, test_csv = 'data/train.csv', 'data/test.csv'
        train_dataset, test_dataset = SequenceDataset(train_csv), SequenceDataset(test_csv)

        tokenizer = AutoTokenizer.from_pretrained(checkpoint, model_max_length=636, padding_side="right",
                                                  trust_remote_code=True)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      collate_fn=lambda x: collate_fn(x, tokenizer))
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                     collate_fn=lambda x: collate_fn(x, tokenizer))

        model = SequenceRegression(checkpoint, lstm_unit_num, use_double_lstm, use_attention, dropout).to(device)
        # print(model)

        # loss fun
        loss_fn = nn.CrossEntropyLoss()

        # transformers 库的优化器会随着训练过程逐步减小学习率
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        # lr
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=epoch_num * len(train_dataloader),
        )

        # train
        use_tqdm = False
        if use_tqdm:
            progress_bar = tqdm(range(epoch_num))
            progress_bar.set_description(f'Train acc: {0:>4f}')

        limit_count_max = 16
        limit_count = limit_count_max

        bast_acc = -sys.maxsize - 1
        for t in range(epoch_num):
            curr_time = datetime.datetime.now()
            limit_count -= 1

            train_data = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, device, use_tqdm=use_tqdm)
            test_data = test_loop(test_dataloader, model, loss_fn, device, mode='Test', use_tqdm=use_tqdm)

            # save bast model
            test_acc = test_data[-1]
            if bast_acc < test_acc:
                bast_acc = test_acc
                logging.msg(f'Bast acc: {bast_acc}')

                limit_count = limit_count_max

            time_difference = datetime.datetime.now() - curr_time
            total_seconds = time_difference.total_seconds()

            if use_tqdm:
                progress_bar.set_description(f'Train acc: {bast_acc:>4f}, {test_acc:>4f}')
                progress_bar.update(1)
            else:
                logging.msg(f'{trial.number}, Epoch {t + 1}/{epoch_num} {round(total_seconds, 2)}s, {round((total_seconds*80*80)/3600.0, 2)}h: {train_data}, {test_data}, best: {bast_acc}')

            if limit_count == 0:
                break

        return bast_acc

    return objective


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='命令行中传入一个数字')
    parser.add_argument('--gpu', type=str, default=0, help='传入的数字')
    parser.add_argument('--ssi', action='store_true', help='启用服务器')
    parser.add_argument('--blstm', action='store_true', help='启用双向LSTM')
    parser.add_argument('--attention', action='store_true', help='添加注意力层')

    args = parser.parse_args()
    # args.blstm = True
    # args.attention = True
    print(args)

    use_gpu_id = args.gpu

    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{use_gpu_id}'

    # set time seed
    seed_everything(2023)

    # log info
    logging = Logging(log_file=f'log-{use_gpu_id}.txt')
    logging.init()
    logging.datatime_now()

    # data
    # preprocessing_data()

    # 创建 Optuna study 并传递额外的参数
    study = optuna.create_study(direction='maximize')
    study.enqueue_trial({'lr': 0.001, 'dropout': 0.1, 'lstm_unit_mum': 512})  # 初始参数设置
    # study.user_attrs['use_double_lstm'] = args.blstm  # 传递额外的参数
    # study.user_attrs['use_attention'] = args.attention  # 传递额外的参数
    study.optimize(custom_objective(args.ssi, args.blstm, args.attention, logging), n_trials=100)

    # result
    logging.msg("Best trial:")
    trial = study.best_trial
    logging.msg("  Value: ", trial.value)
    logging.msg("  Params: ")
    for key, value in trial.params.items():
        logging.msg("    {}: {}".format(key, value))

    logging.msg('Done Done Done')
