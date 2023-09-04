#!/usr/bin/env python
# coding: utf-8

# code ref: https://transformers.run/intro/2021-12-17-transformers-note-4/

# In[1]:


is_debug = False
is_test = True
is_preprocessing_data = True

# 训练方式
# 0: normal training
# 1: bayesian opt
training_methods = 1

print('ok')


# In[2]:


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


# ## Load dataset

# In[3]:


if is_preprocessing_data:
    import pandas as pd

    # 原始数据 name, E, seq
    t7_e_coli_2016_csv = 'data/T7-E.coli-2016.csv'

    data = pd.read_csv(t7_e_coli_2016_csv)
    
    if is_test:
        # 测试 取前 100 个数据
        data = data.head(100)


    # 规整到 [0, 1]
    data['E'] = data['E'] * 0.2

    print(data.head())
    print(data.tail())


# In[4]:


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


# ### Dataset

# In[5]:


from torch.utils.data import Dataset
import csv

class SeqenceDataset(Dataset):
    def __init__(self, data_file: str):

        super(SeqenceDataset, self).__init__()

        # load data from the disk
        with open(data_file, "r") as f:
            data = list(csv.reader(f))[1:]
            
        if len(data[0]) == 3:
            self.texts = [d[2] for d in data]
            self.labels = [d[1] for d in data]
        else:
            raise ValueError(f"Data format not supported. need 2 but get {len(data[0])}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]

if is_debug:
    train_csv, test_csv = 'data/train.csv', 'data/test.csv'
    
    train_dataset = SeqenceDataset(train_csv)
    test_dataset = SeqenceDataset(test_csv)

    print(train_dataset[0])
    print(test_dataset[0])


# ### DataLoader

# In[6]:


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
    #return_token_type_ids：是否返回 token type ids。Token type ids 用于区分文本中不同句子的标识
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
    checkpoint = 'zhihan1996/DNABERT-2-117M'
    cache_dir = 'DNABERT-2-117M'
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=cache_dir, trust_remote_code=True)
    
    batch_size = 4
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x, tokenizer))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x, tokenizer))
    
    print(len(test_dataloader))
    print(len(train_dataloader.dataset))

    batch_X, batch_y = next(iter(train_dataloader))
    batch_lens = [attention_mask.sum().item() for attention_mask in batch_X['attention_mask']]
    print('batch_X shape:', {k: v.shape for k, v in batch_X.items()})
    print('batch_y shape:', batch_y.shape)
    print(batch_X)
    print(batch_lens)
    print(batch_y)


# ## Training model

# In[7]:


if is_debug:
    from transformers import AutoModel
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    dna_bert_2 = AutoModel.from_pretrained(checkpoint, cache_dir=cache_dir, trust_remote_code=True).to(device)
    
    inputs = batch_X['input_ids'].to(device)
    print(inputs.shape, inputs[0])

    with torch.no_grad():
        # [1, 17, 768] 是每个 token 的嵌入，而 [1, 768] 是 [CLS] token 的池化嵌入。由于[CLS]令牌在预训练期间没有经过训练，因此我们不建议将其嵌入用作序列嵌入。
        hidden_states = dna_bert_2(inputs)[0]
    
    print(inputs.shape, hidden_states.shape)

    # Generate per-sequence representations via averaging
    batch_representations = []
    for i, seq_len in enumerate(batch_lens):
        batch_representations.append(hidden_states[i, 1: seq_len-1].mean(0))

    batch_representations = torch.stack(batch_representations, dim=0)
    print(batch_representations.shape)


# ### Build model

# In[8]:


from torch import nn

# 多层感知机
class MLP(nn.Module):
    def __init__(self, param):
        super(MLP, self).__init__()
        
        input_size = param['input_size']
        hidden_sizes = param['hidden_sizes']  # 隐藏层大小列表
        output_size = param['output_size']    # MLP的输出大小
        dropout_prob = param['dropout_prob']  # 被丢弃的概率
        
        layers = []

        # 最后一层不添加 BatchNorm1d, ReLU, Dropout
        sizes = [input_size] + hidden_sizes
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_prob))

        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.mlp = nn.Sequential(*layers)

        # 初试化参数
        for layer in self.mlp.children():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')

    def forward(self, x):
        return self.mlp(x)


# In[9]:


from torch import nn
from transformers import AutoModel

class SequenceRegression(nn.Module):
    def __init__(self, mlp_parma, checkpoint, cache_dir, dna_bert_2_requires_grad=False):
        super(SequenceRegression, self).__init__()
        self.dna_bert_2 = AutoModel.from_pretrained(checkpoint, cache_dir=cache_dir, trust_remote_code=True)
        self.mlp = MLP(mlp_parma)
        
        for param in self.dna_bert_2.parameters():
            param.requires_grad = dna_bert_2_requires_grad

    def forward(self, x, lens):
        # DNABERT-2
        hidden_states = self.dna_bert_2(x)[0]
        
        # Generate per-sequence representations via averaging
        batch_representations = []
        for i, seq_len in enumerate(lens):
            batch_representations.append(hidden_states[i, 1: seq_len-1].mean(0))

        batch_representations = torch.stack(batch_representations, dim=0)
        
        # MLP
        output = self.mlp(batch_representations)
        
        return output


# In[10]:


if is_debug:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    
    mlp_parma = {
        'input_size': 768,
        'hidden_sizes': [768, 256, 128, 64],
        'output_size': 1,
        'dropout_prob': 0.5
    }
    model = SequenceRegression(mlp_parma).to(device)
    print(model)
    
    inputs = batch_X['input_ids'].to(device)

    with torch.no_grad():
        output = model(inputs, batch_lens)

    print(output.shape, output)


# ### Optimize model parameters

# In[11]:


from tqdm.auto import tqdm
import numpy as np

import torch
from sklearn.metrics import mean_squared_error, r2_score

def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, device, total_loss):
    
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'Trian loss: {0:>7f}')
    finish_step_num = (epoch-1)*len(dataloader)
    
    loss_total, rmse_total, r2_total, step_count = 0, 0, 0, 0
    
    model.train()
    for step, (batch_X, batch_y) in enumerate(dataloader, start=1):
        batch_lens = [attention_mask.sum().item() for attention_mask in batch_X['attention_mask']]
        inputs = batch_X['input_ids'].to(device)
        batch_y = batch_y.to(device)
        
        pred = model(inputs, batch_lens)
        
        # tensor[batch_size, 1] -> tensor[batch_size]
        pred = torch.squeeze(pred)

        loss = loss_fn(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        
        # tensor[batch_size, 1] -> tensor[batch_size]
        pred = torch.squeeze(pred)

        # RMSE 衡量了预测值和真实值之间的误差大小，R^2 衡量了模型对总体变异的解释能力。越小的 RMSE 和越接近1的 R^2 表示模型的预测结果越好。
        outputs_np, targets_np = pred.detach().to('cpu').numpy(), batch_y.to('cpu').numpy()

        # 计算均方根误差（RMSE）
        rmse = np.sqrt(mean_squared_error(outputs_np, targets_np))

        # 计算决定系数（R^2） 参数顺序是真实标签在前，预测结果在后
        r2 = r2_score(targets_np, outputs_np)

        loss_total += loss.item()
        rmse_total += rmse
        r2_total += r2
        step_count += 1
            
        progress_bar.set_description(f'Train loss: {total_loss/(finish_step_num + step):>7f}')
        progress_bar.update(1)
        
    loss, rmse, r2 = loss_total / step_count, rmse_total / step_count, r2_total / step_count
#     print(f"Train Loss: {loss:0.1f} RMSE: {rmse:0.1f} R^2: {r2:0.1f}\n")
    
    return total_loss, [loss, rmse, r2]


# In[12]:


def test_loop(dataloader, model, loss_fn, epoch, device, total_loss, mode='Test'):
    assert mode in ['Valid', 'Test']
    
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'{mode} Loss: {0:>7f}')
    finish_step_num = (epoch-1)*len(dataloader)
    
    loss_total, rmse_total, r2_total, step_count = 0, 0, 0, 0

    model.eval()
    with torch.no_grad():
        for step, (batch_X, batch_y) in enumerate(dataloader, start=1):
            batch_lens = [attention_mask.sum().item() for attention_mask in batch_X['attention_mask']]
            inputs = batch_X['input_ids'].to(device)
            batch_y = batch_y.to(device)

            pred = model(inputs, batch_lens)
            
            # tensor[batch_size, 1] -> tensor[batch_size]
            pred = torch.squeeze(pred)
            
            loss = loss_fn(pred, batch_y)
            
            # RMSE 衡量了预测值和真实值之间的误差大小，R^2 衡量了模型对总体变异的解释能力。越小的 RMSE 和越接近1的 R^2 表示模型的预测结果越好。
            outputs_np, targets_np = pred.detach().to('cpu').numpy(), batch_y.to('cpu').numpy()

            # 计算均方根误差（RMSE）
            rmse = np.sqrt(mean_squared_error(outputs_np, targets_np))

            # 计算决定系数（R^2） 参数顺序是真实标签在前，预测结果在后
            r2 = r2_score(targets_np, outputs_np)

            loss_total += loss.item()
            rmse_total += rmse
            r2_total += r2
            step_count += 1
            
            progress_bar.set_description(f'{mode} Loss: {total_loss/(finish_step_num + step):>7f}')
            progress_bar.update(1)

    loss, rmse, r2 = loss_total / step_count, rmse_total / step_count, r2_total / step_count
#     print(f"{mode} Loss: {loss:0.1f} RMSE: {rmse:0.1f} R^2: {r2:0.1f}\n")
    
    return total_loss, [loss, rmse, r2]


# ### Normal training

# In[13]:


# 0: normal training
if training_methods == 0 and not is_debug:
    import os
    import sys
    from transformers import AdamW, get_scheduler

    # hyper-parameters
    checkpoint = 'zhihan1996/DNABERT-2-117M'
    cache_dir = 'DNABERT-2-117M'
    model_max_length = int(3000 * 0.25)

    batch_size = 8
    learning_rate = 1e-5
    epoch_num = 3
    tag = f'{cache_dir}-{model_max_length}-{batch_size}-{learning_rate}-{epoch_num}'
    print(f'Tag: {tag}')

    # path: model and train_result_data
    model_bin = 'data/{tag}.bin'
    train_result_data = f'data/train-{tag}.csv'
    if os.path.exists(train_result_data):
        os.remove(train_result_data)

    # dataset
    train_csv, test_csv = 'data/train.csv', 'data/test.csv'
    train_dataset, test_dataset= SeqenceDataset(train_csv), SeqenceDataset(test_csv)

    # dataloader
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=cache_dir, model_max_length=model_max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collote_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collote_fn)

    # model
    mlp_parma = {
        'input_size': 768,
        'hidden_sizes': [768, 256, 128, 64],
        'output_size': 1,
        'dropout_prob': 0.5
    }
    model = SequenceRegression(mlp_parma, checkpoint, cache_dir)
    # print(model)
    
    # cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"Let's use, {torch.cuda.device_count()}, GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    else:
        print(f"Let's use, CPU!")
    
    model = model.to(device)

    # loss fun
    loss_fn = nn.CrossEntropyLoss()

    # ransformers 库的优化器会随着训练过程逐步减小学习率
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # lr
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=epoch_num*len(train_dataloader),
    )

    # train
    total_train_loss, total_test_loss = 0., 0.
    bast_r2 = -sys.maxsize - 1
    for t in range(epoch_num):
        print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
        total_train_loss, train_data = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t+1, device, total_train_loss)
        total_test_loss, test_data = test_loop(test_dataloader, model, loss_fn, t+1, device, total_test_loss, mode='Test')

        # save train data
        tt_data = train_data + test_data
        with open(train_result_data, 'a', encoding='utf-8') as w:
            strs = ''
            for item in tt_data:
                strs += f'{item:.4f},'
            w.write(f'{strs[:-1]}\n')

        # save bast model
        test_r2 = test_data[-1]
        if bast_r2 < test_r2:
            bast_r2 = test_r2
            torch.save(model.state_dict(), model_bin)

    print("Done!")


# ### Bayesian opt

# In[14]:


# 1: bayesian opt
if training_methods == 1 and not is_debug:
    import os
    import sys
    from transformers import AdamW, get_scheduler
    import optuna
    
    # bayesian opt
    def objective(_trial):
        # 对数均匀分布的下界。通常是超参数的最小值，最大值
        lr = _trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        hidden_dim = [
            int(_trial.suggest_float('hidden_layer_1', 256, 768, log=True)),
            int(_trial.suggest_float('hidden_layer_2', 256, 512, log=True)),
            int(_trial.suggest_float('hidden_layer_3', 64, 384, log=True)),
            int(_trial.suggest_float('hidden_layer_4', 32, 256, log=True))
        ]
        dropout = _trial.suggest_float('dropout', 0.3, 0.6, log=True)

        # hyper-parameters
        checkpoint = 'zhihan1996/DNABERT-2-117M'
        cache_dir = 'DNABERT-2-117M'
        model_max_length = int(3000 * 0.25)

        batch_size = 8
        learning_rate = lr
        epoch_num = 100
        tag = f'{cache_dir}-{model_max_length}-{batch_size}-{learning_rate}-{epoch_num}'
        print(f'Tag: {tag}')

        # path: model and train_result_data
        model_bin = 'data/{tag}.bin'
        train_result_data = f'data/train-{tag}.csv'
        if os.path.exists(train_result_data):
            os.remove(train_result_data)

        # dataset
        train_csv, test_csv = 'data/train.csv', 'data/test.csv'
        train_dataset, test_dataset= SeqenceDataset(train_csv), SeqenceDataset(test_csv)

        # dataloader
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=cache_dir, model_max_length=model_max_length)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x, tokenizer))
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x, tokenizer))

        # model
        mlp_parma = {
            'input_size': 768,
            'hidden_sizes': hidden_dim,
            'output_size': 1,
            'dropout_prob': dropout
        }
    
        model = SequenceRegression(mlp_parma, checkpoint, cache_dir)
        # print(model)

        # cuda
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            print(f"Let's use, {torch.cuda.device_count()}, GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model)
        else:
            print(f"Let's use, {device}!")

        model = model.to(device)

        # loss fun
        loss_fn = nn.CrossEntropyLoss()

        # ransformers 库的优化器会随着训练过程逐步减小学习率
        optimizer = AdamW(model.parameters(), lr=learning_rate)

        # lr
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=epoch_num*len(train_dataloader),
        )

        # train
        total_train_loss, total_test_loss = 0., 0.
        bast_r2 = -sys.maxsize - 1
        for t in range(epoch_num):
            print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
            total_train_loss, train_data = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t+1, device, total_train_loss)
            total_test_loss, test_data = test_loop(test_dataloader, model, loss_fn, t+1, device, total_test_loss, mode='Test')

            # record bast model
            test_r2 = test_data[-1]
            if bast_r2 < test_r2:
                bast_r2 = test_r2
                
        return bast_r2
    
    # start bayesian opt
    n_trials = 100
    
    # 取两个值之一：'minimize' 或 'maximize'。默认值是 'minimize'
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


# ### load model and predicted

# In[ ]:


def predict(dataloader, model, mode='Test'):
    assert mode in ['Valid', 'Test']
    
    progress_bar = tqdm(range(len(dataloader)))
    
    rmse_total, r2_total, step_count = 0, 0, 0

    model.eval()
    with torch.no_grad():
        for step, (batch_X, batch_y) in enumerate(dataloader, start=1):
            batch_lens = [attention_mask.sum().item() for attention_mask in batch_X['attention_mask']]
            inputs = batch_X['input_ids'].to(device)
            batch_y = batch_y.to(device)

            pred = model(inputs, batch_lens)
            
            # tensor[batch_size, 1] -> tensor[batch_size]
            pred = torch.squeeze(pred)
            
            # RMSE 衡量了预测值和真实值之间的误差大小，R^2 衡量了模型对总体变异的解释能力。越小的 RMSE 和越接近1的 R^2 表示模型的预测结果越好。
            outputs_np, targets_np = pred.detach().to('cpu').numpy(), batch_y.to('cpu').numpy()

            # 计算均方根误差（RMSE）
            rmse = np.sqrt(mean_squared_error(outputs_np, targets_np))

            # 计算决定系数（R^2） 参数顺序是真实标签在前，预测结果在后
            r2 = r2_score(targets_np, outputs_np)

            rmse_total += rmse
            r2_total += r2
            step_count += 1
            
            progress_bar.update(1)

    rmse, r2 = rmse_total / step_count, r2_total / step_count
    
    return rmse, r2


# In[ ]:


# model.load_state_dict(torch.load(model_pth))
# print(predict(test_dataloader, model, mode='Test'))

