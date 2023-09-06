#!/usr/bin/env python
# coding: utf-8

# code ref: https://transformers.run/intro/2021-12-17-transformers-note-4/

# In[1]:


is_debug = False

is_preprocessing_data = False
is_test = True

# 先使用 DNABERT-2 编码，再优化 MLP
use_dnabert_single_embedding = True

# 使用服务器
use_main_ssi = False

training_methods = 1

# 使用 贝叶斯优化
use_bayesian_opt = True

print('ok')


# ## Load dataset

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


# ## DNABERT-2 Embedding

# ### Dataset

# In[5]:


if use_dnabert_single_embedding:
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
                self.names = [d[0] for d in data]
            else:
                raise ValueError(f"Data format not supported. need 2 but get {len(data[0])}")

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            return self.names[i], self.texts[i], self.labels[i]

    if is_debug:
        train_csv, test_csv = 'data/train.csv', 'data/test.csv'

        train_dataset = SeqenceDataset(train_csv)
        test_dataset = SeqenceDataset(test_csv)

        print(train_dataset[0])
        print(test_dataset[0])


# ### DataLoader

# In[6]:


if use_dnabert_single_embedding:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    def collate_fn(batch_samples, tokenizer):
        batch_names = []
        batch_texts = []
        batch_labels = []

        for sample in batch_samples:
            batch_names.append(sample[0])
            batch_texts.append(sample[1])
            batch_labels.append(float(sample[2]))

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

        return batch_names, X, y

    if is_debug:
        checkpoint = 'Huggingface-DNABERT-2-117M'
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

        batch_size = 4
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x, tokenizer))
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x, tokenizer))

        print(len(test_dataloader))
        print(len(train_dataloader.dataset))

        batch_names, batch_X, batch_y = next(iter(train_dataloader))
        batch_lens = [attention_mask.sum().item() for attention_mask in batch_X['attention_mask']]
        print(batch_names)
        print('batch_X shape:', {k: v.shape for k, v in batch_X.items()})
        print('batch_y shape:', batch_y.shape)
        print(batch_X)
        print(batch_lens)
        print(batch_y)


# ### Embedding

# In[7]:


if use_dnabert_single_embedding:
    from tqdm.auto import tqdm
    import pickle
    
    # embedding
    def dnabert_embedding(model, dataloader, tag):
        embedding_res_dict, embedding_label_dict = {}, {}
        bar = tqdm(total=len(dataloader))

        for gene_names, batch_X, labels in dataloader:
            bar.update(1)

            # get input and batch_len
            batch_lens = [attention_mask.sum().item() for attention_mask in batch_X['attention_mask']]
            inputs = batch_X['input_ids'].to(device)

            with torch.no_grad():
                hidden_states = model(inputs)[0]

            # Generate per-sequence representations via averaging
            batch_representations = []
            for i, seq_len in enumerate(batch_lens):
                batch_representations.append(hidden_states[i, 1: seq_len-1].mean(0))

    #             batch_representations = torch.stack(batch_representations, dim=0)

            for index, gene_name in enumerate(gene_names):
                embedding_res_dict[gene_name] = batch_representations[index]
                embedding_label_dict[gene_name] = labels[index]

        embedding_res_file = f'data/{tag}_dnabert_seq_embedding.pkl'
        with open(embedding_res_file, 'wb') as w:
            pickle.dump(embedding_res_dict, w)

        embedding_label_file = f'data/{tag}_dnabert_label_embedding.pkl'
        with open(embedding_label_file, 'wb') as w:
            pickle.dump(embedding_label_dict, w)

        bar.close()


# In[8]:


if use_dnabert_single_embedding:
    from transformers import AutoModel, AutoTokenizer
    
    # tokenizer and dnabert-2 model
    if use_main_ssi:
        checkpoint = 'zhihan1996/DNABERT-2-117M'
    else:
        checkpoint = 'Huggingface-DNABERT-2-117M'

    model_max_length = int(3000 * 0.25)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, model_max_length=model_max_length, trust_remote_code=True)
    dna_bert_2 = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
    
    # cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"Let's use, {torch.cuda.device_count()}, GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        dna_bert_2 = nn.DataParallel(mdna_bert_2odel)
    else:
        print(f"Let's use, {device}!")
    
    dna_bert_2 = dna_bert_2.to(device)
    dna_bert_2.eval()
    
    # hyper-parameters
    batch_size = 4
    
    # dataset and dataloader
    train_csv, test_csv = 'data/train.csv', 'data/test.csv'
    train_dataset, test_dataset= SeqenceDataset(train_csv), SeqenceDataset(test_csv)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=lambda x: collate_fn(x, tokenizer))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=lambda x: collate_fn(x, tokenizer))
    
    # embedding
    dnabert_embedding(dna_bert_2, train_dataloader, tag='train')
    dnabert_embedding(dna_bert_2, test_dataloader, tag='test') 


# In[9]:


if is_debug:
    # 查看已经编码的文件
    tag = 'test'
    embedding_res_file = f'data/{tag}_dnabert_seq_embedding.pkl'
    embedding_label_file = f'data/{tag}_dnabert_label_embedding.pkl'
    
    with open(embedding_res_file, 'rb') as file:
        embedding_res_dict = pickle.load(file)
    
    with open(embedding_label_file, 'rb') as file:
        embedding_label_dict = pickle.load(file)
    print(len(embedding_label_dict.keys()))
    key = list(embedding_label_dict.keys())[0]
    print(embedding_res_dict[key].shape, embedding_label_dict[key].shape)


# ## Training model

# ### Dataset

# In[10]:


from torch.utils.data import Dataset, DataLoader

class EmbeddingDataset(Dataset):
    def __init__(self, tag='train'):

        super(EmbeddingDataset, self).__init__()

        # load data from the disk
        embedding_res_file = f'data/{tag}_dnabert_seq_embedding.pkl'
        embedding_label_file = f'data/{tag}_dnabert_label_embedding.pkl'

        with open(embedding_res_file, 'rb') as file:
            self.embedding_res_dict = pickle.load(file)

        with open(embedding_label_file, 'rb') as file:
            self.embedding_label_dict = pickle.load(file)
        
        # gene names
        self.gene_names = list(self.embedding_label_dict.keys())

    def __len__(self):
        return len(self.embedding_res_dict.keys())

    def __getitem__(self, i):
        gene_name = self.gene_names[i]
        
        return self.embedding_res_dict[gene_name], self.embedding_label_dict[gene_name]

if is_debug:
    test_dataset = EmbeddingDataset(tag='test')
    print(test_dataset[0])
    
    # test dataloader
    batch_size = 4
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    batch_X, batch_y = next(iter(test_dataloader))
    print('batch_X shape:', batch_X.shape)
    print('batch_y shape:', batch_y.shape)
    print(batch_X)
    print(batch_y)


# ### Build model

# In[11]:


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


# In[12]:


if is_debug:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    
    mlp_parma = {
        'input_size': 768,
        'hidden_sizes': [768, 256, 128, 64],
        'output_size': 1,
        'dropout_prob': 0.5
    }
    model = MLP(mlp_parma).to(device)
    print(model)
    
    inputs = batch_X.to(device)

    with torch.no_grad():
        output = model(inputs)

    print(output.shape, output)


# ### Train Test Loop

# In[13]:


from tqdm.auto import tqdm
import numpy as np

import torch
from sklearn.metrics import mean_squared_error, r2_score

def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, device):        
    finish_step_num = (epoch-1)*len(dataloader)
    loss_total, rmse_total, r2_total, step_count = 0, 0, 0, 0
    
    model.train()
    for batch_X, batch_y in dataloader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        pred = model(batch_X)
        
        # tensor[batch_size, 1] -> tensor[batch_size]
        pred = torch.squeeze(pred)

        loss = loss_fn(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
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
        
    loss, rmse, r2 = loss_total / step_count, rmse_total / step_count, r2_total / step_count
    
    return [loss, rmse, r2]


# In[14]:


def test_loop(dataloader, model, loss_fn, epoch, device, mode='Test'):
    assert mode in ['Valid', 'Test']
        
    finish_step_num = (epoch-1)*len(dataloader)
    
    loss_total, rmse_total, r2_total, step_count = 0, 0, 0, 0

    model.eval()
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            pred = model(batch_X)
            
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

    loss, rmse, r2 = loss_total / step_count, rmse_total / step_count, r2_total / step_count
    
    return [loss, rmse, r2]


# ### Normal training

# In[15]:


# 0: normal training
training_methods == 0
if training_methods == 0 and not is_debug:
    import os
    import sys
    from transformers import AdamW, get_scheduler
    
    from tqdm.auto import tqdm

    # hyper-parameters
    batch_size = 64
    learning_rate = 1e-5
    epoch_num = 100
    tag = f'{batch_size}-{learning_rate}-{epoch_num}'
    print(f'Tag: {tag}')

    # path: model and train_result_data
    model_bin = f'data/mlp-{tag}.bin'
    train_result_data = f'data/Train-{tag}.csv'
    if os.path.exists(train_result_data):
        os.remove(train_result_data)

    # dataset
    train_dataset, test_dataset= EmbeddingDataset('train'), EmbeddingDataset('test')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # model
    mlp_parma = {
        'input_size': 768,
        'hidden_sizes': [768, 256, 128, 64],
        'output_size': 1,
        'dropout_prob': 0.5
    }
    model = MLP(mlp_parma)
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
    loss_fn = nn.MSELoss()

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
    progress_bar = tqdm(range(epoch_num))
    progress_bar.set_description(f'Trian R^2: {0:>7f}')
    
    bast_r2 = -sys.maxsize - 1
    for t in range(epoch_num):
        # print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
        train_data = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t+1, device)
        test_data = test_loop(test_dataloader, model, loss_fn, t+1, device, mode='Test')

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
            
        progress_bar.set_description(f'Trian R^2: {test_r2:>7f}')
        progress_bar.update(1)

    print("Done!")


# ### Bayesian opt

# In[16]:


# 1: bayesian opt
# training_methods = 1
if training_methods == 1 and not is_debug:
    import os
    import sys
    from transformers import AdamW, get_scheduler
    import optuna
    
    from tqdm.auto import tqdm
    # from tqdm.autonotebook import tqdm
    # from tqdm.notebook import tqdm
    
    # 设置 Optuna 的日志级别为 WARNING
    optuna.logging.set_verbosity(optuna.logging.INFO)
    
    # bayesian opt
    def objective(_trial):
        # 对数均匀分布的下界。通常是超参数的最小值，最大值
        lr = _trial.suggest_float('lr', 1e-5, 1, log=True)
        hidden_dim = [
            int(_trial.suggest_float('hidden_layer_1', 256, 768, log=True)),
            int(_trial.suggest_float('hidden_layer_2', 256, 512, log=True)),
            int(_trial.suggest_float('hidden_layer_3', 64, 384, log=True)),
            int(_trial.suggest_float('hidden_layer_4', 32, 256, log=True))
        ]
        dropout = _trial.suggest_float('dropout', 0.3, 0.6, log=True)

        # hyper-parameters
        batch_size = 64
        learning_rate = lr
        epoch_num = 100

        # dataset
        train_dataset, test_dataset= EmbeddingDataset('train'), EmbeddingDataset('test')
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        # model
        mlp_parma = {
            'input_size': 768,
            'hidden_sizes': hidden_dim,
            'output_size': 1,
            'dropout_prob': dropout
        }
        model = MLP(mlp_parma)
        # print(model)

        # cuda
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
#             print(f"Let's use, {torch.cuda.device_count()}, GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model)
        else:
#             print(f"Let's use, {device}!")

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
        bast_r2 = -sys.maxsize - 1
        
        progress_bar = tqdm(range(epoch_num))
        progress_bar.set_description(f'Trian R^2: {0:>7f}')
        for t in range(epoch_num):
            # print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
            train_data = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t+1, device)
            test_data = test_loop(test_dataloader, model, loss_fn, t+1, device, mode='Test')

            # record bast model
            test_r2 = test_data[-1]
            if bast_r2 < test_r2:
                bast_r2 = test_r2
            
            progress_bar.set_description(f'Trian R^2: {test_r2:>7f}')
            progress_bar.update(1)
                
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

