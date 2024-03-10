import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
from torch.nn import Transformer
import torchtext
from torchtext.data.utils import get_tokenizer

from collections import Counter
from torchtext.vocab import Vocab
from torch.nn.utils.rnn import pad_sequence
# 检查 CUDA（GPU）是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定義位置編碼
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(int(max_len), 1, int(d_model))

        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

# 定義 Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, ntoken_src, ntoken_tgt, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.src_embed = nn.Embedding(ntoken_src, ninp)
        self.tgt_embed = nn.Embedding(ntoken_tgt, ninp)
        self.pos_encoder = PositionalEncoding(ninp, dropout)

        # Convolutional layers
        self.conv1d = nn.Conv1d(in_channels=ninp, out_channels=ninp, kernel_size=3, padding=3 // 2)
        # LSTM layers
        self.lstm = nn.LSTM(ninp, ninp, num_layers=2, batch_first=True)

        self.transformer = nn.Transformer(ninp, nhead, nlayers, nlayers, nhid, dropout)
        self.ninp = ninp
        # Dropout
        self.dropout = nn.Dropout(dropout)
 
        self.layer_norm = nn.LayerNorm(ninp)
        self.decoder = nn.Linear(ninp, ntoken_tgt)

    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        src = self.src_embed(src) * math.sqrt(self.ninp)
        tgt = self.tgt_embed(tgt) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        # Apply convolution
        src = src.transpose(1, 2)  # Conv1d expects (batch, channels, seq_len)
        src = self.conv1d(src)
        src = src.transpose(1, 2)  # Back to (batch, seq_len, channels)

        tgt = tgt.transpose(1, 2)
        tgt = self.conv1d(tgt)
        tgt = tgt.transpose(1, 2)

        # Apply dropout
        src = self.dropout(src)
        tgt = self.dropout(tgt)

        # Pass through LSTM
        src, _ = self.lstm(src)
        tgt, _ = self.lstm(tgt)

        output = self.transformer(src, tgt, src_mask, tgt_mask, None, src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        # Pass through LSTM
        output, _ = self.lstm(output)

        output = self.dropout(output)

        output = self.layer_norm(output + tgt)  # 添加残差连接和层归一化
        output = self.decoder(output)
        return output

# 數據預處理
# 這裡需要根據你的數據集具體情況來定義詞彙表和數據轉換方式

# 定義掩碼
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# 定義數據集
class TranslationDataset(Dataset):
    def __init__(self, src_data, tgt_data):
        self.src_data = src_data
        self.tgt_data = tgt_data

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]

# 加載數據
zh_data = pd.read_csv('train-ZH.csv')['txt'].tolist()
tl_data = pd.read_csv('train-TL.csv')['txt'].tolist()


import spacy
import torchtext.data as data

# 加载 spaCy 的中文模型
spacy_zh = spacy.load('zh_core_web_sm')

def tokenize_zh(text):
    """
    Tokenizes Chinese text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in spacy_zh.tokenizer(text)]

# 使用自定义的分词函数
zh_tokenizer = tokenize_zh
zh_tokens = [zh_tokenizer(sentence) for sentence in zh_data]

# 台語羅馬拼音标记化
tl_tokenizer = get_tokenizer('basic_english')
tl_tokens = [tl_tokenizer(sentence) for sentence in tl_data]


from torchtext.vocab import build_vocab_from_iterator

def yield_tokens(data_iter):
    for data_sample in data_iter:
        yield data_sample

# 构建中文词汇表
zh_vocab = build_vocab_from_iterator(yield_tokens(zh_tokens), specials=['<unk>', '<pad>', '<bos>', '<eos>'])
zh_vocab.set_default_index(zh_vocab['<unk>'])

# 构建台語羅馬拼音词汇表
tl_vocab = build_vocab_from_iterator(yield_tokens(tl_tokens), specials=['<unk>', '<pad>', '<bos>', '<eos>'])
tl_vocab.set_default_index(tl_vocab['<unk>'])


def data_process(src_tokens, tgt_tokens, src_vocab, tgt_vocab):
    data = []
    for (src_sentence, tgt_sentence) in zip(src_tokens, tgt_tokens):
        src_tensor = torch.tensor([src_vocab[token] for token in src_sentence], dtype=torch.long)
        tgt_tensor = torch.tensor([tgt_vocab[token] for token in tgt_sentence], dtype=torch.long)
        data.append((src_tensor, tgt_tensor))
    return data

processed_data = data_process(zh_tokens, tl_tokens, zh_vocab, tl_vocab)


def generate_batch(data_batch):
    src_batch, tgt_batch = [], []
    for (src_item, tgt_item) in data_batch:
        src_batch.append(torch.cat([torch.tensor([zh_vocab['<bos>']]), src_item, torch.tensor([zh_vocab['<eos>']])], dim=0))
        tgt_batch.append(torch.cat([torch.tensor([tl_vocab['<bos>']]), tgt_item, torch.tensor([tl_vocab['<eos>']])], dim=0))
    src_batch = pad_sequence(src_batch, padding_value=zh_vocab['<pad>'])
    tgt_batch = pad_sequence(tgt_batch, padding_value=tl_vocab['<pad>'])
    return src_batch, tgt_batch

BATCH_SIZE = 128
dataloader = DataLoader(processed_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)



ntoken_src = len(zh_vocab)  # 源语言词汇表大小
ntoken_tgt = len(tl_vocab)  # 目标语言词汇表大小
ninp = 1024                 # 嵌入层大小
nhead = 16                   # 多头注意力头数
nhid = 3072                 # 前馈网络维度
nlayers = 18                 # Transformer层数
dropout = 0.2             # Dropout比率

model = TransformerModel(ntoken_src, ntoken_tgt, ninp, nhead, nhid, nlayers, dropout)
# 将模型移至 GPU
model = model.to(device)
# 定義優化器和損失函數
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1, verbose=True)
#criterion = nn.CrossEntropyLoss()
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = nn.functional.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

# 使用 Label Smoothing
criterion = LabelSmoothingLoss(smoothing=0.25)

from tqdm import tqdm

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(iterator, total=len(iterator), desc='Training', leave=False)

    for _, (src, tgt) in enumerate(progress_bar):
        src = src.to(device)
        tgt = tgt.to(device)
        src_mask = generate_square_subsequent_mask(src.size(0))
        tgt_mask = generate_square_subsequent_mask(tgt.size(0) - 1)

        optimizer.zero_grad()
        output = model(src, tgt[:-1, :], src_mask, tgt_mask, None, None, None)
        output_dim = output.shape[-1]
        output = output.view(-1, output_dim)
        tgt = tgt[1:].view(-1)

        loss = criterion(output, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for _, (src, tgt) in enumerate(iterator):
            src = src.to(device)
            tgt = tgt.to(device)
            src_mask = generate_square_subsequent_mask(src.size(0))
            tgt_mask = generate_square_subsequent_mask(tgt.size(0) - 1)

            output = model(src, tgt[:-1, :], src_mask, tgt_mask, None, None, None)
            output_dim = output.shape[-1]
            output = output.view(-1, output_dim)
            tgt = tgt[1:].view(-1)

            loss = criterion(output, tgt)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# Training and Evaluation Loop
for epoch in range(60):
    train_loss = train(model, dataloader, optimizer, criterion, 1)
    eval_loss = evaluate(model, dataloader, criterion)  # Assuming you have a validation dataloader
    scheduler.step(eval_loss)
    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}')