import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
  def __init__(self , embedding_vector):
    super(LSTM,self).__init__()
    self.word_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(embedding_vector).float())
    self.lstm = nn.LSTM(300,150,num_layers=2,dropout=0.1,batch_first=True)
    self.dropout = nn.Dropout(p=0.2)
    self.fc = nn.Linear(150, 64)
    self.act_func = nn.ReLU()
    self.fc2 = nn.Linear(64, 2)

  def forward(self,input_ids):
    embeddings = self.word_embeddings(input_ids)
    out , _ = self.lstm(embeddings)
    out = out[: , -1 , :]
    out = self.act_func(self.fc(out))
    out = self.fc2(self.dropout(out))
    return out