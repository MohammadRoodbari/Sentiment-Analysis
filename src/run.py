import pandas as pd
import torch.nn as nn
from nltk.tokenize import word_tokenize
from datasets import Dataset
from steps.TextPreProcess import *
from steps.glove_embedding import glove_embedding
from steps.model import LSTM
from steps.train_eval import *
from steps.train_eval import *

DataSet = pd.read_csv('../data/training.1600000.processed.noemoticon.csv' ,encoding = "ISO-8859-1" , names = ['label','id','date','the query','user','text'])

for i in DataSet.index:
  if DataSet['label'][i] == 4 :
    DataSet['label'][i] =1

ds = Dataset.from_pandas(DataSet)
ds = ds.train_test_split(test_size=0.2 , shuffle = True)
ds_train_valid = ds["train"].train_test_split(test_size=0.15 , shuffle = True)
train_dataset = ds_train_valid["train"]
validation_dataset = ds_train_valid["test"]
test_dataset = ds["test"]


glove_embedding = glove_embedding()
vocabs, embeddings, word_to_index = glove_embedding.get_embedding()

TextPreProcessor = TextPreProcessPipeLine(
    ConvertCase('lower'),
    RemoveUserHandle(),
    Removehttplinks(),
    RemoveDigit(),
    RemoveSpace(),
    Stemmer(),
    Lemmatizer(),
    RemoveStopWords()
)

def get_ids(txt):
  tokenized_txt = word_tokenize(txt)
  ids = []
  for token in tokenized_txt:
    id = word_to_index.get(token,-1)
    if id != -1:
      ids.append(id)
    else:
        ids.append(word_to_index["<unk>"])
  return ids

def preprocess(data):
  max_len = 512
  comment = TextPreProcessor.transform(data['text'])
  label = data["label"]
  comment_ids = get_ids(comment)

  return { "input_ids" : comment_ids[:max_len]  , "label": label}

train_dataset = train_dataset.map(preprocess ,remove_columns = train_dataset.column_names)
validation_dataset = validation_dataset.map(preprocess,remove_columns = validation_dataset.column_names)
test_dataset = test_dataset.map(preprocess,remove_columns = test_dataset.column_names)

train_dataset = train_dataset.shuffle(seed = 42)

train_dataset.set_format(type='torch')
validation_dataset.set_format(type='torch')
test_dataset.set_format(type='torch')

epoch = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTM(embedding_vector = embeddings)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()
train_loader = torch.utils.data.DataLoader(train_dataset , batch_size=16 , collate_fn=collate_batch,drop_last = True)
valid_loader = torch.utils.data.DataLoader(validation_dataset , batch_size=16 , collate_fn=collate_batch,drop_last = True)
test_loader = torch.utils.data.DataLoader(test_dataset , batch_size=4 , drop_last=True , collate_fn=collate_batch )

y_true , y_pred = train(model , optimizer , loss_fn , train_loader  , valid_loader , test_loader , epoch , device )