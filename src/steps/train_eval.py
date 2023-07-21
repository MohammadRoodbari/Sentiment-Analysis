import torch
from tqdm.notebook import tqdm_notebook
from sklearn.metrics import accuracy_score, classification_report


def train(model , optimizer , loss_fn , train_loader , valid_loader , test_loader , epochs , device = "cuda"):

  model.to(device)
  for epoch in range(epochs):
    training_loss = 0.0
    model.train()
    t_labels = []
    t_preds = []
    for batch in tqdm_notebook(train_loader):
      optimizer.zero_grad()
      input_ids = batch["input_ids"].to(device)
      labels = batch["label"].to(device)
      preds = model(input_ids = input_ids  )
      loss = loss_fn(preds , labels.to(torch.long))
      t_labels += labels.cpu().numpy().tolist()
      loss.backward()
      optimizer.step()
      training_loss += loss.item()
      t_preds +=  preds.argmax(axis=1).cpu().numpy().tolist()

    training_loss = training_loss / len(train_loader)
    train_accuracy =  accuracy_score(t_labels, t_preds, normalize=True)
    print('Epoch: {} ,training loss: {:.2f} , train accuracy: {:.2f} '.format(epoch , training_loss ,train_accuracy))

    validation_loss , validation_accuracy = validation_loss_accuracy(model , valid_loader , loss_fn , device  )
    print('validation loss: {:.2f} , validation accuracy: {:.2f} '.format(validation_loss ,validation_accuracy))

  test_loss , test_accuracy , y_true , y_pred = validation(model , test_loader , loss_fn , device  )
  print('test loss: {:.2f}  test accuracy: {:.2f}'.format( test_loss , test_accuracy))
  return y_true , y_pred

def validation( model , val_loader , loss_fn , device = "cuda"  ):
  val_loss = 0.0
  model.eval()
  reals = []
  preds_list = []
  for batch in val_loader:
    input_ids = batch["input_ids"].to(device)
    labels = batch["label"].to(device)
    with torch.no_grad():
      preds = model(input_ids = input_ids )
    loss = loss_fn(preds , labels.to(torch.long))
    val_loss += loss.item()
    reals += labels.cpu().numpy().tolist()
    preds_list += preds.argmax(axis=1).cpu().numpy().tolist()

  val_loss = val_loss / len(val_loader)
  accuracy = accuracy_score(reals, preds_list, normalize=True)
  print(classification_report(reals , preds_list ))
  return val_loss ,accuracy, reals, preds_list

def validation_loss_accuracy( model , val_loader , loss_fn , device = "cuda"  ):
  val_loss = 0.0
  model.eval()
  reals = []
  preds_list = []
  for batch in val_loader:
    input_ids = batch["input_ids"].to(device)
    labels = batch["label"].to(device)
    with torch.no_grad():
      preds = model(input_ids = input_ids )
    loss = loss_fn(preds , labels.to(torch.long))
    val_loss += loss.item()
    reals += labels.cpu().numpy().tolist()
    preds_list += preds.argmax(axis=1).cpu().numpy().tolist()

  val_loss = val_loss / len(val_loader)
  accuracy = accuracy_score(reals, preds_list, normalize=True)

  return val_loss ,accuracy

def collate_batch(batch):
  input_ids =[]
  labels = []
  for b in batch:
    input_ids.append(b["input_ids"])
    labels.append(b["label"])
  input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0.0)
  return {"input_ids":input_ids ,"label":torch.Tensor(labels)}