import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from custom_datasets import CustomDataset
from tqdm import tqdm
import os
import glob
import pandas as pd
from pathlib import Path
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from skimage import io
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sn

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print("The model will be running on", device, "device\n")


def csv_file(root_dir):
  """
  cats-dogs dataset, makes csv file based on their first 3 chars in filename

  """
  # image paths to csv file
  # image_name - label

  df = pd.DataFrame(columns=['image_path', 'class'])
  # print(df.columns)

  for img in Path(root_dir).glob("*.jpg"):
    # print(type(img)) #<class 'pathlib.WindowsPath'>
    row = pd.DataFrame([{'image_path': os.path.basename(str(img)), 'class': img.stem[:3]}])  # img
    df = df.append(row, ignore_index=True)
    # print(img.stem[:3])

  # print(tuple(np.unique(df['class'])))
  label_names = tuple(np.unique(df['class']))

  le = preprocessing.LabelEncoder()
  le.fit(df['class'])

  df['categorical_label'] = le.transform(df['class'])
  #print(df.tail())
  dataset_csv = df.to_csv("./data/dogs-cats.csv", index=False)
  return dataset_csv, label_names


def saveModel(model):

  path = "./trained_models/dogs-cats.pth"
  torch.save(model.state_dict(), path)


def train(num_epochs, train_loader, val_loader, model):
  model.to(device)
  best_accuracy = 0.0
  criterion = nn.CrossEntropyLoss() #loss func
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  train_loss_value = []
  val_loss_value = []
  total_acc = []

  for epoch in range(num_epochs):
    running_train_loss = []
    running_val_loss = []
    running_accuracy = 0.0
    total = 0

    for batch_idx, (data,targets) in enumerate(tqdm(train_loader)):
      data, targets = data.to(device), targets.to(device)
      # data = data.reshape(data.shape[0], -1)
      # print(data.shape) #torch.Size([64, 784])

      # #clear gradients
      optimizer.zero_grad()

      # forward
      # data = data.unsqueeze(1)
      # data = data.view(data.size(0), -1)
      predicts = model(data)

      #compute loss
      train_loss = criterion(predicts, targets) # her 1 data icin  train loopunda loss hesaplanır

      #backward
      train_loss.backward()

      # update
      optimizer.step()

      #running_train_loss += train_loss.item() # track the loss value. bu train lossları toplayarak 1epochtaki train loss u bulur
      running_train_loss.append(train_loss.item())

    # Calculate training loss value
    #train_loss_value = running_train_loss/len(train_loader) # ? batch size için loss hesaplar
    train_loss_value.append(np.array(running_train_loss).mean())
    # print(f"Train loss of this batch: {train_loss_value[-1]}")

    # Validation Loop
    with torch.no_grad():
      model.eval()
      for _, (val_data,val_targets) in enumerate(val_loader):
        inputs, outputs = val_data.to(device), val_targets.to(device)
        predicted_outputs = model(inputs)
        val_loss = criterion(predicted_outputs, outputs)

        # The label with the highest value will be our prediction
        _, predicted = torch.max(predicted_outputs,dim=1) # find the maximum along the rows, use dim=1.
        # running_val_loss += val_loss.item()
        running_val_loss.append(val_loss.item())

        total += outputs.size(0)
        running_accuracy += (predicted == outputs).sum().item()

    # Calculate validation loss value
    # val_loss_value = running_val_loss/len(val_loader)
    val_loss_value.append(np.array(running_val_loss).mean())
    # print(f"Val loss of this batch: {val_loss_value[-1]}")

    # Calculate accuracy as the number of correct predictions in the validation batch divided by the total number of predictions done
    accuracy = (100 * running_accuracy / total)
    # print(f"Accuracy of this batch: {accuracy}")
    # print(type(accuracy))
    total_acc.append(accuracy)

    # Save model if the accuracy is the best
    if accuracy > best_accuracy:
      saveModel(model)
      best_accuracy = accuracy


    # Print the statistics of the epoch
    #print('Completed training batch', epoch, 'Training loss is: %.4f' %train_loss_value, 'Validation loss is: %.4f' %val_loss_value, 'Accuracy is %d %%' %(accuracy))


  plt.figure(figsize=(10, 5))
  plt.title("Train - Validation Loss")
  plt.plot(train_loss_value, label="train")
  plt.plot(val_loss_value, label="validation")
  plt.xlabel("iterations")
  plt.ylabel("Loss")
  plt.savefig('train-val-loss.png')

  plt.figure(figsize=(10, 5))
  plt.title("Accuracy")
  plt.plot(total_acc, label="accuracy")
  plt.xlabel("iterations")
  plt.ylabel("Accuracy")
  plt.savefig('acc.png')

  plt.legend()
  plt.show()

# testing test data
def test(custom_model, test_loader, labels):
  # load the model that we saved at the end of the training loop
  path = "./trained_models/dogs-cats.pth"
  custom_model.load_state_dict(torch.load(path))


  running_accuracy = 0
  total = 0
  y_pred = []
  y_true = []

  with torch.no_grad():
    custom_model.eval()
    for data,targets in (test_loader):
      # inputs, outputs = data.to("cpu"), targets.to("cpu")
      inputs, outputs = data.to(device), targets.to(device)

      predicted_outputs = custom_model(inputs)
      _, predicted = torch.max(predicted_outputs.data, 1)
      y_pred.extend(predicted)

      #targets=targets.cpu().data.numpy()
      # targets = targets.detach().cpu().numpy()
      y_true.extend(targets)

      total += outputs.size(0)
      running_accuracy += (predicted == outputs).sum().item()

    print('Accuracy of the model based on the test set: %d %%' % (100 * running_accuracy / total) )

    y_true = [t.cpu().numpy() for t in y_true]
    y_true = np.vstack(y_true)

    y_pred = [t.cpu().numpy() for t in y_pred]
    y_pred = np.vstack(y_pred)

    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in labels], columns=[i for i in labels])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('dogs-cats-cm.png')
    plt.plot()

    plt.show()

    # print(y_true)
    # print(y_pred)
    # print(cf_matrix)


# def check_accuracy(loader, model):
#   if loader.dataset.dogs-cats:
#     print("Checking accuracy on training data")
#   else:
#     print("Checking accuracy on test data")
#
#   num_correct = 0
#   num_samples = 0
#   model.eval()
#
#   with torch.no_grad():
#     for x,y in loader:
#       x = x.to(device)
#       y = y.to(device)
#       x = x.reshape(x.shape[0], -1)
#
#       scores = model(x)
#       # 64x10
#       _, predictions = scores.max(1) # 1 burada 2.dimension oluyor
#       num_correct += (predictions == y).sum()
#       num_samples += predictions.size(0)
#
#     print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")
#
#   model.dogs-cats()
#
# check_accuracy(train_loader, model)
# check_accuracy(test_loader, model)

if __name__ == "__main__":

  # HYPERPARAMETERS
  in_channels = 3
  num_classes = 2
  learning_rate = 1e-3
  batch_size = 16
  num_epochs = 100
  image_size = (224, 224)
  root_dir = "./data/dogs-cats/"
  train_ratio = 0.7
  val_ratio = 0.2
  test_ratio = 0.1


  _, labels = csv_file(root_dir)

  transform = transforms.Compose([
    transforms.Resize(size=image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
  ])

  dataset = CustomDataset(root_dir=root_dir, annotations="./data/dogs-cats.csv", transform=transform)

  total_size = len(dataset)
  train_size = int(total_size * train_ratio)
  val_size = int(total_size * val_ratio)
  test_size = total_size - train_size - val_size

  train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

  train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)
  test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

  my_model = torchvision.models.googlenet(pretrained=True)

  train(num_epochs=num_epochs, train_loader=train_loader, val_loader=val_loader, model=my_model)
  print('Finished Training\n')

  test(custom_model=my_model, test_loader=test_loader, labels=labels)
