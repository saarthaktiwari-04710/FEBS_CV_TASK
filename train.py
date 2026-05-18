from torchvision import models
import model.py
from dataset.py import get_dataloaders

trainloader, testloader= get_dataloaders()

def train(num_epochs=10):
  train_losses, train_acc_list, test_acc_list = [], [], []

  for epoch in range(num_epochs):
      model.train()
      running_loss = 0.0
      correct, total = 0, 0
      for inputs, labels in trainloader:
          inputs, labels = inputs.to(device), labels.to(device)
          optimizer.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          
          running_loss += loss.item() * inputs.size(0)
          _, predicted = outputs.max(1)
          total += labels.size(0)
          correct += predicted.eq(labels).sum().item()
      
      train_loss = running_loss / len(trainloader.dataset)
      train_acc = 100. * correct / total
      train_losses.append(train_loss)
      train_acc_list.append(train_acc)

      model.eval()
      correct, total = 0, 0
      with torch.no_grad():
          for inputs, labels in testloader:
              inputs, labels = inputs.to(device), labels.to(device)
              outputs = model(inputs)
              _, predicted = outputs.max(1)
              total += labels.size(0)
              correct += predicted.eq(labels).sum().item()
      test_acc = 100. * correct / total
      test_acc_list.append(test_acc)
      
      scheduler.step()
      print(f'Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%')

  return train_losses, train_acc_list, test_acc_list

train_losses, train_acc_list, test_acc_list=train()


import matplotlib.pyplot as plt
#Plotting graphs of loss vs epoch and trainaccuracy,testaccuracy vs epoch
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_acc_list, label='Train Accuracy')
plt.plot(test_acc_list, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy')
plt.legend()

plt.show()
