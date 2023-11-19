import torch
import torch.nn as nn
import torch.optim as optim
from util.data_def import *

class Solver(object):
    def __init__(self, model, epochs, trainloader, device, learning_rate=0.001):

        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        self.epochs = epochs
        self.device = device
        self.training_loader = trainloader

    def train(self):

        
        train_acc = 0.
        
        for epoch in range(self.epochs):

          running_loss = 0.
          last_loss = 0.

          total_samples = 0
          total_correct = 0

          # Read the dataloader
          for i, (image, label) in enumerate(self.training_loader):
            
            image = image.to(self.device)
            label = label.to(self.device)

            # Zero gradients for every batch
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(image.float())
            _, pred = torch.max(outputs, 1)

            total_correct += (pred == label).sum().item()
            total_samples += label.size(0)

            # Compute the loss and its gradients
            loss = self.criterion(outputs, label)

            # Backpropagation
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            
            # Print loss in every 10 batches
            if i % 10 == 9:
              last_loss = running_loss / 10
              print("Epoch: ", epoch, " Batch : ", i+1, " Loss: ", last_loss)
              running_loss = 0.
          # Save the model in the end of epoch
          torch.save(self.model.state_dict(), current_directory + f'/save_model/seg_network_{epoch}.pt')

          print("################################################################")
          print("Epoch:{}  Training Loss:{}".format(epoch, last_loss))
          print("################################################################")

          # Print the Training accuracy while end of training
          if epoch == 9:
            train_acc = 100 * total_correct / total_samples
            print("Training Accuracy", train_acc)