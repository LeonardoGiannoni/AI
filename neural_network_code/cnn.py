import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import neural_net as n

def load_training_data(batch_size, n_samples_show):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    #Download the MNIST dataset
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True,transform=transform)
    #Create a DataLoader for test and train samples object to load data in batches
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #use iter() to make data iterable 
    data_iter = iter(train_loader)

    fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))
    while n_samples_show > 0:
        images, targets = data_iter.__next__() #get the next batch of images and targets

        axes[n_samples_show - 1].imshow(images[0].numpy().squeeze(), cmap='gray')
        axes[n_samples_show - 1].set_xticks([])
        axes[n_samples_show - 1].set_yticks([])
        axes[n_samples_show - 1].set_title(f"Label: {targets[0].item()}")
        n_samples_show -= 1
    plt.show()
    return train_loader

def load_test_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    #Download the MNIST dataset
    test_dataset = datasets.MNIST('./data', train=False, download=True,transform=transform)
    #Create a DataLoader for test and train samples object to load data in batches
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    return test_loader

def model_traning(epochs, optimizer, loss_function, train_loader,total_loss):
    model.train() #set the model to training mode
    loss_list = []
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            # Forward pass
            output = model(data)
            # Calculating loss
            loss = loss_func(output, target)
            # Backward pass
            loss.backward()
            # Optimize the weights
            optimizer.step()
            
            total_loss.append(loss.item())
        loss_list.append(sum(total_loss)/len(total_loss))
        print('Training [{:.0f}%]\tLoss: {:.4f}'.format(100. * (epoch + 1) / epochs, loss_list[-1]))
    return loss_list 

def model_testing(test_loader,total_loss):
    model.eval() #set the model to evaluation mode
    with torch.no_grad():
        correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        output = model(data)
        #predict the label of each image
        pred = output.argmax(dim=1, keepdim=True) 
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss = loss_func(output, target)
        total_loss.append(loss.item())
    return  (total_loss, correct)

if __name__ == "__main__":
    batch_size = 64
    n_samples_show = 6
    n_traning_epochs = 50
    total_loss = []
    print("Downloading MNIST dataset...")
    train_loader_data = load_training_data(batch_size, n_samples_show)
    test_loader_data = load_test_data()
    #Create a neural network
    print("Creating a neural network...\n")
    model = n.Net()
    
    #Define an optimizer and a loss function, then start traning the model with model_traning()
    print("Define an optmizer and a loss function...\n")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.CrossEntropyLoss()
    
    # Train the model
    print("Training the model...")
    training_result = model_traning(n_traning_epochs, optimizer, loss_func, train_loader_data,total_loss)
    print("Model training completed!\n")

    #Plot the training results
    plt.plot(training_result)
    plt.title('CNN Training Performance')
    plt.xlabel('Training Iterations')
    plt.ylabel('Neg Log Likelihood Loss')
    plt.show()

    #test the model with model_testing()
    print("Testing the model...")
    
    total_loss, correct = model_testing(test_loader_data,total_loss)
    print('Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%'.format(sum(total_loss) / len(total_loss),correct / len(test_loader_data) * 100))