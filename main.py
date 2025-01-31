import pennylane as qml 
import torch
import numpy as np
from sklearn.datasets import make_moons
from torchvision import datasets, transforms
import qneuron

# Create training and test set (X and y)
X, y = make_moons(n_samples=200, noise=0.1)
y_ = torch.unsqueeze(torch.tensor(y), 1)  # used for one-hot encoded labels
y_hot = torch.scatter(torch.zeros((200, 2)), 1, y_, 1)
c = ["#1f77b4" if y_ == 0 else "#ff7f0e" for y_ in y]
print("Download the MNIST dataset")
# Download the MNIST dataset, train samples
n_samples = 100
X_train = datasets.MNIST(root='./data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor()]))
print("Leave only labels 0 and 1")
# Leaving only labels 0 and 1 
idx = np.append(np.where(X_train.targets == 0)[0][:n_samples],np.where(X_train.targets == 1)[0][:n_samples])
X_train.data = X_train.data[idx]
X_train.targets = X_train.targets[idx]
train_loader = torch.utils.data.DataLoader(X_train, batch_size=1, shuffle=True)

# Download the MNIST dataset, test samples
n_samples = 50
X_test = datasets.MNIST(root='./data', train=False, download=True,transform=transforms.Compose([transforms.ToTensor()]))
idx = np.append(np.where(X_test.targets == 0)[0][:n_samples],np.where(X_test.targets == 1)[0][:n_samples])
X_test.data = X_test.data[idx]
X_test.targets = X_test.targets[idx]
test_loader = torch.utils.data.DataLoader(X_test, batch_size=1, shuffle=True)

print("Define the hybrid model")
# Hybrid Model #
#Basic model structure -> classic layer 2 nodes --- quantum layer 1 node --- classic layer 2 node 
n_layers = 6
nQbits=2
weight_shapes = {"weights": (n_layers, nQbits)}
#define a quantum_layer
qlayer = qml.qnn.TorchLayer(qneuron.qnode, weight_shapes)

#layer_classico(2 nodi)->layer quantum(1 node)->layer_classico(2 nodi)#
#define 2 classical layers
classical_layer_1 = torch.nn.Linear(2, 2)
classical_layer_2 = torch.nn.Linear(2, 2)
softmax = torch.nn.Softmax(dim=1)
layers = [classical_layer_1, qlayer, classical_layer_2, softmax]
model = torch.nn.Sequential(*layers)# this method is used to merge serveral layers passed 
# define an optimizer to change the weights  
opt = torch.optim.SGD(model.parameters(), lr=0.2)
# define the type of loss function
loss = torch.nn.L1Loss()

#Training the model 
X = torch.tensor(X, requires_grad=True).float()
y_hot = torch.tensor(y_hot)

batches = 200 
data_loader = torch.utils.data.DataLoader(list(zip(X, y_hot)), batch_size=5, shuffle=True, drop_last=True)
print("Training the model...")
epochs = 50
for epoch in range(epochs):
    running_loss = 0
    for xs, ys in data_loader:
        opt.zero_grad()
        loss_evaluated = loss(model(xs), ys)
        loss_evaluated.backward()
        opt.step()
        running_loss += loss_evaluated
    avg_loss = running_loss / batches
    print("Average loss over epoch {}: {:.4f}".format(epoch + 1, avg_loss))

# testing the model
y_pred = model(X)
predictions = torch.argmax(y_pred, axis=1).detach().numpy()
correct = [1 if p == p_true else 0 for p, p_true in zip(predictions, y)]
accuracy = sum(correct) / len(correct)
print(f"Accuracy: {accuracy * 100}%")
