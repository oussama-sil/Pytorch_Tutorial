import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# device info
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
input_size = 28*28 #size of images 
hidden_size = 100
num_classes = 10
num_epochs = 10
batch_size = 100
lr = 0.001

# MNIST
train_dataset = torchvision.datasets.MNIST(root='./data',train=True,transform=transforms.ToTensor(),download=True)
test_dataset = torchvision.datasets.MNIST(root='./data',train=False,transform=transforms.ToTensor(),download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

#Printing samples 
print("--Size of one batch of data--")
example = iter(train_loader)
samples, labels = next(example)
print(samples.shape,labels.shape)

# plotting
for i in range(6):
    plt.subplot(2,3,i+1) #2 rows with 3 columns at index i+1
    plt.imshow(samples[i][0],cmap='gray')


# plt.show()

class NN(torch.nn.Module):
    """Some Information about MyModule"""
    def __init__(self,input_size,hidden_size,num_classes):
        super(NN, self).__init__()
        self.lin1 = torch.nn.Linear(input_size,hidden_size)
        self.relu = torch.nn.ReLU()
        self.lin2 = torch.nn.Linear(hidden_size,num_classes)

    def forward(self, x):
        out = self.lin1(x)
        out = self.relu(out)
        out = self.lin2(out)

        #! No softmax 

        return out
    

model = NN(input_size,hidden_size,num_classes).to(device)

#Optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# Training loop 
n_total_steps = len(train_loader)

# Moving to device

# model = model.to(device)
# criterion = criterion.to(device)

for epoch in range(num_epochs):
    for i, (x,y) in enumerate(train_loader):

        # Reshape images from (m,1,28,28) to (m,784)
        x = x.reshape(-1,28*28).to(device) # to gpu if available 
        y = y.to(device) # if available 

        # Forward
        y_pred = model(x)
        loss = criterion(y_pred,y)
        #Backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Printing loss 
        if i % 100 ==0 :
            print(f"epoch {epoch+1} / {num_epochs}: step:{i} loss = {loss.item():.6f} ")


# Test => No grad compute


# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')
