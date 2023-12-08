import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time 
from torchsummary import summary
import sys
import torch.nn.functional as F

#? Tensorboard 

from torch.utils.tensorboard import SummaryWriter 

writer = SummaryWriter("runs/mnist2")



#? Start tensorboard 
# pytorch tensorboard --logdir=runs


print("=======> Example of FFN <======")

#? Device , GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

print(f"Device : {device}")
print(torch.cuda.get_device_properties(device))


#? Hyper parameters 
input_size = 28*28 # 28 x 28
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.01



#? MNIST Data 

#* Datasets
train_dataset = torchvision.datasets.MNIST(root='./data',train=True,transform=transforms.ToTensor(),download=True)
test_dataset = torchvision.datasets.MNIST(root='./data',train=False,transform=transforms.ToTensor(),download=True)

#* Loaders => To load data while training
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # Doesn't matter for the evaluation


#? Size of a batch of data
examples = iter(train_loader)
samples, labels = next(examples) # Get one batch from training data 
print( f"Shape of a batch X = {samples.shape} Y = {labels.shape} " ) 


#? Plotting some examples 
for i in range(6):
    plt.subplot(2,3,i+1) #2 rows with 3 columns at index i+1
    plt.imshow(samples[i][0],cmap='gray')
# plt.show()

#? Write to tensorboard
img_grid = torchvision.utils.make_grid(samples)
writer.add_image('MNIST images',img_grid) 

# writer.close() #? Makes sure to write all the events
# sys.exit()


#? Model
print("\n =====> Model <===== \n")

class FFN(nn.Module):
    def __init__(self,input_size, hidden_size,num_classes) :
        super(FFN,self).__init__()
        self.l1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size,num_classes)
    def forward(self,x):
        """
            Input : x of size [m,input_size]
            Output : y_hat of size [m,num_classes]
            Don't apply the softmax
        """
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out) 
        return out
        #! Don't apply softmax caus the cross entropy loss applies the softmax loss

model = FFN(input_size,hidden_size,num_classes)

# To visualize on console 
summary(model,input_size=(batch_size,input_size),device="cpu")


#? Visualize the computational graph
# from torchviz import make_dot
# dummy_input = torch.randn(input_size)
# output = model(dummy_input)
# graph = make_dot(output, params=dict(model.named_parameters()))
# graph.render("FFN_model", format="png", cleanup=True)


#! Important moving the  parameters of the model to the same device as the data
model = model.to(device)

print("\n =====> Training <===== \n")

#? Loss and Optimizer
loss_funct = nn.CrossEntropyLoss() #! Applies the softmax on the output 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#? 
writer.add_graph(model,samples.reshape(-1,28*28).to(device))
# writer.close() #? Makes sure to write all the events
# sys.exit()

#? Training loop
n_total_steps = len(train_loader) # Number of batchs

print(f"NB Epochs = {num_epochs} , NB Batchs = {n_total_steps} ")
start_time = time.time()


running_loss = 0.0
running_correct = 0.0

for epoch in range(num_epochs):
    for i,(images,labels) in enumerate(train_loader):
        # Reshape data and push to the device (GPU)
        images = images.reshape(-1,28*28).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = loss_funct(outputs,labels)

        # Backward
        optimizer.zero_grad() # Set the grads to zero
        loss.backward() # Compute the gradiants for all the parameters 
        optimizer.step()

        #? For tensorboard 
        running_loss += loss.item()
        # Predictions
        _,predictions = torch.max(outputs,1) # Returns value and index of max along the dimension 1 
        running_correct += (predictions==labels).sum().item() # Check if the prediction is correct

        # Printing some informations 
        if (i+1) % 100 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1} / {n_total_steps}, loss = {loss.item():.4f} ')
            writer.add_scalar("Training Loss",running_loss/100,epoch*n_total_steps+i)
            writer.add_scalar("Accuracy",running_correct/100,epoch*n_total_steps+i)
            running_loss = 0.0
            running_correct = 0.0

end_time = time.time()
training_time = end_time - start_time
print(f"End of training, training time: {training_time:.2f} seconds")

#? Testing 
print("\n =====> Testing <===== \n")


#? To diplay precision/recall graph 
#! For one function 
labels = []
preds = [] #! must be probabilities (apply softmax)

with torch.no_grad(): #! Don't compute the gradiant 
    n_correct = 0
    n_samples = 0
    for images,labels1 in test_loader: # Loop over test data
        images = images.reshape(-1,28*28).to(device)
        labels1 = labels1.to(device)
        outputs = model(images)

        class_predictions = [F.softmax(output,dim=0) for output in outputs]

        # Predictions : predicted labels
        _,predictions = torch.max(outputs,1) # Returns value and index of max along the dimension 1 
        n_samples += labels1.shape[0] # number of samples in current batch
        n_correct = (predictions==labels1).sum().item() # Check if the prediction is correct

        #
        preds.append(class_predictions)
        labels.append(predictions) #
    
    print(labels)
    print(preds)

    labels = torch.cat(labels) # One dim tensor
    preds = torch.cat([torch.stack(batch) for batch in preds])

    print(labels.shape)
    print(preds.shape)


    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy on test data = {acc}')

    classes = range(10)
    for i in classes:
        labels_i = labels==i
        preds_i = preds[:,i]
        writer.add_pr_curve(str(i),labels_i,preds_i,global_step=0)
    

writer.close() #? Makes sure to write all the events
# sys.exit()