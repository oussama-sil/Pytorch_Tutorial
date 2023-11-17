import torch 
import torch.nn as nn

import numpy as np

print("------------------Manually-------------------")
weights = torch.randn(1)
X = np.array([1,2,3,4],dtype=np.float32)
Y = np.array([2,4,6,8],dtype=np.float32)
w = 0
# Model
def forward(x):
    return w*x
#loss MSE
def loss(y,y_hat):
    return np.mean((y-y_hat)**2)
# Gradd
def grad(x,y,y_hat):
    return np.dot(2*x,y_hat-y).mean()
print(f"Predict before training f(5) ={forward(5)}")

lr = 0.01
nb_iters  = 10

for epoch in range(nb_iters):
    # Prediction
    y_hat = forward(X)

    # Loss
    loss_ = loss(Y,y_hat)

    # Grad
    dy_dw  = grad(X,Y,y_hat)

    # Update formulat 
    w -= lr*dy_dw

    if epoch % 2 ==0 :
        print(f"epoch {epoch+1}: w = {w:.3f}  loss = {loss_:.6f} ")


print(f"Predict After training f(5) ={forward(5)}")



print("\n------------------Using pytorch-------------------")
X = torch.from_numpy(np.array([1,2,3,4],dtype=np.float32))
Y = torch.from_numpy(np.array([2,4,6,8],dtype=np.float32))
w = torch.tensor(0.0,dtype=torch.float32,requires_grad=True)
# Model
def forward(x):
    return w*x
#loss MSE
def loss(y,y_hat):
    return ((y-y_hat)**2).mean()

print(f"Predict before training f(5) ={forward(5)}")

lr = 0.01
nb_iters  = 10

for epoch in range(nb_iters):
    # Prediction
    y_hat = forward(X)

    # Loss
    loss_ = loss(Y,y_hat)

    #* Grad : Backward pass
    loss_.backward() # COmput the gradients
    dy_dw  = w.grad

    # Update formula ==> Out of computation graph
    with torch.no_grad():
        w -= lr*w.grad
    
    w.grad.zero_()

    if epoch % 2 ==0 :
        print(f"epoch {epoch+1}: w = {w:.3f}  loss = {loss_:.6f} ")


print(f"Predict After training f(5) ={forward(5)}")



print("\n------------------Using pytorch training pipeline-------------------")
#? Training pipeline in pytorch
#* 1-Design model(input , output size , forward pass)
#* 2- Construct loss and optimizer
#* 3- Training loop
#*      Forward pass => output
#*      Backward pass => gradients
#*      Update weigts

X = torch.from_numpy(np.array([1,2,3,4],dtype=np.float32))
Y = torch.from_numpy(np.array([2,4,6,8],dtype=np.float32))
w = torch.tensor(0.0,dtype=torch.float32,requires_grad=True)

# Model
def forward(x):
    return w*x

# Loss 
loss = nn.MSELoss() #* Callable function

# Optimizer 
optimizer = torch.optim.SGD([w],lr=0.01)




print(f"Predict before training f(5) ={forward(5)}")

nb_iters  = 10

for epoch in range(nb_iters):
    # Prediction
    y_hat = forward(X)

    # Loss
    loss_ = loss(Y,y_hat)

    #* Grad : Backward pass
    loss_.backward() # COmput the gradients
    
    #* Update weights 
    optimizer.step()
    
    optimizer.zero_grad()


    if epoch % 2 ==0 :
        print(f"epoch {epoch+1}: w = {w:.3f}  loss = {loss_:.6f} ")


print(f"Predict After training f(5) ={forward(5)}")





print("\n------------------Full pytorch training pipeline-------------------")
#? Training pipeline in pytorch
#* 1-Design model(input , output size , forward pass)
#* 2- Construct loss and optimizer
#* 3- Training loop
#*      Forward pass => output
#*      Backward pass => gradients
#*      Update weigts

X = torch.from_numpy(np.array([[1],[2],[3],[4]],dtype=np.float32)) #! Modify shape to (nb_row,nb_features)
Y = torch.from_numpy(np.array([[2],[4],[6],[8]],dtype=np.float32))
w = torch.tensor(0.0,dtype=torch.float32,requires_grad=True)

# Model

n_sample,n_features = X.shape

#! Input for one sample 
input_size = n_features
output_size = n_features
model = nn.Linear(input_size,output_size)


#! Custom Model
class LinearRegression(nn.Module):
    def __init__(self, input_size,output_size) -> None:
        super(LinearRegression, self).__init__()

        # define the layers 
        self.lin = nn.Linear(input_size,output_size)

    def forward(self,x):
        return self.lin(x)

model = LinearRegression(input_size,output_size)


# Loss 
loss = nn.MSELoss() #* Callable function

# Optimizer 
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)



X_test = torch.tensor([5],dtype=torch.float32)
print(f"Predict before training f(5) ={model(X_test)}")

nb_iters  = 100

for epoch in range(nb_iters):
    # Prediction
    y_hat = model(X)

    # Loss
    loss_ = loss(Y,y_hat)

    #* Grad : Backward pass
    loss_.backward() # COmput the gradients
    
    #* Update weights 
    optimizer.step()
    
    #* Reset weights 
    optimizer.zero_grad()


    if epoch % 10 ==0 :
        [w,b]=model.parameters() # unpack parameters
        print(f"epoch {epoch+1}: w = {w[0][0].item():.3f}  loss = {loss_:.6f} ")


print(f"Predict After training f(5) ={model(X_test).item()}")


print(model)
print("Model Device:", next(model.parameters()).device)


from torchsummary import summary
summary(model, (input_size,),device="cpu")