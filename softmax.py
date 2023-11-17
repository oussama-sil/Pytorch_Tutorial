import torch 
import numpy as np 

x = torch.from_numpy(np.array([[1,2,3],[2,2,2]],dtype=np.float32))


print(torch.nn.functional.softmax(x))
print(torch.nn.functional.softmax(x,dim=0))
print(torch.nn.functional.softmax(x,dim=1))


#! Not apply softmax in last layer
#! Y has class labels 
#! Y_pred has raw score (not softmax)
loss = torch.nn.CrossEntropyLoss()

Y = torch.tensor([0,2,1])
Y_pred = torch.tensor([[10,10,10],[1,1,2],[1,1,1]],dtype=torch.float64)# size m*nb_classes

print(loss(Y_pred,Y))


# Prediction

_,pred = torch.max(Y_pred,-1) #-1 or 0 to specify the dimension (-1 for last dim)
print(pred)


class NN(torch.nn.Module):
    """Some Information about MyModule"""
    def __init__(self,input_size,hidden_size,num_classes):
        super(NN, self).__init__()
        self.lin1 = torch.nn.Linear(input_size,hidden_size,dtype=torch.float64)
        self.relu = torch.nn.ReLU()
        self.lin2 = torch.nn.Linear(hidden_size,num_classes,dtype=torch.float64)

    def forward(self, x):
        out = self.lin1(x)
        print(out)
        out = self.relu(out)
        print(out)
        out = self.lin2(x)

        #! No softmax 

        return x

model   = NN(input_size=3,hidden_size=3 ,num_classes=3)
loss = torch.nn.CrossEntropyLoss()


X = torch.tensor([[10,10,10],[1,1,2],[1,1,1]],dtype=torch.float64)# size m*nb_classes

print(model(X))

from torchsummary import summary
summary(model, (3,),device="cpu")