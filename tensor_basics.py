import torch
import numpy as np


#* Tensor creation 
x1 = torch.empty(3,dtype=torch.int32)
x2 = torch.rand(3,2,dtype=torch.float32)
x3 = torch.zeros(3,2,2)
x4 = torch.ones(3,2,2)
x5 = torch.tensor([2.5,0.1])
print(x1.dtype)
print(x5)

#* Tensor operation
x=torch.rand(2,2)
y=torch.rand(2,2)

z=  torch.add(x,y) # sub div mul 
print(x)
print(y)
y.add_(x) # Modify y
print(y)

#* Slicing operation 
x = torch.rand(5,3)
print(x)
print(x[:,:1])
print(x[0,0]) # print tensor
print(x[0,0].item()) #! case one element



#* Reshaping 
print("--------------Reshaping------------------")

x = torch.rand(4,4)
print(x)
print(x.view(16)) # reshape to one dim 16 = 4*4
print(x.view(-1,4,1).size()) # -1 , first dim, second dim => pytorch will findthe dim for remining dimension 
print(x.view(-1,8).size()) # -1 , first dim, second dim => pytorch will findthe dim for remining dimension 

#* Reshaping 

print("--------------Numpy------------------")
x = torch.ones(2,2)
a = x.numpy() #! Share the same localtion in memory 
print(x)
print(a)
x.add_(1)
print(x)
print(a)
y = torch.from_numpy(np.ones(5)) #! Share the same localtion in memory 
print(y)

print("--------------Cuda------------------")
#! Numpy handle GPU tensors only
if torch.cuda.is_available():
    print(f"Cuda devices {torch.cuda.device_count() }")
    device= torch.device("cuda")
    #* Create on device
    x = torch.ones(5,device=device)

    #* Move to the device 
    x = torch.ones(5)
    y = y.to(device)
    y = y.to("cpu")
