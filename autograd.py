import torch


print("Compute Gradiants")


x = torch.ones(3,requires_grad=True) #! W'll calculate gradiant later for this variable =====> Create computational graph 
print(x)  

print("Functions")
y = x + 1 
print(f'y = x+1 :{y} ')

z = y*y*2
z = z.mean()
print(z)

print("Backward dz_dx")
z.backward() #* No need for argument if z is a scalar 
print(x.grad)

print(y)
v= torch.ones_like(y)
y.backward(v)  #! y has one than one value ==> Have to give it gradiant argument 
print(x.grad)


print("Disable requiring grad")
 #* x.requires_grad_(False)
print(x)
print(x+2)
print(x.detach())

with torch.no_grad():
    print(x+2)  #* don't track computation

print("\n-------------Training inside epoch--------------")

#! When computing grad ==> grad are accumulated in the grad objcect for variable 

#! Without emptying 
weights = torch.ones(4,requires_grad=True)
for epoch in range(2):
    model = (weights*3).sum()
    model.backward()
    print(f"Weights grad after {epoch} epochs = {weights.grad}")
#! Emptying the grad 
for epoch in range(2):
    weights.grad.zero_() # set the grads to zero
    model = (weights*3).sum()
    model.backward()
    print(f"Weights grad after {epoch} epochs = {weights.grad}")




print("\n-------------Cuda--------------")
if torch.cuda.is_available():
    gpu = torch.device("cuda")
    x = torch.ones(5,device=gpu)
    y = x+2
    y = y.mean()
    print(x)
    print(y)
