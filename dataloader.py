


from typing import Any
import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
import numpy as np
import math


class CustomDataset(Dataset):
    
    def __init__(self,transform=None):
        xy = np.loadtxt('./wine.csv',delimiter=',',dtype=np.float32,skiprows=1)
        self.x = xy[:,1:] #=>size (m,n)
        self.y = xy[:,[0]] # => size (m,1)
        self.m = xy.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        """
            Load one item, if large dataset use loading with openning file..
            How to load one element from the dataset 
        """
        sample = self.x[index],self.y[index]
        if self.transform:
            sample =self.transform(sample)
        return sample 

    def __len__(self):
        return self.m

#* Example of transform method (apply on each element after being loaded)
class ToTensor:
    def __call__(self, sample):
        x,y = sample
        return torch.from_numpy(x),torch.from_numpy(y)


if __name__ == '__main__':

    dataset = CustomDataset(transform=ToTensor())
    first = dataset[0]
    x,y = first
    print(x,y)

    #? DataLoader
    #* Speed up data retrieving using multiple workers 
    #* retrieve data with minibatches shuffled 

    #! case error , use __name__='__main__'
    dataloader = DataLoader(dataset=dataset,batch_size=4,shuffle=True,num_workers=1)#* num_workers : loading faster with multiple sub procesors


    m = dataset.__len__()
    num_epochs = 10
    n_iterations = math.ceil(m/4) # 4 batch size
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print( device)
    print(m,n_iterations)
    # Loop for epochs 
    for epoch in range(100):
        for i, (x,y) in enumerate(dataloader): # Loop over batchs 
            (x,y)= x.to(device), y.to(device) 
            if (i+1) % 5 ==0 :
                print(f'epoch {epoch+1}, step{i+1}, inputs {x.shape}')
        #Loop over batches : iteration
