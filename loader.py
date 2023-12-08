import torch
import torch.nn as nn


#?  ### COMPLETE MODEL ####
# torch.save(model,PATH)
# model = torch.load(PATH)
# model.eval()


#? ##### SAVE DICT #### => Only params 
# torch.save(model.state_dict(),PATH) # Save params
# model = Model(args)
# model.load_state_dict(torch.load(PATH))
# model.eval() # Evaluation mode

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
    

### LAZY OPTION
#save
model = FFN(100,50,10)
torch.save(model,'save/model.pth')
#load
model = torch.load('save/model.pth')
model.eval()

### SAVE STATE
torch.save(model.state_dict(),'save/model_state.pth')

loaded_model = FFN(100,50,10)
loaded_model.load_state_dict(torch.load('save/model_state.pth'))
loaded_model.eval()
print(loaded_model)


###? SAVE CHECKPOINT DURING TRAINING
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
#* optimizer is a dict so we can save it 

checkpoint = {
    "epoch" : 90, #current epoch
    "model_state" : model.state_dict(),
    "optimizer_state" : optimizer.state_dict()
}

torch.save(checkpoint,'save/checkpoint.pth')

##* Loading the checkpoint 
loaded_checkpoint = torch.load('save/checkpoint.pth')

epoch  = loaded_checkpoint["epoch"]

loaded_model = FFN(100,50,10)
loaded_model.load_state_dict(loaded_checkpoint["model_state"])
loaded_model.eval()
print(loaded_model)

# LR will be loaded
optimizer = torch.optim.SGD(model.parameters(),lr=0)
optimizer.load_state_dict(checkpoint['optimizer_state'])
print(optimizer)


###? LOADING WITH GPU

# device = torch.device("cuda")
# model.to(device)
# torch.save(model.state_dict(),PATH)

# model = Model()
# model.load_state_dict(..)
# model.to(device)