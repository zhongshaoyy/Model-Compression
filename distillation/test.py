import torch 

x=torch.randn(2,3,3)
print (x)

s=x.mean(dim=(1, 2), keepdim=False)
t= x.mean(dim=(0), keepdim=False)
print (s) 
print (t)
