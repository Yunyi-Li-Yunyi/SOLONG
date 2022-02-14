import torch
a=torch.tensor([1,2,3])
b=torch.unsqueeze(a,0)
print(len(b))
print(len(a))