import torch

ext = torch.load('tensor_data.pth')
a = ext[:260, :, :]
print(len(a))