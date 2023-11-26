import torch

# Define the tensor
tensor = torch.tensor([
    [[2, 4, 3], [2, 1, 2]],
    [[1, 2, 3], [4, 1, 1]],
    [[2, 1, 3], [3, 2, 4]],
    [[1, 4, 2], [2, 4, 3]]
], dtype=torch.float32)

print (tensor.shape)


mu_batch = tensor.mean(dim=[0,2], keepdim=True)
sigma2_batch = tensor.var(dim=[0,2], keepdim=True, unbiased=False)

mu_layer = tensor.mean(dim=[1, 2], keepdim=True)
sigma2_layer = tensor.var(dim=[1, 2], keepdim=True, unbiased=False)

mu_instance = tensor.mean(dim=2, keepdim=True)
sigma2_instance = tensor.var(dim=2, keepdim=True, unbiased=False)

mu_batch, sigma2_batch, mu_layer, sigma2_layer, mu_instance, sigma2_instance


