import neural_networks
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = neural_networks.models.NN.DRM(2, 10, 4)
# model = neural_networks.models.NN.simple_linear(1, 1, 32, 5, torch.nn.ReLU())
print(model)
model.to(device)
model.initialize_weights(torch.nn.init.xavier_uniform_, torch.nn.init.zeros_, weight_init_kwargs={'gain': 0.1})

def drm_forcing_2d_poisson(input):
    # eq 12 in paper: f=1
    return input[:, 0] * 0 + 1

def drm_loss_2d_poisson(input: torch.Tensor, output: torch.Tensor, grid_shape, beta=500): # model input and output
    nabla_u = torch.autograd.grad(output, input, grad_outputs=torch.ones_like(output), create_graph=True)[0]
    forcing = drm_forcing_2d_poisson(input)
    # eq 13 in paper:
    loss1 = 0.5 * (nabla_u[:, 0] ** 2 + nabla_u[:, 1] ** 2) - forcing*output #  first term (integrate over domain)
    loss2 = output**2 # second term (integrate over boundary)
    # assuming lexicographically ordered grid
    loss1 = loss1.reshape(grid_shape)
    loss2 = loss2.reshape(grid_shape)
    # domain loss:
    loss1 = torch.sum(loss1[1:-1, 1:-1])
    # boundary loss:
    loss2 = torch.sum(loss2[0, :]) + torch.sum(loss2[-1, :]) + torch.sum(loss2[:, 0]) + torch.sum(loss2[:, -1])
    return loss1 + beta * loss2

resolution = 128
input = torch.meshgrid(torch.linspace(-1, 1, resolution), torch.linspace(-1, 1, resolution), indexing='ij')
input = torch.stack(input, dim=-1).reshape(-1, 2).to(device)
print(input.shape)
n_epochs = 5000
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.MSELoss()
for epoch in range(n_epochs):
    input.requires_grad = True
    optimizer.zero_grad()
    output = model(input)
    loss = drm_loss_2d_poisson(input, output, (resolution, resolution))
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch} loss: {loss.item()}')


