
# dont know how pytest works yet

import neural_networks
import torch
import matplotlib.pyplot as plt

device = 'cuda:0'
model = neural_networks.models.NN.DRM(1, 5, 3)
# model = neural_networks.models.NN.simple_linear(1, 1, 32, 5, torch.nn.ReLU())
print(model)
model.to(device)
model.initialize_weights(torch.nn.init.xavier_uniform_, torch.nn.init.zeros_, weight_init_kwargs={'gain': 0.1})
x = torch.linspace(0, 2*3.14159, 128).to(device).unsqueeze(1)
target = torch.sin(2*x)
n_epochs = 5000
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.MSELoss()
for epoch in range(n_epochs):
    x.requires_grad = True
    optimizer.zero_grad()
    output = model(x)
    print(output.shape)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch} loss: {loss.item()}')
output = model(x)
plt.plot(x.cpu().detach().numpy(), target.cpu().detach().numpy())
plt.plot(x.cpu().detach().numpy(), output.cpu().detach().numpy())
plt.show()