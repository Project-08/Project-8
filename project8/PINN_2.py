import neural_networks
import convenience as cv
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def source4(input):
    return torch.sin(4  * input[:, 0] * input[:, 1]).unsqueeze(1)

# to change which is used:
def source(input):
    return source4(input)

def pinn_domain_loss_nd_laplacian(model: neural_networks.models.diff_NN):
    f = source(model.input)
    return (model.laplacian() + f).pow(2).mean()

def pinn_bndry_loss(model: neural_networks.models.diff_NN):
    return model.output.pow(2).mean()

device = cv.get_device()
# init model
act_fn = neural_networks.modules.Sin(torch.pi)
model = neural_networks.models.diff_NN.simple_linear(2, 1, 64, 9, act_fn=act_fn)
print(model)
model.to(device)
model.double()
model.initialize_weights(torch.nn.init.xavier_uniform_, torch.nn.init.zeros_, weight_init_kwargs={'gain': 0.5})

# training params
n_epochs = 5000
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
coord_space = cv.float_parameter_space([[-1, 2], [-1, 1]], device) # domain defined here
tmr = cv.timer()

# plot source function
f_loc = coord_space.fgrid(300)
f = source(f_loc)
x, y = coord_space.regrid(f_loc)
f = coord_space.regrid(f)[0]
f = f.detach().to('cpu')
x = x.detach().to('cpu')
y = y.detach().to('cpu')
cv.plot_2d(x, y, f, title='source function', fig_id=1)

# train
tmr.start()
for epoch in range(n_epochs):
    model(coord_space.rand(1000).requires_grad_(True)) # forward pass on domain
    domain_loss = pinn_domain_loss_nd_laplacian(model)
    model(coord_space.bndry_rand(500).requires_grad_(True)) # forward pass on boundary
    bndry_loss = pinn_bndry_loss(model)
    loss = domain_loss + 10* bndry_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, domain loss: {domain_loss.item()}, boundary loss: {bndry_loss.item()}, total: {loss.item()}')
tmr.rr()

# plot output
grid = coord_space.fgrid(200)
output = model(grid)
x, y = coord_space.regrid(grid)
f = coord_space.regrid(output)[0]
f = f.detach().to('cpu')
x = x.detach().to('cpu')
y = y.detach().to('cpu')
cv.plot_2d(x, y, f, title='output_pinn_2')