import neural_networks
import convenience as cv
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def drm_forcing_2d_poisson(input):
    # eq 12 in paper: f=1
    f = input[:, 0] * 0 + 1
    return f.unsqueeze(1)

def sourcefunc_nmfde_A3(input, alpha=40):
    sum = 0
    for i in range(1,10):
        for j in range(1,5):
            sum += torch.exp(-alpha*(input[:, 0]-i)**2 - alpha*(input[:,1]-j)**2)
    return sum

def source3(input):
    f = torch.exp(-20 * (input[:, 0] - 0.5) ** 2 - 20 * (input[:, 1] - 0.5) ** 2) + 0.1
    return f.unsqueeze(1)

def source4(input):
    return torch.sin(4  * input[:, 0] * input[:, 1]).unsqueeze(1)

# to change which is used:
def source(input):
    return source4(input)

def drm_loss_2d_poisson_domain(domain_input, domain_output):
    # sum/mean of 0.5 * (u_x^2 + u_y^2) - f*u in the domain
    # eq 13 first term
    model.zero_grad()
    nabla_u = torch.autograd.grad(domain_output,domain_input,grad_outputs=torch.ones_like(domain_output),retain_graph=True,create_graph=True,only_inputs=True)[0]
    f = source(domain_input)
    return torch.mean(0.5 * torch.sum(nabla_u * nabla_u, 1).unsqueeze(1) - f * domain_output)

def drm_loss_2d_poisson_bndry(bndry_output):
    # sum/mean of u^2 on the boundary
    # eq 13 second term
    return torch.mean(bndry_output * bndry_output)

device = cv.get_device()
# init model
act_fn = neural_networks.modules.Sin(torch.pi)
model = neural_networks.models.NN.drm(2, 20, 4, act_fn=act_fn)
# model = neural_networks.models.NN.simple_linear(2, 1, 32, 9, act_fn=act_fn)
print(model)
model.to(device)
model.double()
model.initialize_weights(torch.nn.init.xavier_uniform_, torch.nn.init.zeros_, weight_init_kwargs={'gain': 0.5})

# training params
n_epochs = 5000
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
coord_space = cv.float_parameter_space([[-1, 2], [-1, 1]], device) # domain defined here

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
for epoch in range(n_epochs):
    domain_input = coord_space.rand(1000).requires_grad_(True)
    domain_output = model(domain_input)
    domain_loss = drm_loss_2d_poisson_domain(domain_input, domain_output)
    bndry_input = coord_space.bndry_rand(500).requires_grad_(True)
    bndry_output = model(bndry_input)
    bndry_loss = drm_loss_2d_poisson_bndry(bndry_output)
    loss = domain_loss + 100 * bndry_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, domain loss: {domain_loss.item()}, boundary loss: {bndry_loss.item()}, total: {loss.item()}')

# plot output
grid = coord_space.fgrid(200)
output = model(grid)
x, y = coord_space.regrid(grid)
f = coord_space.regrid(output)[0]
f = f.detach().to('cpu')
x = x.detach().to('cpu')
y = y.detach().to('cpu')
cv.plot_2d(x, y, f, title='output_drm')
