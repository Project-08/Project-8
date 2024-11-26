import neural_network
from project8.neural_network import utils as util
import torch


def source(input):
    return torch.sin(4 * input[:, 0] * input[:, 1]).unsqueeze(1)


def drm_loss_2d_poisson_domain(model: neural_network.models.diff_NN):
    # sum/mean of 0.5 * (u_x^2 + u_y^2) - f*u in the domain
    # eq 13 first term
    grad = model.gradient()
    f = source(model.input)
    return torch.mean(0.5 * torch.sum(grad.pow(2), 1).unsqueeze(1) - f * model.output)


def drm_loss_2d_poisson_bndry(model: neural_network.models.diff_NN):
    # sum/mean of u^2 on the boundary
    # eq 13 second term
    return model.output.pow(2).mean()


device = util.get_device()
# init model
act_fn = neural_network.modules.Sin(torch.pi)
model = neural_network.models.diff_NN.drm(2, 1, 20, 4, act_fn=act_fn)
print(model)
model.to(device)
model.double()
model.initialize_weights(torch.nn.init.xavier_uniform_, torch.nn.init.zeros_, weight_init_kwargs={'gain': 0.5})

# training params
n_epochs = 5000
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
coord_space = util.ParameterSpace([[-1, 2], [-1, 1]], device)  # domain defined here
tmr = util.timer()

# plot source function
f_loc = coord_space.fgrid(300)
f = source(f_loc)
x, y = coord_space.regrid(f_loc)
f = coord_space.regrid(f)[0]
util.plot_2d(x, y, f, title='source function', fig_id=1)

tmr.start()
# train
for epoch in range(n_epochs):
    model(coord_space.rand(1000).requires_grad_(True))
    domain_loss = drm_loss_2d_poisson_domain(model)
    model(coord_space.bndry_rand(500).requires_grad_(True))
    bndry_loss = drm_loss_2d_poisson_bndry(model)
    loss = domain_loss + 100 * bndry_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(
            f'Epoch {epoch}, domain loss: {domain_loss.item()}, boundary loss: {bndry_loss.item()}, total: {loss.item()}')
tmr.rr()

# plot output
grid = coord_space.fgrid(200)
output = model(grid)
x, y = coord_space.regrid(grid)
f = coord_space.regrid(output)[0]
util.plot_2d(x, y, f, title='output_drm')
