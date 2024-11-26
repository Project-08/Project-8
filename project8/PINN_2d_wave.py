import neural_network
from project8.neural_network import utils as util
import torch
from copy import deepcopy


def sourcefunc_A4(x, y, alpha=40):
    Lx = 10
    Ly = 5
    f = torch.zeros_like(x)
    a = [
        [0.25*Lx, 0.25*Ly],
        [0.25*Lx, 0.75*Ly],
        [0.75*Lx, 0.75*Ly],
        [0.75*Lx, 0.25*Ly]
    ]
    for i in a:
        f += torch.exp(-alpha * (x - i[0]) ** 2 - alpha * (y - i[1]) ** 2)
    return f

def sourcefunc_wave(x, y, t, alpha=40, omega=4*torch.pi):
    f = sourcefunc_A4(x, y,  alpha)
    return f*torch.sin(omega*t)

def F(coords):
    return sourcefunc_wave(coords[:, 0], coords[:, 1], coords[:, 2]).unsqueeze(1)

def coeffK3(x, y, args=None):
    return 1 + 0.1 * (x + y + x * y)

def K(coords):
    return coeffK3(coords[:, 0], coords[:, 1]).unsqueeze(1)

def pinn_wave_pde(model: neural_network.models.diff_NN):
    f = F(model.input)
    k = K(model.input)
    residual = model.diff(2, 2) - model.divergence(k * model.gradient()) - f
    return residual.pow(2).mean()

def pinn_wave_bc(model: neural_network.models.diff_NN):
    return model.output.pow(2).mean()

def pinn_wave_ic(model: neural_network.models.diff_NN):
    return model.output.pow(2).mean()

device = util.get_device()
# init model
act_fn = neural_network.modules.Sin(torch.pi)
model = neural_network.models.diff_NN.rectangular_fnn(3, 1, 256, 9, act_fn=act_fn)
print(model)
model.to(device)
model.double()
model.initialize_weights(torch.nn.init.xavier_uniform_, torch.nn.init.zeros_, weight_init_kwargs={'gain': 0.5})

# training params
n_epochs = 5000
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
coord_space = util.ParameterSpace([[0, 10], [0, 5], [0, 4]], device) # xyt domain defined here
tmr = util.timer()
domain_data_loader = util.DataLoader(coord_space.rand(100000), 800, device=device, output_requires_grad=True)
bc_select = torch.tensor([[1, 1], [1, 1], [0, 0]], device=device, dtype=torch.float64)
bc_data_loader = util.DataLoader(coord_space.select_bndry_rand(100000, bc_select), 400, device=device, output_requires_grad=True)
ic_select = torch.tensor([[0, 0], [0, 0], [1, 0]], device=device, dtype=torch.float64)
ic_data_loader = util.DataLoader(coord_space.select_bndry_rand(100000, ic_select), 100, device=device, output_requires_grad=True)

# plot source function
# f_loc = coord_space.fgrid(300)
# f = F(f_loc)
# x, y = coord_space.regrid(f_loc)
# f = coord_space.regrid(f)[0]
# util.plot_2d(x, y, f, title='wave source function', fig_id=1)

# train
tmr.start()
best = 1e10
for epoch in range(n_epochs):
    model(domain_data_loader())
    domain_loss = pinn_wave_pde(model)
    model(bc_data_loader())
    bndry_loss = pinn_wave_bc(model)
    model(ic_data_loader())
    ic_loss = pinn_wave_ic(model)
    loss = domain_loss + 25 * bndry_loss + 10 * ic_loss
    if loss < best:
        best = loss.item()
        best_state = deepcopy(model.state_dict())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, pde: {domain_loss.item()}, bc: {bndry_loss.item()}, ic: {ic_loss.item()}, total: {loss.item()}, best: {best}')
tmr.rr()
model.load_state_dict(best_state)
print(f'best loss: {best}')

# plot output
grid = coord_space.fgrid(50)
model.eval()
output = model(grid)
x, y, t = coord_space.regrid(grid)
f = coord_space.regrid(output)[0]

t_id = 0
util.plot_2d(x[t_id,:,:], y[t_id,:,:], f[t_id,:,:], title='output_wave1', fig_id=2)
t_id = 20
util.plot_2d(x[t_id,:,:], y[t_id,:,:], f[t_id,:,:], title='output_wave2', fig_id=3)
t_id = 49
util.plot_2d(x[t_id,:,:], y[t_id,:,:], f[t_id,:,:], title='output_wave3', fig_id=4)
