from project8.neural_network import models
from project8.neural_network import checkpoints
import torch

max_loss = 0.4


def train(model: models.NN) -> tuple[float, models.NN]:
    device = 'cpu'
    model.to(device)
    model.initialize_weights(
        torch.nn.init.xavier_uniform_,
        torch.nn.init.zeros_,
        weight_init_kwargs={'gain': 0.1})
    x = torch.linspace(0, 2 * 3.14159, 128).to(device).unsqueeze(1)
    target = torch.sin(2 * x)
    n_epochs = 10000
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    best = 1e10
    for epoch in range(n_epochs):
        x.requires_grad = True
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        if loss.item() < best:
            best = loss.item()
        if best < max_loss:
            break
        loss.backward()
        optimizer.step()
    return best, model


def test_drm() -> None:
    model = models.NN.drm(1, 1, 5, 3)
    loss = train(model)[0]
    assert loss < max_loss


def test_pinn() -> None:
    model = models.NN.rectangular_fnn(1, 1, 10, 5, torch.nn.ReLU())
    loss = train(model)[0]
    assert loss < max_loss


def test_save_load_pinn() -> None:
    loss, model = train(
        models.NN.rectangular_fnn(1, 1, 32, 5, torch.nn.ReLU()))
    model_str = str(model)
    checkpoints.save_model_state(model, 'test.pth')
    model2 = checkpoints.load_model('test.pth')
    loss, model2 = train(model2)
    assert model_str == str(model2) and loss < max_loss


def test_save_load_drm() -> None:
    loss, model = train(models.NN.drm(1, 1, 5, 3))
    model_str = str(model)
    checkpoints.save_model_state(model, 'test.pth')
    model2 = checkpoints.load_model('test.pth')
    loss, model2 = train(model2)
    assert model_str == str(model2) and loss < max_loss
