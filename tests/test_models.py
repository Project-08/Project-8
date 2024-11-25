from project8 import neural_network
import torch
import matplotlib.pyplot as plt

import pytest


def train(model) -> float:
    device = 'cpu'
    model.to(device)
    model.initialize_weights(torch.nn.init.xavier_uniform_, torch.nn.init.zeros_, weight_init_kwargs={'gain': 0.1})
    x = torch.linspace(0, 2 * 3.14159, 128).to(device).unsqueeze(1)
    target = torch.sin(2 * x)
    n_epochs = 5000
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
        loss.backward()
        optimizer.step()
    return best


def test_drm() -> None:
    model = neural_network.models.NN.drm(1, 5, 3)
    loss = train(model)
    assert loss < 0.5


def test_pinn() -> None:
    model = neural_network.models.NN.rectangular_fnn(1, 1, 32, 5, torch.nn.ReLU())
    loss = train(model)
    assert loss < 0.5


def test_diff_drm() -> None:
    model = neural_network.models.diff_NN.drm(1, 5, 3)
    loss = train(model)
    assert loss < 0.5


def test_diff_pinn() -> None:
    model = neural_network.models.diff_NN.rectangular_fnn(1, 1, 32, 5, torch.nn.ReLU())
    loss = train(model)
    assert loss < 0.5
