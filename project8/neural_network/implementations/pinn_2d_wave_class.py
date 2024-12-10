from project8.neural_network import utils, modules, models, autograd_wrapper as aw
import torch
import logging


class PINN2DWave:
    def __init__(self, n_inputs=3, n_outputs=1, width=64, n_blocks=9, act_fn=None,
                 weight_init_fn=torch.nn.init.xavier_uniform_, bias_init_fn=torch.nn.init.zeros_,
                 weight_init_kwargs=None, lr=1e-3, n_epochs=5000, loss_weights=(1, 1, 1), device=None):
        """
        Initialize the wave PINN model and training parameters.

        :param n_inputs: Number of input dimensions.
        :param n_outputs: Number of output dimensions.
        :param width: Width of the network layers.
        :param n_blocks: Number of blocks in the model.
        :param act_fn: Activation function to use in the model.
        :param weight_init_fn: Weight initialization function.
        :param bias_init_fn: Bias initialization function.
        :param weight_init_kwargs: Additional arguments for weight initialization.
        :param lr: Learning rate for the optimizer.
        :param n_epochs: Number of training epochs.
        :param loss_weights: Weights for PDE, boundary, and initial condition losses.
        :param device: Device to use (CPU/GPU).
        """
        self.device = device or utils.get_device()
        self.model = models.NN.rectangular_fnn(n_inputs, n_outputs, width, n_blocks, act_fn=act_fn or modules.FourierActivation(width))
        self.model.to(self.device)
        self.model.double()
        self.model.initialize_weights(
            weight_init_fn, bias_init_fn, weight_init_kwargs=weight_init_kwargs or {'gain': 0.5}
        )
        self.n_epochs = n_epochs
        self.lr = lr
        self.loss_weights = loss_weights
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.coord_space = utils.ParameterSpace([[0, 10], [0, 5], [0, 4]], self.device) # hard-coded for 2D wave
        self.diff = aw.Differentiator(self.model)

        # Prepare DataLoaders
        self.domain_data_loader = utils.DataLoader(
            self.coord_space.rand(100000), 3000, device=self.device, output_requires_grad=True)
        bc_select = torch.tensor([[1, 1], [1, 1], [0, 0]], device=self.device, dtype=torch.float64)
        self.bc_data_loader = utils.DataLoader(
            self.coord_space.select_bndry_rand(100000, bc_select), 800, device=self.device, output_requires_grad=True)
        ic_select = torch.tensor([[0, 0], [0, 0], [1, 0]], device=self.device, dtype=torch.float64)
        self.ic_data_loader = utils.DataLoader(
            self.coord_space.select_bndry_rand(100000, ic_select), 300, device=self.device, output_requires_grad=True)

        # Define problem: loss functions, dataloaders, and weights
        self.problem = [
            [self.pde_loss, self.domain_data_loader, self.loss_weights[0]],
            [self.boundary_loss, self.bc_data_loader, self.loss_weights[1]],
            [self.initial_condition_loss, self.ic_data_loader, self.loss_weights[2]]
        ]

    def sourcefunc_A4(self, x: torch.Tensor, y: torch.Tensor, alpha: float = 40) -> torch.Tensor:
        Lx = 10
        Ly = 5
        f = torch.zeros_like(x)
        a = [
            [0.25 * Lx, 0.25 * Ly],
            [0.25 * Lx, 0.75 * Ly],
            [0.75 * Lx, 0.75 * Ly],
            [0.75 * Lx, 0.25 * Ly]
        ]
        for i in a:
            f += torch.exp(-alpha * (x - i[0]) ** 2 - alpha * (y - i[1]) ** 2)
        return f

    def sourcefunc_wave(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor,
                        alpha: float = 40, omega: float = 4 * torch.pi) -> torch.Tensor:
        f = self.sourcefunc_A4(x, y, alpha)
        return f * torch.sin(omega * t)

    def F(self, coords: torch.Tensor) -> torch.Tensor:
        return self.sourcefunc_wave(coords[:, 0], coords[:, 1], coords[:, 2]).unsqueeze(1)

    def pde_loss(self, diff: aw.Differentiator) -> torch.Tensor:
        f = self.F(diff.input())
        residual: torch.Tensor = diff(2, 2) - diff.laplacian() - f
        return residual.pow(2).mean()

    def boundary_loss(self, diff: aw.Differentiator) -> torch.Tensor:
        return diff.output().pow(2).mean()

    def initial_condition_loss(self, diff: aw.Differentiator) -> torch.Tensor:
        return diff.output().pow(2).mean()

    def train(self) -> float:
        """
        Train the model and return the final loss value.
        """
        logging.info(self.model)
        for epoch in range(self.n_epochs):
            loss: torch.Tensor = torch.tensor(0.0, device=self.device)
            for loss_fn, data_loader, weight in self.problem:
                self.diff.set_input(data_loader())
                loss += weight * loss_fn(self.diff)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, total loss: {loss.item()}")

        return loss.item()

    def evaluate(self) -> None:
        """
        Plot the model output after training.
        """
        grid = self.coord_space.fgrid(50)
        self.model.eval()
        output = self.model(grid)
        x, y, t = self.coord_space.regrid(grid)
        f = self.coord_space.regrid(output)[0]

        for t_id, title in zip([0, 20, 49], ['output_wave1', 'output_wave2', 'output_wave3']):
            utils.plot_2d(x[t_id, :, :], y[t_id, :, :], f[t_id, :, :], title=title, fig_id=t_id + 2)


if __name__ == "__main__":
    # Initialize the model with hyperparameters
    model = PINN2DWave(n_epochs=5000, lr=1e-3, width=64, n_blocks=9)

    # Train the model
    final_loss = model.train()

    # Evaluate the model
    model.evaluate()

    print(f"Final Loss: {final_loss}")
