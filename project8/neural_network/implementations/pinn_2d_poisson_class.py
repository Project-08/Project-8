from project8.neural_network import utils, modules, models, autograd_wrapper as aw
import torch
import logging


class PINN2DPoisson:
    def __init__(self, n_inputs=2, n_outputs=1, width=64, n_blocks=9, act_fn=None,
                 weight_init_fn=torch.nn.init.xavier_uniform_, bias_init_fn=torch.nn.init.zeros_,
                 weight_init_kwargs=None, lr=1e-3, n_epochs=5000, loss_weights=(1, 1), device=None):
        """
        Initialize the Poisson PINN model and training parameters.

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
        :param loss_weights: Weights for domain and boundary losses.
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
        self.coord_space = utils.ParameterSpace([[-1, 2], [-1, 1]], self.device)
        self.diff = aw.Differentiator(self.model)

        # Prepare DataLoaders
        self.pde_data = utils.DataLoader(self.coord_space.rand(100000), 1000, True)
        self.bndry_data = utils.DataLoader(self.coord_space.bndry_rand(100000), 500, True)

        # Define problem: loss functions, dataloaders, and weights
        self.problem = [
            [self.domain_loss, self.pde_data, self.loss_weights[0]],
            [self.boundary_loss, self.bndry_data, self.loss_weights[1]]
        ]

    def source(self, input: torch.Tensor) -> torch.Tensor:
        """
        Source term function for the Poisson problem.
        """
        return torch.sin(12 * input[:, 0] * input[:, 1]).unsqueeze(1)

    def domain_loss(self, diff: aw.Differentiator) -> torch.Tensor:
        """
        Compute the domain loss term.
        """
        f = self.source(diff.input())
        laplace = diff.laplacian()
        return (laplace + f).pow(2).mean()

    def boundary_loss(self, diff: aw.Differentiator) -> torch.Tensor:
        """
        Compute the boundary loss term.
        """
        return diff.output().pow(2).mean()

    def train(self) -> float:
        """
        Train the model and return the final loss value.
        """
        logging.info(self.model)
        timer = utils.Timer()
        timer.start()

        # Plot source function
        f_loc = self.coord_space.fgrid(300)
        f = self.source(f_loc)
        x, y = self.coord_space.regrid(f_loc)
        f = self.coord_space.regrid(f)[0]
        utils.plot_2d(x, y, f, title='source_pinn_2d_poisson', fig_id=1)

        # Training loop
        for epoch in range(self.n_epochs):
            loss = torch.tensor(0.0, device=self.device)
            for loss_fn, data_loader, weight in self.problem:
                self.diff.set_input(data_loader())
                loss += weight * loss_fn(self.diff)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Logging
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, total loss: {loss.item()}")

        timer.rr()
        return loss.item()

    def evaluate(self) -> None:
        """
        Plot the model output after training.
        """
        grid = self.coord_space.fgrid(200)
        output = self.model(grid)
        x, y = self.coord_space.regrid(grid)
        f = self.coord_space.regrid(output)[0]
        utils.plot_2d(x, y, f, title='output_pinn_2d_poisson', fig_id=2)


if __name__ == "__main__":
    # Initialize the model with hyperparameters
    model = PINN2DPoisson(n_epochs=5000, lr=1e-3, width=64, n_blocks=9)

    # Train the model
    final_loss = model.train()

    # Evaluate the model
    model.evaluate()

    print(f"Final Loss: {final_loss}")
