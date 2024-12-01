from project8.neural_network import utils, modules, models
import torch
import logging


class DRM2DPoisson:
    def __init__(self, n_inputs=2, n_outputs=1, width=20, n_blocks=4, act_fn=None,
                 weight_init_fn=torch.nn.init.xavier_uniform_, bias_init_fn=torch.nn.init.zeros_,
                 weight_init_kwargs=None, lr=1e-3, n_epochs=5000, loss_weights=(1, 100), device=None):
        """
        Initialize the Poisson model and training parameters.

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
        self.model = models.diff_NN.drm(n_inputs, n_outputs, width, n_blocks, act_fn=act_fn or modules.Sin(torch.pi))
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

    def source(self, input: torch.Tensor) -> torch.Tensor:
        """
        Source term function for the Poisson problem.
        """
        return torch.sin(4 * input[:, 0] * input[:, 1]).unsqueeze(1)

    def domain_loss(self) -> torch.Tensor:
        """
        Compute the domain loss term.
        """
        grad = self.model.gradient()
        f = self.source(self.model.input)
        return torch.mean(
            0.5 * torch.sum(grad.pow(2), 1).unsqueeze(1) - f * self.model.output
        )

    def boundary_loss(self) -> torch.Tensor:
        """
        Compute the boundary loss term.
        """
        return self.model.output.pow(2).mean()

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
        utils.plot_2d(x, y, f, title='source function', fig_id=1)

        # Training loop
        for epoch in range(self.n_epochs):
            # Domain loss
            self.model(self.coord_space.rand(1000).requires_grad_(True))
            domain_loss = self.domain_loss()

            # Boundary loss
            self.model(self.coord_space.bndry_rand(500).requires_grad_(True))
            boundary_loss = self.boundary_loss()

            # Combined loss
            loss = self.loss_weights[0] * domain_loss + self.loss_weights[1] * boundary_loss

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Logging
            if epoch % 100 == 0:
                logging.info(
                    f"Epoch {epoch}, "
                    f"domain loss: {domain_loss.item()}, "
                    f"boundary loss: {boundary_loss.item()}, "
                    f"total: {loss.item()}"
                )

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
        utils.plot_2d(x, y, f, title='output_drm')


if __name__ == "__main__":
    # Initialize the model with hyperparameters
    model = DRM2DPoisson(n_epochs=5000, lr=1e-3, width=20, n_blocks=4)
    
    # Train the model
    final_loss = model.train()
    
    # Evaluate the model
    model.evaluate()
    
    print(f"Final Loss: {final_loss}")
