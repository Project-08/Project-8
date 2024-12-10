from project8.neural_network import utils
from project8.neural_network.models import NN
import torch


class trainer:
    def __init__(self, model: NN, params: utils.Params,
                 verbose: bool = False
                 ) -> None:
        self.model = model
        self.params: utils.Params = params
        self.verbose = verbose
        self.optimizer = params['optimizer'](model.parameters(),
                                             lr=params['lr'])
        self.data_loaders: list[utils.DataLoader] = []
        for i in range(len(self.params['loss_fn_data'])):
            self.data_loaders.append(
                utils.DataLoader(
                    params['loss_fn_data'][i],
                    params['loss_fn_batch_sizes'][i],
                    True,
                    device=params['device']
                )
            )
        self.loss_fns = params['loss_fns']
        self.loss_weights = params['loss_weights']
        self.device = params['device']
        self.loss_history: list[float] = []
        self.timer = utils.Timer()

    def eval_loss_fn(self, i: int) -> torch.Tensor:
        self.model(self.data_loaders[i]())
        return self.loss_weights[i] * self.loss_fns[i](self.model)

    def loss(self) -> torch.Tensor:
        out = torch.tensor(0.0, device=self.device)
        for i in range(len(self.loss_fns)):
            out += self.eval_loss_fn(i)
        return out

    def step(self) -> None:
        self.optimizer.zero_grad()
        loss = self.loss()
        loss.backward()  # type: ignore
        self.loss_history.append(loss.item())
        self.optimizer.step()

    def train(self) -> None:
        self.timer.start()
        for epoch in range(1, self.params['n_epochs']+1):
            self.step()
            if (epoch % 100 == 0) and self.verbose:
                print(f'Epoch: {epoch}, '
                      f'Loss: {self.loss_history[-1]:.4f}, '
                      f'Time: {utils.format_time(self.timer.elapsed())}')
        self.timer.stop()
