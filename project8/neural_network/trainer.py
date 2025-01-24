import matplotlib.pyplot as plt

from project8.neural_network import utils
from project8.neural_network.models import NN
from copy import deepcopy
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
        self.error_history: list[list[float]] = [[], [], []]
        self.best_state = deepcopy(self.model.state_dict())
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.timer = utils.Timer()
        # other settings not in params
        self.clip_grads = True
        self.clip_value = 1.0
        self.terminate_on_stagnation = True
        self.stagnation_epochs = 1500

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
        if self.clip_grads:
            torch.nn.utils.clip_grad_value_(self.model.parameters(),
                                            self.clip_value)
        if loss.item() < self.best_loss:
            self.best_epoch = len(self.loss_history)
            self.best_loss = loss.item()
            self.best_state = deepcopy(self.model.state_dict())
        self.loss_history.append(loss.item())
        self.optimizer.step()

    def train(self) -> None:
        self.timer.start()
        for epoch in range(1, self.params['n_epochs'] + 1):
            if self.terminate_on_stagnation and (
                    epoch - self.best_epoch) > self.stagnation_epochs:
                if self.verbose:
                    print(f'Terminating due to stagnation at epoch {epoch}')
                break
            self.step()
            if (epoch % 100 == 0) and self.verbose:
                print(f'Epoch: {epoch}, '
                      f'Loss: {self.loss_history[-1]:.6f}, '
                      f'Time: {utils.format_time(self.timer.elapsed())}')
        self.model.load_state_dict(self.best_state)
        if self.verbose:
            print(f'Best loss: {self.best_loss:.6f}'
                  f' at epoch {self.best_epoch}')
        self.timer.stop()

    def train_plot(self, exact, title, fig_id):
        coord_space = utils.ParameterSpace.from_rand_data(
            self.params['loss_fn_data'][0])
        grid = coord_space.fgrid(30)
        grid.requires_grad = False
        real = exact(grid)
        self.error_history[0].append(torch.sqrt(torch.mean((self.model(grid) - real) ** 2)).item())
        self.error_history[1].append(0)
        self.error_history[2].append(0)
        self.timer.start()
        for epoch in range(1, self.params['n_epochs'] + 1):
            if self.timer.elapsed() > 15:
                break
            self.step()
            rmse = torch.sqrt(torch.mean((self.model(grid) - real) ** 2))
            if rmse.item() < self.error_history[0][-1]:
                self.error_history[0].append(rmse.item())
                self.error_history[1].append(self.timer.elapsed())
                self.error_history[2].append(epoch)
            if (epoch % 100 == 0) and self.verbose:
                print(f'Epoch: {epoch}, '
                      f'Loss: {self.loss_history[-1]:.6f}, '
                      f'Time: {utils.format_time(self.timer.elapsed())}')
        self.timer.stop()
        self.model.load_state_dict(self.best_state)
        if self.verbose:
            print(f'Best loss: {self.best_loss:.6f}'
                  f' at epoch {self.best_epoch}')

        print(len(self.error_history[0]))
        plt.figure(fig_id)
        plt.plot(self.error_history[0], self.error_history[1])
        # plt.scatter(self.error_history[0], self.error_history[1], c='r', marker='x')
        plt.ylabel('Time [s]')
        plt.xlabel('RMSE')
        plt.xscale('log')
        plt.gca().invert_xaxis()
        plt.title(title)
        filename = title.replace(' ', '_').replace(':', '').replace('$', '').replace('\\', '') + '.png'
        plt.savefig(utils.plot_folder + '/' + filename)

