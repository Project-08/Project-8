import os
import torch
from typing import Any
from torch.nn import *  # noqa
from project8.neural_network.modules import *  # noqa
from project8.neural_network.models import NN

model_folder = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    ), 'models')


class file_synced_dict:
    """
    Class to create a dictionary that is synced with a file.
    Uses torch's save and load function.
    Files are in the models folder.
    """

    def __init__(self, filename: str) -> None:
        if not filename.endswith('.pth'):
            filename += '.pth'
        elif '.' in filename and '.pth' not in filename:
            raise ValueError(
                f'Invalid filename: {filename},'
                f' must not have file extension other than .pth')
        self.filename: str = filename
        self.__contents: dict[str, Any] = {}
        if not os.path.exists(os.path.join(model_folder, filename)):
            self.save()
        else:
            self.load()

    def save(self) -> None:
        torch.save(self.__contents, os.path.join(model_folder, self.filename))

    def load(self) -> None:
        self.__contents = torch.load(os.path.join(model_folder, self.filename),
            map_location=torch.device('cpu'))

    def __getitem__(self, key: Any) -> Any:
        return self.__contents[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self.__contents[key] = value
        self.save()

    def __delitem__(self, key: Any) -> None:
        del self.__contents[key]
        self.save()

    def keys(self) -> list[str]:
        return list(self.__contents.keys())

    def values(self) -> list[Any]:
        return list(self.__contents.values())

    def __str__(self) -> str:
        return str(self.__contents)


def save_model_state(model: NN, filename: str) -> None:
    model_file = file_synced_dict(filename)
    model_file['model_state'] = model.state_dict()
    model_file['model_definition'] = model.definition()


def load_model(filename: str) -> Any:
    model_file = file_synced_dict(filename)
    model = eval(model_file['model_definition'])
    model.load_state_dict(model_file['model_state'])
    return model
