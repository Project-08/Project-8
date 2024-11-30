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
        self.__contents = torch.load(os.path.join(model_folder, self.filename))

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
    model_file['model_str'] = str(model)


def str_remove_line_before_dots_if_dots_in_line(s: str) -> str:
    if s[-1] != '\n':  # make sure last line ends with a newline
        s += '\n'
    out = ''
    line = ''
    before_dots = True
    for char in s:
        line += char
        if char == '\n':
            if ':' not in line:
                out += line
            else:
                out += char
            line = ''
            before_dots = True
        if not before_dots:
            out += char
        if char == ':':
            before_dots = False
    return out


def str_format_modulelist_args_as_list(s: str) -> str:
    start = s.find('ModuleList') + len('ModuleList') + 1
    before = s[:start] + '['
    out = ''
    bracket_depth = 0
    done = False
    for char in s[start:]:
        if done:
            out += char
            continue
        if char == '(':
            bracket_depth += 1
            out += char
        elif char == ')':
            bracket_depth -= 1
            if bracket_depth == 0:
                out += char + ','
            elif bracket_depth == -1:
                out += ']' + char
                done = True
            else:
                out += char
        else:
            out += char
    if 'ModuleList' in out:
        out = str_format_modulelist_args_as_list(out)
    return before + out


def format_model_str(s: str) -> str:
    s = str_remove_line_before_dots_if_dots_in_line(s)
    s = str_format_modulelist_args_as_list(s)
    return s


def load_model(filename: str) -> NN:
    model_file = file_synced_dict(filename)
    if 'format_model_str' not in model_file.keys():
        model_file['format_model_str'] = format_model_str(
            model_file['model_str'])
    model: NN = eval(model_file['format_model_str'])
    model.load_state_dict(model_file['model_state'])
    return model


def main() -> None:
    pass


if __name__ == '__main__':
    main()
