import os

import torch


def _get_or_create_checkpoint_path(input_path) -> str:
    augmented_path = os.path.join("checkpoint", input_path)
    checkpoint_dir: str = os.path.dirname(augmented_path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    return augmented_path

def save(state_dict: dict, save_path: str, train_params: typing.List):
    """Save model

    :param train_params: all the previous and current train parameters
    :param state_dict: the weightings of the model
    :param save_path: where to save the model
    """
    print('==> Save to checkpoint..', save_path)
    augmented_path = _get_or_create_checkpoint_path(save_path)
    data = {'state_dict': state_dict, 'param_dict': str(train_params)}
    torch.save(data, augmented_path)


def load(load_path: str, m):
    print('==> Resuming from checkpoint ', load_path)
    augmented_path = _get_or_create_checkpoint_path(load_path)
    checkpoint = torch.load(augmented_path)
    m.load_state_dict(checkpoint['state_dict'])