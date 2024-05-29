import os 

import numpy as np
from gym.spaces import Box, Dict
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils import try_import_torch

torch, nn = try_import_torch()

_WORLD_MAP_NAME = "world-map"
_WORLD_IDX_MAP_NAME = "world-idx_map"
_MASK_NAME = "action_mask"

def get_flat_obs_size(obs_space):
    if isinstance(obs_space, Box):
        return np.prod(obs_space.shape)
    elif not isinstance(obs_space, Dict):
        raise TypeError
    
    def rec_size(obs_dict_shape, n=0):
        for subspace in obs_dict_shape.spaces.values():
            if isinstance(subspace, Box):
                n = n + np.prod(subspace.shape)
            elif isinstance(subspace, Dict):
                n = rec_size(subspace, n=n)
            else:
                raise TypeError
        return n

    return rec_size(obs_space)

def apply_logit_mask(logits, mask):
    """Mask values of 1 are valid actions."
    " Add huge negative values to logits with 0 mask values."""
    logit_mask = torch.ones_like(logits) * -10000000
    logit_mask = logit_mask * (1 - mask)

    return logits + logit_mask

def get_conv_output_size(input_shape, kernel, stride, num_layers, out_channels, padding=0):
    in_size = input_shape
    for _ in range(num_layers):
        in_size = (in_size - kernel + padding) // stride + 1
    
    return in_size * in_size * out_channels


class TorchConvLSTM(RecurrentNetwork):
    custom_name = "torch_conv_lstm"

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        input_emb_vocab = self.model_config["custom_options"]["input_emb_vocab"]
        emb_dim = self.model_config["custom_options"]["idx_emb_dim"]
        num_conv = self.model_config["custom_options"]["num_conv"]
        num_fc = self.model_config["custom_options"]["num_fc"]
        fc_dim = self.model_config["custom_options"]["fc_dim"]
        cell_size = self.model_config["custom_options"]["lstm_cell_size"]
        generic_name = self.model_config["custom_options"].get("generic_name", None)

        self.cell_size = cell_size

        if hasattr(obs_space, "original_space"):
            obs_space = obs_space.original_space

        if not isinstance(obs_space, Dict):
            if isinstance(obs_space, Box):
                raise TypeError(
                    "({}) Observation space should be a gym Dict."
                    " Is a Box of shape {}".format(name, obs_space.shape)
                )
            raise TypeError(
                "({}) Observation space should be a gym Dict."
                " Is {} instead.".format(name, type(obs_space))
            )

        # Define input layers
        self._input_keys = []
        non_conv_input_keys = []
        input_dict = {}
        conv_shape_r = None
        conv_shape_c = None
        conv_map_channels = None
        conv_idx_channels = None
        found_world_map = False
        found_world_idx = False

        for k, v in obs_space.spaces.items():
            shape = (None,) + v.shape
            # no need to specifically declare input instance in pytorch
            input_dict[k] = shape 
            self._input_keys.append(k)
            if k == _MASK_NAME:
                pass
            elif k == _WORLD_MAP_NAME:
                conv_shape_r, conv_shape_c, conv_map_channels = (
                    v.shape[1],
                    v.shape[2],
                    v.shape[0]
                )
                found_world_map = True
            elif k == _WORLD_IDX_MAP_NAME:
                conv_idx_channels = v.shape[0] * emb_dim 
                found_world_idx = True
            else:
                non_conv_input_keys.append(k)
        
        # no need to concatenate the layers in torch

        if found_world_map:
            assert found_world_idx
            use_conv = True
            conv_shape = (
                conv_shape_r,
                conv_shape_c,
                conv_map_channels + conv_idx_channels
            )
        else:
            assert not found_world_idx
            use_conv = False
            conv_shape = None
            conv_input_map = None
            conv_input_idx = None

        logits, values, state_h_p, state_c_p, state_h_v, state_c_v = (
            None,
            None,
            None,
            None,
            None,
            None,
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=conv_shape[2],
                      out_channels=16,
                      kernel_size=3, 
                      stride=2),
            nn.ReLU(), 
        )

        for i in range(num_conv - 1):
            _in_ch = 16 if i == 0 else 32
            self.conv.append(
                nn.Conv2d(in_channels=_in_ch,
                          out_channels=32,
                          kernel_size=3,
                          stride=2)
            )
            self.conv.append(nn.ReLU())

        self.conv.append(nn.Flatten())

        assert get_conv_output_size(conv_shape[2], 3, 2, num_conv, out_channels=32) > 0

        self.fc = nn.Sequential(
            nn.Linear(get_conv_output_size(conv_shape[2], 3, 2, num_conv, out_channels=32),
                      fc_dim),
            nn.ReLU(),
        )

        for _ in range(num_fc - 1):
            self.fc.append(nn.Linear(fc_dim, fc_dim))
            self.fc.append(nn.ReLU())

        