import numpy as np
import skimage.transform
import torch
from ..utils import linalg_utils
from ..utils import ml_utils


def pad_to_3x3_transform_matrix(x: torch.tensor, dim=-1):
    shape = list(x.shape)
    to_pad = 9-shape[dim]
    shape[dim] = 1
    ones = torch.ones(*shape, dtype=x.dtype)
    if to_pad > 1:
        shape[dim] = to_pad - 1
        zeros = torch.zeros(*shape, dtype=x.dtype)
        pad_tensor = torch.cat((zeros, ones), dim=dim)
    else:
        pad_tensor = ones
    x = torch.cat((x, pad_tensor), dim=dim)
    shape[dim] = 3
    shape.insert(dim+1, 3)
    return x.reshape(*shape)


class MyConvolutionalEncoder(torch.nn.Module):
    """
    simple encoder of Conv2d and AvgPool2d(2,2) LeakyReLU() layers stacked
    :var input_channels: the module input channels,
    :var channel_list: the channel output of the convolutional layers
    channel_list[-1] is the module output channels
    :var conv_kwargs are applied to all convolutions. default: {'kernel_size': 3, 'dilation': 1, 'padding': 1}
    """
    def __init__(self, input_channels: int,
                 channel_list: 'Iterable[int]',
                 conv_kwargs: dict = None):
        super().__init__()
        channel_list = [input_channels, *channel_list]
        conv_kwargs = conv_kwargs or {'kernel_size': 3, 'dilation': 1, 'padding': 1}
        layers = []
        act = torch.nn.LeakyReLU()
        for i in range(len(channel_list)-1):
            inp = channel_list[i]
            out = channel_list[i+1]
            layers.append(torch.nn.Conv2d(inp, out, **conv_kwargs))
            layers.append(torch.nn.AvgPool2d(2, 2))
            layers.append(act)
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class GRULoop(torch.nn.Module):
    """
    GRU cells 'stacked' together.
    Each cell uses the output of previous one as a hidden state.
    On forward pass, all cells use same tensor (:var x) as an input.
    """
    def __init__(self, input_size: int, hidden_size: int, units: int):
        """
        :param input_size: same for all GRUCell modules
        :param hidden_size: same for all GRUCell modules
        :param units: number of GRUCell modules
        """
        super().__init__()
        gru_list = []
        for _ in range(units):
            gru_list.append(torch.nn.GRUCell(input_size, hidden_size))
        self.gru_list = torch.nn.ModuleList(gru_list)
        self.register_parameter('h0', torch.nn.Parameter(torch.zeros(hidden_size)))

    def forward(self, x):
        """
        :param x: explicit input to each GRUCell in stack
        :return: hidden state h_n
        """
        h = self.h0.expand(x.shape[0], -1)
        for gru in self.gru_list:
            h = gru(x, h)
        return h


class DifferentiableImageTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: (torch.Tensor, torch.Tensor)):
        """
        x: tuple of (images, transforms).
        image dimensions are assumed bchw.
        transform dimensions are assumed to be (b, 3, 3) or (b, c, 3, 3)
        if each channel is transformed independently
        transform is applied as-is (first row is for y axis, second is for x axis)
        """
        image, transform = x
        b, c, h, w = image.shape
        image = image.reshape(b*c, h, w).unsqueeze(1)
        if len(transform.shape) == 3:
            transform = transform.unsqueeze(1)
        transform = transform.expand(b, c, 2, 3).reshape(b*c, 2, 3)
        grid = torch.nn.functional.affine_grid(transform, (b*c, 1, h, w))
        image = torch.nn.functional.grid_sample(image, grid, mode='bilinear')
        image = image.squeeze(1).reshape(b, c, h, w)
        return image


class LocalizationNetwork(torch.nn.Module):
    """learns coefficients for an affine transform from an image"""
    def __init__(self, input_channels: int = 1,
                 encoder_channel_numbers: 'Iterable[int]' = (10, 20, 30, 20, 10, 4),
                 adaptive_average_pool_size: (int, int) = (4, 4),
                 gru_cells_hidden_state_size: int = 24,
                 number_of_gru_cells: int = 100,
                 return_affine_transform=False):
        super().__init__()
        config = locals()
        config.pop('self')
        config.pop('__class__')
        self.config = config  # cls(**config) recreates an instance
        gru_input_size = (adaptive_average_pool_size[0]*adaptive_average_pool_size[1])*encoder_channel_numbers[-1]
        self.encoder = MyConvolutionalEncoder(input_channels, encoder_channel_numbers)
        self.pool = torch.nn.AdaptiveAvgPool2d(adaptive_average_pool_size)
        self.gru_loop = GRULoop(gru_input_size, gru_cells_hidden_state_size, number_of_gru_cells)
        self.fc = torch.nn.Linear(gru_cells_hidden_state_size, 6)
        self.tform = DifferentiableImageTransform()

    def forward(self, x):
        image = x
        x = self.encoder(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.gru_loop(x)
        x = self.fc(x)
        transform = x.reshape(x.shape[0], 2, 3)
        image = self.tform((image, transform))
        return transform if self.config['return_affine_transform'] else image


class LocNetDatasetTransform:
    def __init__(self, return_affine_transform=False):
        self.h_gen = linalg_utils.RandomHomographyGenerator(max_pan_xy=0)
        self.return_affine_transform = return_affine_transform

    def __call__(self, inp: np.ndarray, gnd: np.ndarray):
        """
        return 2 images:
        """
        h = self.h_gen(*inp.shape[0:2])
        inp = skimage.transform.warp(inp, h)
        gnd_transform = np.linalg.inv(h) @ gnd
        gnd_image = skimage.transform.warp(inp, gnd)

        inp = ml_utils.image_bhwc_to_bcwh(inp)
        gnd_image = ml_utils.image_bhwc_to_bcwh(gnd_image)
        if self.return_affine_transform:
            return inp, gnd_transform
        else:
            return inp, gnd_image
