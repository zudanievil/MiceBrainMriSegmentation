"""
Unet implementation in pytorch
paper from the authors of the repository: https://doi.org/10.1016/j.compbiomed.2019.05.002
original code and weights: https://github.com/mateuszbuda/brain-segmentation-pytorch
"""

import pathlib
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import skimage.transform


class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super().__init__()
        self.config = {  # the idea is to pickle config with weights and restore from single .pth file
            'in_channels': in_channels,
            'out_channels': out_channels,
            'init_features': init_features
            }
        pool_kw = dict(kernel_size=2, stride=2)
        ch = init_features
        self._encoder1 = self._block(in_channels, ch, name='enc1')
        self._encoder2 = self._block(ch, ch * 2, name='enc2')
        self._encoder3 = self._block(ch * 2, ch * 4, name='enc3')
        self._encoder4 = self._block(ch * 4, ch * 8, name='enc4')
        self._pool = nn.MaxPool2d(**pool_kw)
        self._bottleneck = self._block(ch * 8, ch * 16, name='bottleneck')
        self._upconv4 = nn.ConvTranspose2d(ch * 16, ch * 8, **pool_kw)
        self._upconv3 = nn.ConvTranspose2d(ch * 8, ch * 4, **pool_kw)
        self._upconv2 = nn.ConvTranspose2d(ch * 4, ch * 2, **pool_kw)
        self._upconv1 = nn.ConvTranspose2d(ch * 2, ch, **pool_kw)
        self._decoder4 = self._block(ch * 16, ch * 8, name="dec4")
        self._decoder3 = self._block(ch * 8, ch * 4, name="dec3")
        self._decoder2 = self._block(ch * 4, ch * 2, name="dec2")
        self._decoder1 = self._block(ch * 2, ch, name="dec1")
        self._conv = nn.Conv2d(in_channels=ch, out_channels=out_channels+1, kernel_size=1)

    def forward(self, x):
        enc1 = self._encoder1(x)
        enc2 = self._encoder2(self._pool(enc1))
        enc3 = self._encoder3(self._pool(enc2))
        enc4 = self._encoder4(self._pool(enc3))
        bottleneck = self._bottleneck(self._pool(enc4))
        dec4 = self._upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self._decoder4(dec4)
        dec3 = self._upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self._decoder3(dec3)
        dec2 = self._upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self._decoder2(dec2)
        dec1 = self._upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self._decoder1(dec1)
        return torch.softmax(self._conv(dec1), dim=1)[:, 0:-1]  # last channel is zero label

    @staticmethod
    def _block(in_channels, features, name):
        block = OrderedDict()
        conv_kw = {'kernel_size': 3, 'padding': 1, 'bias': False}

        block[name+'conv1'] = nn.Conv2d(in_channels=in_channels, out_channels=features, **conv_kw)
        block[name+'norm1'] = nn.BatchNorm2d(num_features=features)
        block[name+'relu1'] = nn.ReLU(inplace=True)

        block[name+'conv2'] = nn.Conv2d(in_channels=features, out_channels=features, **conv_kw)
        block[name + 'norm2'] = nn.BatchNorm2d(num_features=features)
        block[name + 'relu2'] = nn.ReLU(inplace=True)
        return nn.Sequential(block)


class FolderAsUnetDataset(torch.utils.data.Dataset):
    def __init__(self, folder: pathlib.Path, transform: callable = None) -> None:
        self.paths = []
        for path in folder.iterdir():
            if path.suffix == '.npz':
                self.paths.append(path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        x = dict(np.load(self.paths[idx]))
        if self.transform:
            x = self.transform(x)
        inp = torch.from_numpy(x['inp']).permute([2, 0, 1])
        gnd = torch.from_numpy(x['gnd']).permute([2, 0, 1])
        return inp, gnd


class RandomTransformWrapper:
    def __init__(self, max_rot_z: 'rads' = 1, max_scale: 'times-1' = 0.2,
                 max_shear: 'side length fraction' = 0.1,
                 max_translation: 'side length fraction' = 0.2,
                 max_pan_xy: 'not more than 0.001 recommended' = 0.0005):
        self.m = np.array([max_rot_z, max_scale, max_scale, max_shear, max_shear,
                           max_translation, max_translation, max_pan_xy, max_pan_xy], dtype=np.float)

    def __call__(self, x: 'dict[np.ndarray]') -> 'dict[np.ndarray]':
        theta, scx, scy, shx, shy, \
            trx, try_, panx, pany = np.random.uniform(-1, 1, 9)*self.m
        scx += 1; scy += 1
        for k in x:
            shape = x[k].shape
            try_sc = try_ * shape[0]
            trx_sc = trx * shape[1]
            cy = shape[0] // 2
            cx = shape[1] // 2
            h = self._get_matrix(theta, cx, cy, scx, scy, shx, shy, trx_sc, try_sc, panx, pany)
            x[k] = skimage.transform.warp(x[k], h, preserve_range=True)
        return x

    @staticmethod
    def _get_matrix(theta, cx, cy, scx, scy, shx, shy, trx, try_, panx, pany):
        a = np.cos(-theta)  # uv coordinates have swapped direction
        b = np.sin(-theta)
        h = np.array([
            [a,  b + shx, (scx - a) * cx - b * cy + trx],
            [-b, a + shy, (scy - a) * cy + b * cx + try_],
            [panx, pany, 1]])
        return h
