
import pathlib
from collections import OrderedDict
import numpy as np
import torch
import skimage.transform

"""
default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    }
"""


class SuperGlueNetwork(torch.nn.Module):
    def __init__(self, ):
        self.config = locals()
        self.config.pop('self')
        super().__init__()
        self._point_proposal = PointProposal(self.config)
        self._descriptor_matching = DescriptorMatching(self.config)

    def forward(self, x1, x2):
        self._point_proposal(x1)
        self._point_proposal(x2)
        matches = self._descriptor_matching(x1, x2)
        return matches


class PointProposal(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        in_, c1, c2, c3, c4, c5, out = 1, 64, 64, 128, 128, 256, config['descriptor_dim']
        conv_kw = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        pc_kw = {'kernel_size': 1, 'stride': 1, 'padding': 0}

        self._relu = torch.nn.ReLU(inplace=True)
        self._pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self._conv1a = torch.nn.Conv2d(in_, c1, **conv_kw)
        self._conv1b = torch.nn.Conv2d(c1, c1, **conv_kw)
        self._conv2a = torch.nn.Conv2d(c1, c2, **conv_kw)
        self._conv2b = torch.nn.Conv2d(c2, c2, **conv_kw)
        self._conv3a = torch.nn.Conv2d(c2, c3, **conv_kw)
        self._conv3b = torch.nn.Conv2d(c3, c3, **conv_kw)
        self._conv4a = torch.nn.Conv2d(c3, c4, **conv_kw)
        self._conv4b = torch.nn.Conv2d(c4, c4, **conv_kw)
        self._convPa = torch.nn.Conv2d(c4, c5, **conv_kw)
        self._convPb = torch.nn.Conv2d(c5, 65, **pc_kw)
        self._convDa = torch.nn.Conv2d(c4, c5, **conv_kw)
        self._convDb = torch.nn.Conv2d(c5, out, **pc_kw)

    def forward(self, x):
        x = self._relu(self._conv1a(x))
        x = self._relu(self._conv1b(x))
        x = self._pool(x)
        x = self._relu(self._conv2a(x))
        x = self._relu(self._conv2b(x))
        x = self._pool(x)
        x = self._relu(self._conv3a(x))
        x = self._relu(self._conv3b(x))
        x = self._pool(x)
        x = self._relu(self._conv4a(x))
        x = self._relu(self._conv4b(x))
        # score branch
        scores = self._relu(self._convPa(x))
        scores = self._convPb(scores)
        scores = torch.nn.functional.sigmoid(scores, dim=1)[:, :-1]
        # this was softmax, but i replaced it with sigmoid, because the images have small rank
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        scores = self.simple_nms(scores)
        scores = self.remove_borders(scores)
        # descriptor branch
        desc = self._relu(self._convDa(x))
        desc = self.convDb(desc)
        desc = torch.nn.functional.normalize(desc, p=2, dim=1)
        uv, scores, desc = self.topk_to_uv_list(scores, desc)
        return uv, scores, desc

    def simple_nms(self, scores):
        radius = self.config['nms_radius']
        if radius < 0:
            return scores
        max_mask = torch.nn.functional.max_pool2d(scores, kernel_size=radius * 2 + 1, stride=1, padding=radius)
        zeros = torch.zeros_like(scores)
        scores = torch.where(max_mask == scores, scores, zeros)
        # the original code contained iterative refining of the result
        # but i concluded it has no significant effect, so i removed it
        return scores

    def remove_borders(self, scores):
        if self.config['border_fraction'] <= 0:
            return scores
        b, c, h, w = scores.shape
        hf = int(h*self.config['border_fraction'])
        wf = int(h*self.config['border_fraction'])
        scores = scores[..., hf:-hf, wf:-wf]
        scores = torch.nn.functional.pad(scores, (wf, wf, hf, hf))
        return scores

    def topk_to_uv_list(self, scores, desc):
        k =
        b, _, h, w = scores.shape
        uv = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), dim=-1)
        uv = torch.stack([uv.reshape(h*w, 2)]*b, dim=0)
        scores = scores.reshape(b, h*w)
        desc = desc.permute((0, 2, 3, 1)).reshape(b, h*w, -1)
        mask=scores.argsort(dim=1)
        k = self.config['max_keypoints']
        mask = mask[:, :k]
        k = torch.stack([torch.arange(b)]*k, dim=-1).flatten()


        return uv, scores, desc


class DescriptorMatching(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        in_ = config['descriptor_dim']
        self.kenc = KeypointEncoder(in_, self.config['keypoint_encoder'])
        self.gnn = AttentionalGNN(in_, self.config['GNN_layers'])
        self.final_proj = torch.nn.Conv1d(in_, in_, kernel_size=1, bias=True)
        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

    def forward(self, x1, x2):
        uv1, scores1, desc1 = x1
        uv1, scores1, desc1 = x2