
import pathlib
from collections import OrderedDict
import numpy as np
import torch
import copy
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

        self._relu = torch.nn.LeakyReLU(inplace=True)
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
        b, _, h, w = scores.shape
        uv = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), dim=-1)
        uv = torch.stack([uv.reshape(h*w, 2)]*b, dim=0)
        scores = scores.reshape(b, h*w)
        desc = desc.permute((0, 2, 3, 1)).reshape(b, h*w, -1)
        mask = scores.argsort(dim=1)
        k = self.config['max_keypoints']
        mask = mask[:, :k]
        k = torch.stack([torch.arange(b)]*k, dim=-1).flatten()
        raise NotImplementedError
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

    def forward(self, x0, x1):
        uv0, scores0, desc0 = x0
        uv1, scores1, desc1 = x1

        # Keypoint normalization.
        kpts0 = self.normalize_keypoints(kpts0, data['image0'].shape)
        kpts1 = self.normalize_keypoints(kpts1, data['image1'].shape)

        # Keypoint MLP encoder.
        desc0 = desc0 + self.kenc(kpts0, scores0)
        desc1 = desc1 + self.kenc(kpts1, scores0)

        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)

        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim'] ** .5

        # Run the optimal transport.
        scores = self.log_optimal_transport(scores)
        matches = self.filter_matches(scores)
        return matches

    def filter_matches(self, scores):
        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = self.arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = self.arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))
        return {
            'matches0': indices0,  # use -1 for invalid match
            'matches1': indices1,  # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
        }

    def log_optimal_transport(self, scores):
        """
        Perform Sinkhorn Normalization in Log-space for stability
        about this: http://www.stat.columbia.edu/~gonzalo/pubs/SinkhornOT.pdf
        """
        b, m, n = scores.shape
        one = scores.new_tensor(1)
        ms, ns = (m * one).to(scores), (n * one).to(scores)
        bins0 = self.bin_score.expand(b, m, 1)
        bins1 = self.bin_score.expand(b, 1, n)
        alpha = self.bin_score.expand(b, 1, 1)

        z = torch.cat([torch.cat([scores, bins0], -1),
                               torch.cat([bins1, alpha], -1)], 1)
        """
        slice of z[i] is a (m+1) * (n+1) matrix:
        [scores_i00 ... scores_i0n,  bin_score]
        [ ...                   ...        ...]
        [scores_im0 ... scores_imn,  bin_score]
        [bin_score  ... bin_score,   bin_score]
        """
        norm = - torch.log(ms + ns)
        log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
        log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
        log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)
        """
        log_nu[i] is a (n+1) vector:
        [norm ... norm, log(ms) + norm]
        log_mu[i] is a (n+1) vector:
        [norm ... norm, log(ns) + norm]
        """
        u = torch.zeros_like(log_mu)
        v = torch.zeros_like(log_nu)
        for _ in range(self.config['sinkhorn_iterations']):
            u = log_mu - torch.logsumexp(z + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(z + u.unsqueeze(2), dim=1)
        z = z + u.unsqueeze(2) + v.unsqueeze(1)
        return z - norm  # multiply probabilities by M+N


class AttentionalGNN(torch.nn.Module):
    def __init__(self, feature_dim, layer_types):
        super().__init__()
        self.layer_types = layer_types
        self.layers = torch.nn.ModuleList(
            [AttentionalPropagation(feature_dim, 4) for _ in range(len(layer_types))])

    def forward(self, desc0, desc1):
        for layer, ltype in zip(self.layers, self.layer_types):
            if ltype in ('cross', 'c'):
                src0, src1 = desc1, desc0
            elif ltype in ('self', 's'):
                src0, src1 = desc0, desc1
            delta0 = layer(desc0, src0)
            delta1 = layer(desc1, src1)
            desc0 = (desc0 + delta0)
            desc1 = (desc1 + delta1)
        return desc0, desc1


class AttentionalPropagation(torch.nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = mlp_factory([feature_dim * 2, feature_dim * 2, feature_dim])
        torch.nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class MultiHeadedAttention(torch.nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = torch.nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = torch.nn.ModuleList([copy.deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class KeypointEncoder(torch.nn.Module):
    """
    Joint encoding of visual appearance and location using MLPs
    """
    def __init__(self, feature_dim, layers):
        super().__init__()
        channels = [3, ] + layers + [feature_dim]
        self.encoder = mlp_factory(channels)
        torch.nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


def mlp_factory(channels: 'list[int]', do_bn=True):
    """
    Multi-layer perceptron as a series of 1d convolutions
    """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(torch.nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(torch.nn.BatchNorm1d(channels[i]))
            layers.append(torch.nn.LeakyReLU())
    return torch.nn.Sequential(*layers)

# ===========================================================
# ===========================================================
# ===========================================================
# ===========================================================


class R_NN(torch.nn.Module):  # TODO: there must be 1 image per batch
    def __init__(self, config):
        """
        Each submodule uses config keys for initialization.
        Config is always available via "config" attribute, but it
        should be used for reading only.
        """
        super().__init__()
        self.config = config
        self.point_proposal = R_Point_Proposal(**config)
        self.keypoint_encoder = R_Keypoint_Encoder(**config)
        self.attentional_gnn = R_Attentional_GNN(**config)
        self.optimal_transport = R_Sinkhorn_Optimal_Transport(**config)

    def forward(self, image0, image1):
        keypoints0, descriptors0 = self.point_proposal(image0)
        keypoints1, descriptors1 = self.point_proposal(image1)
        query0 = self.keypoint_encoder(keypoints0, descriptors0)
        query1 = self.keypoint_encoder(keypoints1, descriptors1)
        query0, query1 = self.attentional_gnn(query0, query1)
        matches0, matches1 = self.optimal_transport(query0, query1)
        return {
            'image0': image0,
            'image1': image1,
            'keypoints0': keypoints0,
            'keypoints1': keypoints1,
            'matches0': matches0,
            'matches1': matches1,
            }


class R_Point_Proposal(torch.nn.Module):
    def __init__(self, point_proposal_descriptor_dim: int = 256,
                 keypoint_score_threshold: float = 0.5,
                 image_border_size: int = 5, nms_radius: int = 1, **ignored_kwargs):
        """
        :param point_proposal_descriptor_dim: descriptor vector length
        :param keypoint_score_threshold: from 0 to 1.0, adjusts sensivity
        """
        super().__init__()
        in_, c1, c2, c3, c4, c5, out = 1, 64, 64, 128, 128, 256, point_proposal_descriptor_dim
        conv_kw = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        pc_kw = {'kernel_size': 1, 'stride': 1, 'padding': 0}
        self.keypoint_threshold = keypoint_score_threshold
        self.image_border_size = image_border_size
        self.nms_pool = lambda x: torch.nn.functional.max_pool2d(x,
                        kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

        self.relu = torch.nn.LeakyReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1a = torch.nn.Conv2d(in_, c1, **conv_kw)
        self.conv1b = torch.nn.Conv2d(c1, c1, **conv_kw)
        self.conv2a = torch.nn.Conv2d(c1, c2, **conv_kw)
        self.conv2b = torch.nn.Conv2d(c2, c2, **conv_kw)
        self.conv3a = torch.nn.Conv2d(c2, c3, **conv_kw)
        self.conv3b = torch.nn.Conv2d(c3, c3, **conv_kw)
        self.conv4a = torch.nn.Conv2d(c3, c4, **conv_kw)
        self.conv4b = torch.nn.Conv2d(c4, c4, **conv_kw)
        self.convPa = torch.nn.Conv2d(c4, c5, **conv_kw)
        self.convPb = torch.nn.Conv2d(c5, 65, **pc_kw)
        self.convDa = torch.nn.Conv2d(c4, c5, **conv_kw)
        self.convDb = torch.nn.Conv2d(c5, out, **pc_kw)

    def forward(self, x):
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        # score branch
        scores = self.relu(self.convPa(x))
        scores = self.convPb(scores)
        scores = torch.nn.functional.sigmoid(scores, dim=1)[:, :-1] # this activation was softmax,
        # but i replaced it with sigmoid, because my images have small rank
        b, _, h, w = scores.shapes
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        scores = self.non_maximum_supression(scores)

        # descriptor branch
        descriptors = self.relu(self.convDa(x))
        descriptors = self.convDb(descriptors)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)
        kpts = []
        desc = []
        for scores_i, descriptors_i in zip(scores, descriptors):
            mask_i = scores_i > self.keypoint_threshold
            coords_i = torch.nonzero(mask_i)
            keypoints_i = torch.cat([coords_i, scores_i[mask_i]])  # each keypoint is (height, width, score)
            del mask_i, scores_i
            keypoints_i = self.remove_border(keypoints_i, shape=(h*8, w*8))
            keypoints_i = self.top_k_keypoints(keypoints_i)
            descriptors_i = self.sample_descriptors(keypoints_i, descriptors_i)
            kpts.append(keypoints_i)
            desc.append(descriptors_i)
        return kpts, desc  # list[tensor]

    def non_maximum_supression(self, scores):
        zeros = torch.zeros_like(scores)
        max_mask = scores == self.nms_pool(scores)
        for _ in range(2):
            supp_mask = self.nms_pool(max_mask.float()) > 0
            supp_scores = torch.where(supp_mask, zeros, scores)
            new_max_mask = supp_scores == self.nms_pool(supp_scores)
            max_mask = max_mask | (new_max_mask & (~supp_mask))
        return torch.where(max_mask, scores, zeros)

    def remove_border(self, keypoints, shape):
        b = self.image_border_size
        mask = (keypoints[..., 0] > b) & (keypoints[..., 0] < shape[0] - b) & \
               (keypoints[..., 1] > b) & (keypoints[..., 1] < shape[1] - b)
        return keypoints[mask]

    @staticmethod
    def sample_descriptors(descriptors, keypoints):
        """
        :param descriptors: torch.tensor, dim = (descriptor_dim, h/8, w/8)
        :param keypoints: torch.tensor, dim = (n, 3), last dim is [u, v, score]
        :return: torch.tensor, dim = (n, descriptor_dim)
        """
        s = 8
        c, h, w = descriptors.shape
        coord = keypoints[..., :-1].unsqueeze(0)  # batch dimension is reqired for grid_sample
        descriptors = descriptors.unsqueeze(0)
        max_coord = torch.tensor((h*s - 1, w*s - 1), dtype=torch.float).view(1, 1, 1, 2)
        coord = 2 * coord / max_coord - 1  # scale to (-1, 1)
        descriptors = torch.nn.functional.grid_sample(descriptors, coord, mode='bilinear', align_corners=True)
        descriptors = torch.nn.functional.normalize(descriptors.reshape(c, -1), p=2, dim=1)
        return descriptors


class R_Keypoint_Encoder(torch.nn.Module):
    def __init__(self, args, **ignored_kwargs):
        pass


class R_Attentional_GNN(torch.nn.Module):
    def __init__(self, args, **ignored_kwargs):
        self.graph_convolution = graph_convolution_factory(*args)

    def multiheaded_attention(self, *args):
        pass

    def forward(self, x):
        self.multiheaded_attention(x)
        self.graph_convolution(x)  # multilayer_perceptron


class R_Sinkhorn_Optimal_Transport(torch.nn.Module):
    def __init__(self, args, **ignored_kwargs):
        pass

    def forward(self, x):
        c = self.prepare_normalization_consts(x)
        x = self.sinkhorn_normalization(x, c)
        self.filter_results(x)


