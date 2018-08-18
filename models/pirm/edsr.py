if __name__ == '__main__':
    import common
else:
    from models.pirm import common

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


class EDSR(nn.Module):
    def __init__(self, n_resblocks, in_channels, n_feats, scale, res_scale=0.1, conv=common.default_conv,
    se_reduction=None, bn_feats=None, groups=1, tail_feats=None):
        super(EDSR, self).__init__()

        kernel_size = 3
        act = nn.ReLU(True)
        if tail_feats is None: tail_feats=n_feats

        head, _ = os.path.split(__file__)
        stats = np.load(os.path.join(head, 'pirm_stats.npz'))
        data_mean = stats['data_mean']
        data_std = tuple([1] * len(data_mean))

        self.sub_mean = common.MeanShift(data_mean, data_std, norm=True)
        
        # define head module
        m_head = [conv(in_channels, n_feats, kernel_size)]
        self.shortcut = None
        if n_feats != tail_feats:
            self.shortcut = conv(n_feats, tail_feats, 1)
        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale,
                se_reduction=se_reduction, bn_feats=bn_feats,groups=groups
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, tail_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, tail_feats, act=False),
            nn.Conv2d(
                tail_feats, in_channels, kernel_size,
                padding=(kernel_size//2)
            )
        ]

        self.add_mean = common.MeanShift(data_mean, data_std, norm=False)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.sub_mean(x)
        # data_mean = F.adaptive_avg_pool2d(x, 1)
        # x = x - data_mean
        x = self.head(x)

        res = self.body(x)
        if self.shortcut is not None:
            res += self.shortcut(x)
        else:
            res += x
        x = self.tail(res)
        x = self.add_mean(x)
        # x = x + data_mean
        x = self.relu(x)
        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

if __name__ == '__main__':
    v1 = torch.cuda.FloatTensor(16, 14, 48, 48).fill_(0)
    net = EDSR(n_resblocks=32, in_channels=14, n_feats=256, scale=2).cuda()
    out = net(v1)
    out.mean().backward()
    import ipdb; ipdb.set_trace()
