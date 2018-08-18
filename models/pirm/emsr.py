if __name__ == '__main__':
    import common
else:
    from models.pirm import common

import torch
import torch.nn as nn
import numpy as np
import os


class MemoryBlock(nn.Module):
    def __init__(self, n_resblocks, n_memblocks, n_feats, kernel_size, 
    res_scale=1, conv=common.default_conv, se_reduction=None):
        super(MemoryBlock, self).__init__()
        self.memory_unit = nn.ModuleList(
            [common.ResBlock(conv, n_feats, kernel_size, res_scale=res_scale, se_reduction=se_reduction) for _ in range(n_resblocks)]
        )
        
        self.gate_unit = conv((n_resblocks+n_memblocks) * n_feats, n_feats, 1)
        self.act = nn.ReLU(True)

    def forward(self, x, ys):
        xs = []
        for layer in self.memory_unit:
            x = layer(x)
            xs.append(x)

        gate_out = self.act(self.gate_unit(torch.cat(xs+ys, 1)))
            
        ys.append(gate_out)
        return gate_out
        

class EMSR(nn.Module):
    def __init__(self, n_resblocks, n_memblocks, in_channels, n_feats, scale, 
    res_scale=1, conv=common.default_conv, se_reduction=None):
        super(EMSR, self).__init__()

        kernel_size = 3
        act = nn.ReLU(True)

        head, _ = os.path.split(__file__)
        stats = np.load(os.path.join(head, 'pirm_stats.npz'))
        data_mean = stats['data_mean']
        data_std = tuple([1] * len(data_mean))

        self.sub_mean = common.MeanShift(data_mean, data_std, norm=True)
        
        # define head module
        m_head = [conv(in_channels, n_feats, kernel_size)]
        self.head = nn.Sequential(*m_head)

        # define body module
        self.body = nn.ModuleList(
            [MemoryBlock(n_resblocks, i+1, n_feats, kernel_size, 
            res_scale=res_scale, se_reduction=se_reduction) for i in range(n_memblocks)]
        )

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            nn.Conv2d(
                n_feats, in_channels, kernel_size,
                padding=(kernel_size//2)
            )
        ]

        self.tail = nn.Sequential(*m_tail)
        self.add_mean = common.MeanShift(data_mean, data_std, norm=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = x
        ys = [x]
        for memory_block in self.body:
            x = memory_block(x, ys)
        x = x + res

        x = self.tail(x)
        x = self.add_mean(x)
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
