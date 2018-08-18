from .pirm import EDSR
from .pirm import EMSR


"""PIRM Baseline"""
def edsrx2():
    net = EDSR(n_resblocks=32, in_channels=14, n_feats=256, scale=2)
    net.use_2dconv = True
    net.bandwise = False
    net.use_upsample = True
    return net

def edsrx3():
    net = EDSR(n_resblocks=32, in_channels=14, n_feats=256, scale=3)
    net.use_2dconv = True
    net.bandwise = False
    net.use_upsample = True
    return net

def edsrcax2():
    net = EDSR(n_resblocks=32, in_channels=14, n_feats=256, scale=2, se_reduction=8, res_scale=1)
    net.use_2dconv = True
    net.bandwise = False
    net.use_upsample = True
    return net

def edsrcax3():
    net = EDSR(n_resblocks=32, in_channels=14, n_feats=256, scale=3, se_reduction=8, res_scale=1)
    net.use_2dconv = True
    net.bandwise = False
    net.use_upsample = True
    return net

def emsrx2():
    net = EMSR(n_resblocks=6, n_memblocks=6, in_channels=14, n_feats=256, scale=2, res_scale=1)
    net.use_2dconv = True
    net.bandwise = False
    net.use_upsample = True
    return net

def emsrx3():
    net = EMSR(n_resblocks=6, n_memblocks=6, in_channels=14, n_feats=256, scale=3, res_scale=1)
    net.use_2dconv = True
    net.bandwise = False
    net.use_upsample = True
    return net

def emsrcax2():
    net = EMSR(n_resblocks=6, n_memblocks=6, in_channels=14, n_feats=256, scale=2, se_reduction=8, res_scale=1)
    net.use_2dconv = True
    net.bandwise = False
    net.use_upsample = True
    return net

def emsrcax3():
    net = EMSR(n_resblocks=6, n_memblocks=6, in_channels=14, n_feats=256, scale=3, se_reduction=8, res_scale=1)
    net.use_2dconv = True
    net.bandwise = False
    net.use_upsample = True
    return net
