import numpy as np
import torch
from skimage.measure import compare_ssim, compare_psnr
from functools import partial



class Bandwise(object): 
    def __init__(self, index_fn):
        self.index_fn = index_fn

    def __call__(self, X, Y):
        C = X.shape[-1]
        bwindex = []
        for ch in range(C):
            x = X[...,ch]
            y = Y[...,ch]
            index = self.index_fn(x, y)
            bwindex.append(index)
        return bwindex


cal_bwssim = Bandwise(partial(compare_ssim, data_range=65536))
cal_bwpsnr = Bandwise(partial(compare_psnr, data_range=65536))


def cal_mrae(X, Y, eps=1e-6):
    return np.sum(np.abs(X - Y) / (Y+eps) ) / np.size(Y)


def cal_sam(X, Y, eps=1e-8):
    tmp = (np.sum(X*Y, axis=0) + eps) / (np.sqrt(np.sum(X**2, axis=0)) + eps) / (np.sqrt(np.sum(Y**2, axis=0)) + eps)    
    return np.mean(np.real(np.arccos(tmp)))


def cal_sid(h1, h2):
    
    H, W, C = h1.shape
    res = 0
    h1 = h1.reshape(-1,C)
    h2 = h2.reshape(-1,C)
    for i in range(H*W):
        res += SID(h1[i,:], h2[i,:])
    return res / H / W


def SID(s1, s2, eps=1e-6):
    """
    Computes the spectral information divergence between two vectors.

    Parameters:
        s1: `numpy array`
            The first vector.

        s2: `numpy array`
            The second vector.

    Returns: `float`
            Spectral information divergence between s1 and s2.

    Reference
        C.-I. Chang, "An Information-Theoretic Approach to SpectralVariability,
        Similarity, and Discrimination for Hyperspectral Image"
        IEEE TRANSACTIONS ON INFORMATION THEORY, VOL. 46, NO. 5, AUGUST 2000.

    """
    p = (s1 / (np.sum(s1) + eps)) + eps
    q = (s2 / (np.sum(s2) + eps)) + eps
    return np.sum(p * np.log10(p / q) + q * np.log10(q / p))


def find_mse(imageA, imageB):
	# 'Mean Squared Error'
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1] * imageA.shape[2])
    return err
 

def find_psnr(imageA, imageB):
	# 'Mean Squared Error'
    mse = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    mse /= float(imageA.shape[0] * imageA.shape[1] * imageA.shape[2])
    PIXEL_MAX = 65536
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse + 1e-3))
    return psnr


def find_appsa(gt,rc):
    
    nom = np.sum(gt * rc, axis=2)
    denom = np.linalg.norm(gt, axis=2) * np.linalg.norm(rc, axis=2)
    
    cos = np.where((nom/(denom + 1e-3)) > 1, 1, (nom/(denom + 1e-3)))
    appsa = np.arccos(cos)
        
    return np.sum(appsa)/(gt.shape[1] * gt.shape[0])


def find_sid(gt, rc):
    N = gt.shape[2]
    err = np.zeros(N)
    for i in range(N):
        err[i] = abs(np.sum(rc[:,:,i] * np.log10((rc[:,:,i] + 1e-3)/(gt[:,:,i] + 1e-3))) +
                     np.sum(gt[:,:,i] * np.log10((gt[:,:,i] + 1e-3)/(rc[:,:,i] + 1e-3))))
    return err / (gt.shape[1] * gt.shape[0])

def find_mrae(gt, rc):
    diff = gt - rc
    abs_diff = np.abs(diff)
    relative_abs_diff = np.divide(abs_diff,gt+1) # added epsilon to avoid division by zero.
    mrae = np.mean(relative_abs_diff)
    return mrae

def MSIQA(X, Y):
    X = quantify_img(X)
    Y = quantify_img(Y)
    # psnr = np.mean(cal_bwpsnr(X, Y))
    # ssim = np.mean(cal_bwssim(X, Y))
    psnr = find_psnr(Y, X)
    ssim = compare_ssim(Y, X)
    # sam = cal_sam(X, Y)
    # mrae = cal_mrae(X, Y)
    # sid = cal_sid(X, Y)
    mrae = find_mrae(Y,X)
    sid = np.mean(find_sid(Y,X))
    appsa = find_appsa(Y, X)
    mse = find_mse(Y, X)

    return {'MRAE': mrae, 'SID': sid}
    # return {'PSNR': psnr, 'SSIM': ssim, 'MRAE': mrae, 'SID': sid, 'APPSA': appsa, 'MSE': mse}


def quantify_img(img):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)
    img = np.array(img*65535)
    return img
