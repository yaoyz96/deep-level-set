from __future__ import division
import numpy as np
import cv2
from scipy import misc, ndimage
import torch
from torch.nn import functional as F
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import axes

def Heaviside(v, epsilon=-1/2):
    pi = 3.141593
    v = 0.5 * (1 + 2/pi * torch.atan(v/epsilon))
    return v


def Dirac(x, sigma=0.2, dt_max=30):
    '''This is an adapted version of Dirac function'''
    x = x / dt_max
    f = (1.0 / 2.0) * (1 + torch.cos(np.pi * x / sigma))
    f[(x >= sigma) | (x <= -sigma)] = 0
    return f


def del2(x):
    """
        torch version del2
        x: (N, C, H, W)
        Pay attention to the signal!
    """
    assert x.dim() == 4
    laplacian = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    laplacian = torch.FloatTensor(laplacian).unsqueeze(0).unsqueeze(0).cuda()

    x = F.conv2d(x, laplacian, padding=0)
    x = torch.nn.ReplicationPad2d(1)(x)
    return x


def gradient(x, split=True):
    """
        returns (gy, gx) following the rules of np.gradient!
        torch version gradient for 2D
        x: (N, C, H, W)
    """
    [nrow, ncol] = x.shape[-2:]

    gy = x.clone()
    gy[..., 1:nrow - 1, :] = (x[..., 2:nrow, :] - x[..., 0:nrow - 2, :]) / 2
    gy[..., 0, :] = x[..., 1, :] - x[..., 0, :]
    gy[..., nrow - 1, :] = x[..., nrow - 1, :] - x[..., nrow - 2, :]

    gx = x.clone()
    gx[..., 1:ncol - 1] = (x[..., 2:ncol] - x[..., 0:ncol - 2]) / 2
    gx[..., 0] = x[..., 1] - x[..., 0]
    gx[..., ncol - 1] = x[..., ncol - 1] - x[..., ncol - 2]

    if not split:
        return torch.cat((gy, gx), dim=1)
    return gy, gx


def gradient_sobel(map, split=True):
    """
    returns (gy, gx) following the rules of np.gradient!
    :param map: (N, C=1, H, W)
    :return:
    """

    return gradient(map, split)

    # sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    # sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    #
    # sobel_x = torch.FloatTensor(sobel_x).unsqueeze(0).unsqueeze(0).cuda() / 8
    # sobel_y = torch.FloatTensor(sobel_y).unsqueeze(0).unsqueeze(0).cuda() / 8
    # sobel = torch.cat((sobel_y, sobel_x), dim=0)
    #
    # g = F.conv2d(map, sobel, padding=0)    # (N, C=2, H, W)
    # g = torch.nn.ReplicationPad2d(1)(g)
    #
    # if split:
    #     gy = g[:, 0, :, :].unsqueeze(1)
    #     gx = g[:, 1, :, :].unsqueeze(1)
    #     return gy, gx
    # return g


def div(nx, ny):
    [_, nxx] = gradient_sobel(nx, split=True)
    [nyy, _] = gradient_sobel(ny, split=True)
    return nxx + nyy


def distReg_p2(phi):
    """
        compute the distance regularization term with the double-well potential p2 in equation (16)
    """
    [phi_y, phi_x] = gradient_sobel(phi, split=True)
    s = torch.sqrt(torch.pow(phi_x, 2) + torch.pow(phi_y, 2) + 1e-10)
    a = ((s >= 0) & (s <= 1)).float()
    b = (s > 1).float()
    # compute first order derivative of the double-well potential p2 in equation (16)
    ps = a * torch.sin(2 * np.pi * s) / (2 * np.pi) + b * (s - 1)
    # compute d_p(s)=p'(s)/s in equation (10). As s-->0, we have d_p(s)-->1 according to equation (18)
    neq0 = lambda x: ((x < -1e-10) | (x > 1e-10)).float()
    eq0 = lambda x: ((x >= -1e-10) & (x <= 1e-10)).float()
    dps = (neq0(ps) * ps + eq0(ps)) / (neq0(s) * s + eq0(s))
    return div(dps * phi_x - phi_x, dps * phi_y - phi_y) + del2(phi)


def NeumannBoundCond(f):
    """
        Make a function satisfy Neumann boundary condition
    """
    N, K, H, W = f.shape

    g = f  # f.clone()
    g = torch.reshape(g, (N * K, H, W))
    [_, nrow, ncol] = g.shape

    g[..., [0, nrow - 1], [0, ncol - 1]] = g[..., [2, nrow - 3], [2, ncol - 3]]
    g[..., [0, nrow - 1], 1: ncol - 1] = g[..., [2, nrow - 3], 1: ncol - 1]
    g[..., 1: nrow - 1, [0, ncol - 1]] = g[..., 1: nrow - 1, [2, ncol - 3]]

    g = torch.reshape(g, (N, K, H, W))
    return g


def levelset_evolution(phi, vf, g=None, T=5, timestep=5, dirac_ratio=0.3, dt_max=30, _test=False, _normalize=False):
    if _normalize:
        vf = F.normalize(vf, p=2, dim=1)
    vy = vf[:, 0, :, :].unsqueeze(1)
    vx = vf[:, 1, :, :].unsqueeze(1)

    for k in range(T):
        phi = NeumannBoundCond(phi)
        [phi_y, phi_x] = gradient_sobel(phi, split=True)
        s = torch.sqrt(torch.pow(phi_x, 2) + torch.pow(phi_y, 2) + 1e-10)
        Nx = phi_x / s
        Ny = phi_y / s
        curvature = div(Nx, Ny)
        diracPhi = Dirac(phi, dirac_ratio)
        motion_term = vx * phi_x + vy * phi_y

        if g is None:
            phi = phi + timestep * diracPhi * (motion_term + curvature.detach())
            phi = phi + 0.2 * distReg_p2(phi.detach())
        else:
            phi = phi + timestep * diracPhi * (motion_term + g * curvature.detach())
            phi = phi + 0.2 * distReg_p2(phi.detach())

        # phi_vis = phi[0][0].cpu().numpy()
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # im = ax.imshow(phi_vis, cmap=plt.cm.bwr_r)
        # plt.colorbar(im)
        # plt.savefig('/data/zyy/code/DELSE/checkpoints/img/' + str(i) + '_' + str(k) + '.jpg')
        #plt.show()
    return phi
