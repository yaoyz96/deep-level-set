from __future__ import division
import numpy as np
import torch
from torch.nn import functional as F
from layers.lse import *


def bd_loss(phi, sdt, sigma=0.08, dt_max=30, size_average=True):
    dist = torch.mul(torch.pow(phi - sdt, 2), Dirac(sdt, sigma, dt_max) + Dirac(phi.detach(), sigma, dt_max))
    loss = torch.sum(dist)
    if size_average:
        sz = Dirac(sdt, sigma, dt_max).sum()
        loss = loss / sz
    return loss


def class_balanced_bce_loss(outputs, labels, size_average=False, batch_average=True):
    assert(outputs.size() == labels.size())

    num_labels_pos = torch.sum(labels)
    num_labels_neg = torch.sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg

    loss_val = -(torch.mul(labels, torch.log(outputs)) + torch.mul((1.0 - labels), torch.log(1.0 - outputs)))

    loss_pos = torch.sum(torch.mul(labels, loss_val))
    loss_neg = torch.sum(torch.mul(1.0 - labels, loss_val))

    final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg

    if size_average:
        final_loss /= torch.numel(labels)
    elif batch_average:
        final_loss /= labels.size()[0]
    return final_loss


def level_map_loss(output, sdt, alpha=1):
    assert (output.size() == sdt.size())
    mse = lambda x: torch.mean(torch.pow(x, 2))

    sdt_loss = mse(sdt - output)
    return alpha*sdt_loss


def vector_field_loss(vf_pred, vf_gt, sdt=None, _print=False):
    # (n_batch, n_channels, H, W)
    vf_pred = F.normalize(vf_pred, p=2, dim=1)
    vf_gt = F.normalize(vf_gt, p=2, dim=1)
    cos_dist = torch.sum(torch.mul(vf_pred, vf_gt), dim=1)
    angle_error = torch.acos(cos_dist * (1-1e-4))

    angle_loss = torch.mean(torch.pow(angle_error, 2))
    if _print:
        print('[vf_loss] vf_loss: ' + str(angle_loss.item()))

    return angle_loss


def LSE_output_loss(phi_T, gts, sdts, epsilon=-1, dt_max=30):
    pixel_loss = class_balanced_bce_loss(Heaviside(phi_T, epsilon=epsilon), gts, size_average=True)
    # boundary_loss = bd_loss(phi_T, sdts, sigma=2.0/dt_max, dt_max=dt_max)
    loss = 100 * pixel_loss
    return loss


def cal_levelset_loss(H_pi_batch, gts, num_classes):

    def zero():
        return 0
    H_pi_batch = F.softmax(H_pi_batch, dim=1) # (batch,2,h,w)
    H_pi_batch = H_pi_batch - 0.5

    batch = list(H_pi_batch.size())[0]  # batch size
    H_pi_batch = 0.5 * (1 + F.tanh(H_pi_batch * 40))

    TotalBatchLoss = 0

    for i in range(batch):
        H_pi = H_pi_batch[i, :, :, :]  #(2,112,112)
        mask = gts[i, :, :, :]        #(2,112,112)

        TotalImgLoss = 0

        for lb in range(num_classes):
            LevelSetEnergy = calculate(H_pi, mask, lb, num_classes)

            TotalImgLoss += LevelSetEnergy

        TotalBatchLoss += TotalImgLoss/float(batch)

    return TotalBatchLoss

def calculate(H_pi, mask, lb, mask_num):
    mask_k = mask[lb,:,:]  #(112,112)
    H_pi_k = H_pi[lb,:,:]  #(112,112)
    eps = 0.00001

    Area_H_pi_k = torch.sum(H_pi_k)
    u0_mul_H = torch.sum(torch.mul(H_pi_k, mask_k))
    c1 = u0_mul_H / (Area_H_pi_k + eps)

    Area_1_H_pi_k = torch.sum(1- H_pi_k)
    u0_mul_H_1 = torch.sum(torch.mul((1 - H_pi_k), mask_k))
    c2 = u0_mul_H_1 / (Area_1_H_pi_k + eps)

    InternalArea = (mask_k - c1) * (mask_k - c1) * H_pi_k
    Loss_I = torch.sum(InternalArea)

    ExternalArea = (mask_k - c2) * (mask_k - c2) * (1 - H_pi_k)
    Loss_0 = torch.sum(ExternalArea)

    return (Loss_I + Loss_0) / mask_num


def active_contour_loss(y_pred, y_true):
    """
    Active Contour Loss

    y_true, y_pred: tensor of shape (N, C, H, W),
    where y_true[:,:,region_in_contour] == 1, y_true[:,:,region_out_contour] == 0.
    weight: scalar, length term weight.

    shape:
    y_pred: (N, 2, H, W)
    y_true: (N, 2, H, W)
    """
    y_pred = torch.sigmoid(y_pred).cuda()
    y_true = y_true.float().cuda()

    weight = 10  # length term weight
    lambdaP = 5.  # lambda parameter could be various

    # length term
    delta_r = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]  # horizontal gradient (N, C, H-1, W)
    delta_c = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]  # vertical gradient   (N, C, H, W-1)

    delta_r = delta_r[:, :, 1:, :-2] ** 2  # (N, C, H-2, W-2)
    delta_c = delta_c[:, :, :-2, 1:] ** 2  # (N, C, H-2, W-2)
    delta_pred = torch.abs(delta_r + delta_c)

    epsilon = 1e-8  # where is a parameter to avoid square root is zero in practice.
    lenth = torch.sum(torch.sqrt(delta_pred + epsilon))  # eq.(11) in the paper, mean is used instead of sum.

    # region term
    c_in = torch.ones(y_true.shape, dtype=torch.float32).cuda()  # shape: (H, W)
    c_out = torch.zeros(y_true.shape, dtype=torch.float32).cuda()

    region_in = torch.abs(
        torch.sum(y_pred[:, 1, :, :] * (y_true[:,1,:,:] - c_in[:,0,:,:]) ** 2))  # equ.(12) in the paper, mean is used instead of sum.
    region_out = torch.abs(torch.sum((1. - y_pred[:, 1, :, :]) * (y_true[:,1,:,:] - c_out[:,0,:,:]) ** 2))
    region = region_in + region_out

    acloss = weight * lenth + lambdaP * region

    return acloss

def surface_loss(y_pred, y_true):
    """
    :param y_pred: shape: [N, 2, W, H]
    :param y_true: shape: [N, 2, W, H]
    """
    y_pred = torch.sigmoid(y_pred).cuda()

    # assert simplex(y_pred)
    # assert not one_hot(y_true)
    y_true = y_true.type(torch.float32).cuda()

    #y_true = y_true.unsqueeze(1)
    #y_true_b = 1 - y_true
    #y_true = torch.cat((y_true, y_true_b), dim=1).type(torch.float32).cuda()

    multipled = torch.einsum("bcwh,bcwh->bcwh", y_pred, y_true)
    surloss = multipled.mean()

    return surloss