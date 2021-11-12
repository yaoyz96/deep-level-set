import os.path

import cv2
import numpy as np
from PIL import Image
import scipy.misc as sm

import dataloaders.helpers as helpers
from evaluation.f_boundary import *

def jaccard(annotation, segmentation, void_pixels=None):

    assert(annotation.shape == segmentation.shape)

    if void_pixels is None:
        void_pixels = np.zeros_like(annotation)
    assert(void_pixels.shape == annotation.shape)

    annotation = annotation.astype(np.bool)
    segmentation = segmentation.astype(np.bool)
    void_pixels = void_pixels.astype(np.bool)
    if np.isclose(np.sum(annotation & np.logical_not(void_pixels)), 0) and np.isclose(np.sum(segmentation & np.logical_not(void_pixels)), 0):
        return 1
    else:
        return np.sum(((annotation & segmentation) & np.logical_not(void_pixels))) / \
               np.sum(((annotation | segmentation) & np.logical_not(void_pixels)), dtype=np.float32)


def eval_one_result(network, loader, folder, mask_thres=0.5, bd_threshold=(2,),
                    one_mask_per_image=False, use_void_pixels=True):
    def mAPr(per_cat, thresholds):
        n_cat = len(per_cat)
        all_apr = np.zeros(len(thresholds))
        for ii, th in enumerate(thresholds):
            per_cat_recall = np.zeros(n_cat)
            for jj, categ in enumerate(per_cat.keys()):
                per_cat_recall[jj] = np.sum(np.array(per_cat[categ]) > th)/len(per_cat[categ])
            all_apr[ii] = per_cat_recall.mean()
        return all_apr.mean()

    # Allocate
    eval_result = dict()
    eval_result["all_jaccards"] = np.zeros(len(loader))
    eval_result["per_categ_jaccard"] = dict()
    eval_result["per_categ_mean_jaccard"] = dict()

    eval_result["all_F"] = dict()
    eval_result["per_categ_F"] = dict()
    eval_result["per_categ_mean_F"] = dict()
    for bd_th in bd_threshold:
        eval_result["all_F"][bd_th] = np.zeros(len(loader))
        eval_result["per_categ_F"][bd_th] = dict()
        eval_result["per_categ_mean_F"][bd_th] = dict()

    print('Mask_TH = ' + str(mask_thres))

    # Iterate
    for i, sample in enumerate(loader):

        if i % 500 == 0:
            print('Evaluating: {} of {} objects'.format(i, len(loader)))

        # check void pixels
        if use_void_pixels and 'void_pixels' not in sample.keys():
            use_void_pixels = False

        # Load result
        if not one_mask_per_image:
            filename = os.path.join(folder, sample["meta"]["image"][0] + '-' + sample["meta"]["object"][0] + '.npy')
        else:
            filename = os.path.join(folder, sample["meta"]["image"][0] + '.npy')

        if not os.path.exists(filename):
            continue

        data = np.load(filename)
        mask = (data <= mask_thres).astype(np.float32)      # 0-1 mask

        # save mask result
        save = os.path.join(os.path.dirname(folder), 'mask_' + str(mask_thres))
        if not os.path.exists(save):
            os.makedirs(save)
        maskname = os.path.join(save, sample["meta"]["image"][0] + '-' + sample["meta"]["object"][0] + '.png')
        sm.imsave(maskname, mask)

        if network == 'resnet101-skip-pretrain':
            gt = np.squeeze(helpers.tens2image(sample["gt"]))
        elif network == 'resnet101-skip-bd-pretrain':
            gt = np.squeeze(helpers.tens2image(sample["gt_f"]))

        if use_void_pixels:
            void_pixels = np.squeeze(helpers.tens2image(sample["void_pixels"]))
        if mask.shape != gt.shape:
            mask = cv2.resize(mask, gt.shape[::-1], interpolation=cv2.INTER_CUBIC)

        mask = (mask >= 0.5)
        if use_void_pixels:
            void_pixels = (void_pixels >= 0.5)

        # Evaluate
        if use_void_pixels:
            eval_result["all_jaccards"][i] = jaccard(gt, mask, void_pixels)
        else:
            eval_result["all_jaccards"][i] = jaccard(gt, mask)

        for bd_th in bd_threshold:
            eval_result['all_F'][bd_th][i] = db_eval_boundary(mask, gt, bd_th)

        # Store in per category
        if "category" in sample["meta"]:
            cat = sample["meta"]["category"][0]
        else:
            cat = 1

        if cat not in eval_result["per_categ_jaccard"]:
            eval_result["per_categ_jaccard"][cat] = []
        eval_result["per_categ_jaccard"][cat].append(eval_result["all_jaccards"][i])

        for bd_th in bd_threshold:
            if cat not in eval_result['per_categ_F'][bd_th]:
                eval_result['per_categ_F'][bd_th][cat] = []
            eval_result['per_categ_F'][bd_th][cat].append(eval_result['all_F'][bd_th][i])

    # Compute some stats
    eval_result["J mAPr0.5"] = mAPr(eval_result["per_categ_jaccard"], [0.5])
    eval_result["J mAPr0.7"] = mAPr(eval_result["per_categ_jaccard"], [0.7])
    eval_result["J mAPr-vol"] = mAPr(eval_result["per_categ_jaccard"], np.linspace(0.1, 0.9, 9))

    for cat in eval_result["per_categ_jaccard"].keys():
        eval_result["per_categ_mean_jaccard"][cat] = np.mean(eval_result["per_categ_jaccard"][cat])
    eval_result["mIoU"] = np.mean(list(eval_result["per_categ_mean_jaccard"].values()))

    for bd_th in bd_threshold:
        eval_result["F mAPr0.5 TH" + str(bd_th)] = mAPr(eval_result['per_categ_F'][bd_th], [0.5])
        eval_result["F mAPr0.7 TH" + str(bd_th)] = mAPr(eval_result['per_categ_F'][bd_th], [0.7])

        for cat in eval_result["per_categ_F"][bd_th].keys():
            eval_result["per_categ_mean_F"][bd_th][cat] = np.mean(eval_result["per_categ_F"][bd_th][cat])
        eval_result["F mean TH" + str(bd_th)] = np.mean(list(eval_result["per_categ_mean_F"][bd_th].values()))

    return eval_result

