import torch, cv2
from scipy import misc, ndimage
from skimage.transform import warp, AffineTransform
import numpy.random as random
import numpy as np
import dataloaders.helpers as helpers
import dataloaders.skewed_axes_weight_map as tools
import seaborn as sns
from matplotlib import pyplot as plt

class ScaleNRotate(object):
    """Scale (zoom-in, zoom-out) and Rotate the image and the ground truth.
    Args:
        two possibilities:
        1.  rots (tuple): (minimum, maximum) rotation angle
            scales (tuple): (minimum, maximum) scale
        2.  rots [list]: list of fixed possible rotation angles
            scales [list]: list of fixed possible scales
    """
    def __init__(self, rots=(-30, 30), scales=(.75, 1.25), semseg=False):
        assert (isinstance(rots, type(scales)))
        self.rots = rots
        self.scales = scales
        self.semseg = semseg

    def __call__(self, sample):

        if type(self.rots) == tuple:
            # Continuous range of scales and rotations
            rot = (self.rots[1] - self.rots[0]) * random.random() - \
                  (self.rots[1] - self.rots[0])/2

            sc = (self.scales[1] - self.scales[0]) * random.random() - \
                 (self.scales[1] - self.scales[0]) / 2 + 1
        elif type(self.rots) == list:
            # Fixed range of scales and rotations
            rot = self.rots[random.randint(0, len(self.rots))]
            sc = self.scales[random.randint(0, len(self.scales))]

        for elem in sample.keys():
            if 'meta' in elem:
                continue

            tmp = sample[elem]

            h, w = tmp.shape[:2]
            center = (w / 2, h / 2)
            assert(center != 0)  # Strange behaviour warpAffine
            M = cv2.getRotationMatrix2D(center, rot, sc)

            if ((tmp == 0) | (tmp == 1)).all():
                flagval = cv2.INTER_NEAREST
            elif 'gt' in elem and self.semseg:
                flagval = cv2.INTER_NEAREST
            else:
                flagval = cv2.INTER_CUBIC
            tmp = cv2.warpAffine(tmp, M, (w, h), flags=flagval)

            sample[elem] = tmp

        return sample

    def __str__(self):
        return 'ScaleNRotate:(rot='+str(self.rots)+',scale='+str(self.scales)+')'


class RandomHorizontalFlip(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __call__(self, sample):

        if random.random() < 0.5:
            for elem in sample.keys():
                if 'meta' in elem:
                    continue
                tmp = sample[elem]
                tmp = cv2.flip(tmp, flipCode=1)
                sample[elem] = tmp

        return sample

    def __str__(self):
        return 'RandomHorizontalFlip'


class RandomAffineTransform(object):
    def __init__(self,
                 scale_range=(0.9, 1.1),
                 rotation_range=(-0.2, 0.2),
                 shear_range=(-0.2, 0.2),
                 translation_range=(-0.05, 0.05)
                 ):
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.shear_range = shear_range
        self.translation_range = translation_range

    def __call__(self, img_data):
        img = img_data.copy()
        h, w = img.shape
        scale_x = np.random.uniform(*self.scale_range)
        scale_y = np.random.uniform(*self.scale_range)
        scale = (scale_x, scale_y)
        rotation = np.random.uniform(*self.rotation_range)
        shear = np.random.uniform(*self.shear_range)
        translation = (
            np.random.uniform(*self.translation_range) * w,
            np.random.uniform(*self.translation_range) * h
        )
        af = AffineTransform(scale=scale, shear=shear, rotation=rotation, translation=translation)
        img = warp(img, af.inverse)
        return img

class CropFromBox(object):
    """
    Returns image cropped based on bounding box.
    """
    def __init__(self, crop_elems=('image', 'gt'),
                 mask_elem='gt',
                 relax=0,
                 zero_pad=False):
        self.crop_elems = crop_elems
        self.mask_elem = mask_elem
        self.relax = relax
        self.zero_pad = zero_pad

    def __call__(self, sample):
        _target = sample[self.mask_elem]
        if _target.ndim == 2:
            _target = np.expand_dims(_target, axis=-1)
        for elem in self.crop_elems:
            _img = sample[elem]
            _crop = []
            if self.mask_elem == elem:
                if _img.ndim == 2:
                    _img = np.expand_dims(_img, axis=-1)
                for k in range(0, _target.shape[-1]):
                    _tmp_img = _img[..., k]
                    _tmp_target = _target[..., k]
                    if np.max(_target[..., k]) == 0:
                        _crop.append(np.zeros(_tmp_img.shape, dtype=_img.dtype))
                    else:
                        _crop.append(helpers.crop_from_bbox(_tmp_img, bbox=sample['bbox'][0], zero_pad=self.zero_pad))
            else:
                for k in range(0, _target.shape[-1]):
                    if np.max(_target[..., k]) == 0:
                        _crop.append(np.zeros(_img.shape, dtype=_img.dtype))
                    else:
                        _tmp_target = _target[..., k]
                        _crop.append(helpers.crop_from_bbox(_img, bbox=sample['bbox'][0], zero_pad=self.zero_pad))
            if len(_crop) == 1:
                sample['crop_' + elem] = _crop[0]
            else:
                sample['crop_' + elem] = _crop
        return sample

    def __str__(self):
        return 'CropFromBox:(crop_elems='+str(self.crop_elems)+', mask_elem='+str(self.mask_elem)+\
               ', relax='+str(self.relax)+',zero_pad='+str(self.zero_pad)+')'


class CropFromMask(object):
    """
    Returns image cropped in bounding box from a given mask
    """
    def __init__(self, crop_elems=('image', 'gt'),
                 mask_elem='gt',
                 relax=0,
                 zero_pad=False,
                 dummy=False):

        self.crop_elems = crop_elems
        self.mask_elem = mask_elem
        self.relax = relax
        self.zero_pad = zero_pad
        self.dummy = dummy  # if True: keep a copy of data without crop.

    def __call__(self, sample):
        if self.dummy:
            for elem in self.crop_elems:
                sample['crop_' + elem] = sample[elem].copy()
            return sample

        _target = sample[self.mask_elem]
        if _target.ndim == 2:
            _target = np.expand_dims(_target, axis=-1)
        for elem in self.crop_elems:
            _img = sample[elem]
            _crop = []
            if self.mask_elem == elem:
                if _img.ndim == 2:
                    _img = np.expand_dims(_img, axis=-1)
                for k in range(0, _target.shape[-1]):
                    _tmp_img = _img[..., k]
                    _tmp_target = _target[..., k]
                    if np.max(_target[..., k]) == 0:
                        _crop.append(np.zeros(_tmp_img.shape, dtype=_img.dtype))
                    else:
                        _crop.append(helpers.crop_from_mask(_tmp_img, _tmp_target, relax=self.relax, zero_pad=self.zero_pad))
            else:
                for k in range(0, _target.shape[-1]):
                    if np.max(_target[..., k]) == 0:
                        _crop.append(np.zeros(_img.shape, dtype=_img.dtype))
                    else:
                        _tmp_target = _target[..., k]
                        _crop.append(helpers.crop_from_mask(_img, _tmp_target, relax=self.relax, zero_pad=self.zero_pad))
            if len(_crop) == 1:
                sample['crop_' + elem] = _crop[0]
            else:
                sample['crop_' + elem] = _crop
        return sample

    def __str__(self):
        return 'CropFromMask:(crop_elems='+str(self.crop_elems)+', mask_elem='+str(self.mask_elem)+\
               ', relax='+str(self.relax)+',zero_pad='+str(self.zero_pad)+')'


class FixedResize(object):
    """Resize the image and the ground truth to specified resolution.
    Args:
        resolutions (dict): the list of resolutions
    """
    def __init__(self, resolutions=None, flagvals=None):
        self.resolutions = resolutions
        self.flagvals = flagvals
        if self.flagvals is not None:
            assert(len(self.resolutions) == len(self.flagvals))

    def __call__(self, sample):

        # Fixed range of scales
        if self.resolutions is None:
            return sample

        elems = list(sample.keys())
        for elem in elems:
            if 'meta' in elem or 'bbox' in elem or ('extreme_points_coord' in elem and elem not in self.resolutions):
                continue
            if 'extreme_points_coord' in elem and elem in self.resolutions:
                bbox = sample['bbox']
                crop_size = np.array([bbox[3]-bbox[1]+1, bbox[4]-bbox[2]+1])
                res = np.array(self.resolutions[elem]).astype(np.float32)
                sample[elem] = np.round(sample[elem]*res/crop_size).astype(np.int)
                continue
            if elem in self.resolutions:
                if self.resolutions[elem] is None:
                    continue
                if isinstance(sample[elem], list):
                    if sample[elem][0].ndim == 3:
                        output_size = np.append(self.resolutions[elem], [3, len(sample[elem])])
                    else:
                        output_size = np.append(self.resolutions[elem], len(sample[elem]))
                    tmp = sample[elem]
                    sample[elem] = np.zeros(output_size, dtype=np.float32)
                    for ii, crop in enumerate(tmp):
                        if self.flagvals is None:
                            sample[elem][..., ii] = helpers.fixed_resize(crop, self.resolutions[elem])
                        else:
                            sample[elem][..., ii] = helpers.fixed_resize(crop, self.resolutions[elem], flagval=self.flagvals[elem])
                else:
                    if self.flagvals is None:
                        sample[elem] = helpers.fixed_resize(sample[elem], self.resolutions[elem])
                    else:
                        sample[elem] = helpers.fixed_resize(sample[elem], self.resolutions[elem], flagval=self.flagvals[elem])
            else:
                del sample[elem]

        return sample

    def __str__(self):
        return 'FixedResize:'+str(self.resolutions)


class ExtremePoints(object):
    """
    Returns the four extreme points (left, right, top, bottom) (with some random perturbation) in a given binary mask
    sigma: sigma of Gaussian to create a heatmap from a point
    pert: number of pixels fo the maximum perturbation
    elem: which element of the sample to choose as the binary mask
    """
    def __init__(self, sigma=10, pert=0, elem='gt'):
        self.sigma = sigma
        self.pert = pert
        self.elem = elem

    def __call__(self, sample):
        if sample[self.elem].ndim == 3:
            raise ValueError('ExtremePoints not implemented for multiple object per image.')
        _target = sample[self.elem]
        if np.max(_target) == 0:
            sample['extreme_points'] = np.zeros(_target.shape, dtype=_target.dtype) #  TODO: handle one_mask_per_point case
        else:
            _points = helpers.extreme_points(_target, self.pert)
            sample['extreme_points'] = helpers.make_gt(_target, _points, sigma=self.sigma, one_mask_per_point=False)

        return sample

    def __str__(self):
        return 'ExtremePoints:(sigma='+str(self.sigma)+', pert='+str(self.pert)+', elem='+str(self.elem)+')'


class ConfidenceMap(object):
    """
    Returns the confidence map derived from four extreme points
    """
    def __init__(self, sigma=10, pert=0, elem1='crop_image', elem2='crop_gt'):
        self.sigma =sigma
        self.pert = pert
        self.elem1 = elem1  # image
        self.elem2 = elem2  # gt
    def __call__(self, sample):
        if sample[self.elem2].ndim == 3:
            raise ValueError('ExtremePoints not implemented for multiple object per image.')
        _target = sample[self.elem2] # gt
        if np.max(_target) == 0:
            _points = np.zeros(_target.shape, dtype=_target.dtype) #  TODO: handle one_mask_per_point case
        else:
            _points = helpers.extreme_points(_target, self.pert) # four extreme points

        _target = sample[self.elem1] # image
        h, w, c = np.shape(_target)
        x = np.linspace(0, w-1, num=w)
        y = np.linspace(0, h-1, num=w)
        d1, d2 = tools.compute_d1_d2_fast_skewed_axes(x, y, _points)

        if isinstance(d1, int) and isinstance(d2, int):
            if d1 == 0 and d2 == 0:
                _target = sample[self.elem2] # gt
                tmp = helpers.make_gt(_target, _points, sigma=self.sigma, one_mask_per_point=False)
                sample['confidence_map'] = 255 * (tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-10)
        else:
            _target = sample[self.elem2]  # gt
            tmp = helpers.make_gt(_target, _points, sigma=self.sigma, one_mask_per_point=False)
            ep_map = 255 * (tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-10)
            cof_map = 1 / (1 + d1*d2)
            cof_map_norm = 255 * (cof_map - cof_map.min()) / (cof_map.max() - cof_map.min() + 1e-10)
            sample['confidence_map'] = ep_map + cof_map_norm

        return sample


class SDT(object):
    """
    Returns the distance transform: dt, sdt, sdt_noise
    """
    def __init__(self, elem='crop_gt', sigma=0.1, dt_max=15, video_mode=False, static_train=True):
        self.elem = elem
        self.sigma = sigma
        self.dt_max = dt_max

        self.video_mode = video_mode        # in video mode, additional processing for prev frame is needed
        self.static_train = static_train    # added random affine transform for static training mode of video setting
        self.prev_tr = RandomAffineTransform() if video_mode and static_train else None

    def __call__(self, sample):
        _target = sample[self.elem]
        if np.max(_target) != 0:
            _points = cv2.findContours(_target.copy().astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2]
            _contours = cv2.drawContours(np.zeros(_target.shape), _points, -1, 1)

            dt = ndimage.distance_transform_edt(_contours == 0)
            sample['dt'] = dt

            sdt = dt.copy()
            sdt[sdt > self.dt_max] = self.dt_max  # truncate
            sdt[_target > 0] *= -1
            sample['sdt'] = sdt
        else:
            sample['dt'] = _target.copy()
            sample['sdt'] = _target.copy()

        if self.video_mode:
            # generate sdt of prev frame
            if self.static_train:
                assert 'crop_gt' in sample.keys() and self.prev_tr is not None
                curr_gt = sample['crop_gt']
                _target = self.prev_tr(curr_gt)
            else:
                assert 'crop_prev_gt' in sample.keys()
                _target = sample['crop_prev_gt']

            if np.max(_target) != 0:
                _points = cv2.findContours(_target.copy().astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2]
                _contours = cv2.drawContours(np.zeros(_target.shape), _points, -1, 1)

                sdt = ndimage.distance_transform_edt(_contours == 0)
                sdt[sdt > self.dt_max] = self.dt_max  # truncate
                sdt[_target > 0] *= -1
                sample['crop_prev_sdt'] = sdt
            else:
                sample['crop_prev_sdt'] = _target.copy()

        return sample

    def __str__(self):
        return 'SDT:(elem='+str(self.elem)+')'


class ConcatInputs(object):

    def __init__(self, elems=('image', 'point')):
        self.elems = elems

    def __call__(self, sample):

        res = sample[self.elems[0]]

        for elem in self.elems[1:]:
            assert(sample[self.elems[0]].shape[:2] == sample[elem].shape[:2])

            # Check if third dimension is missing
            tmp = sample[elem]
            if tmp.ndim == 2:
                tmp = tmp[:, :, np.newaxis]

            res = np.concatenate((res, tmp), axis=2)

        sample['concat'] = res

        return sample

    def __str__(self):
        return 'ConcatInputs:'+str(self.elems)


class ToImage(object):
    """
    Return the given elements between 0 and 255
    """
    def __init__(self, norm_elem='image', custom_max=255.):
        self.norm_elem = norm_elem
        self.custom_max = custom_max

    def __call__(self, sample):
        if isinstance(self.norm_elem, tuple):
            for elem in self.norm_elem:
                tmp = sample[elem]
                sample[elem] = self.custom_max * (tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-10)
        else:
            tmp = sample[self.norm_elem]
            sample[self.norm_elem] = self.custom_max * (tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-10)
        return sample

    def __str__(self):
        return 'NormalizeImage'


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        for elem in sample.keys():
            if 'meta' in elem:
                continue
            elif 'bbox' in elem:
                tmp = sample[elem]
                sample[elem] = torch.from_numpy(tmp)
                continue

            tmp = sample[elem]

            if tmp.ndim == 2:
                tmp = tmp[:, :, np.newaxis]

            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            tmp = tmp.transpose((2, 0, 1))
            tmp = tmp.astype(np.float16)
            sample[elem] = torch.FloatTensor(tmp)

        return sample

    def __str__(self):
        return 'ToTensor'


class InverseEdge(object):
    """
    Returns the edges
    """
    def __init__(self, elem='gt', out_elem='inv_edge'):
        self.elem = elem
        self.out_elem = out_elem
        self.strel = np.ones((3, 3))

    def __call__(self, sample):
        _target = sample[self.elem]
        if np.max(_target) != 0:
            _points = cv2.findContours(_target.copy().astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2]
            _contours = cv2.drawContours(np.zeros(_target.shape), _points, -1, 1)
            _expanded = cv2.dilate(_contours, self.strel)
            sample[self.out_elem] = (_expanded < 0.5).astype(np.float32)
            return sample
        else:
            sample[self.out_elem] = np.ones(_target.shape)
            return sample

    def __str__(self):
        return 'InverseEdge'

