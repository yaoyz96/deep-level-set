import socket
import timeit
from datetime import datetime
import cv2
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import cm, axes
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
from tensorboardX import SummaryWriter

from dataloaders import prostate, refuge, dgs
from dataloaders import custom_transforms as tr
import networks.backend_cnn as backend_cnn
from layers.loss import *
from layers.lse import *
from dataloaders.helpers import *
from evaluation.eval import *


class DELSE(object):
    def __init__(self, args):
        self.args = args

        # model
        self.mode = args.mode
        self.cm = args.cmap
        self.loss = args.loss
        self.model_name = args.model_name
        self.resolution = (args.resolution, args.resolution)

        self.net = backend_cnn.backend_cnn_model(args)

        self.weight = args.loss_weight
        self.num_classes = args.num_classes

        if self.args.resume_epoch == 0:
            print('Initializing from pretrained VOC model...')
        else:
            resume_dir = os.path.join(args.save_dir_root, 'run_%04d' % args.resume_id)
            print("Initializing weights from: {}".format(
                os.path.join(resume_dir, 'models', self.model_name + '_epoch-' + str(args.resume_epoch - 1) + '.pth')))
            self.net.load_state_dict(
                torch.load(os.path.join(resume_dir, 'models', self.model_name + '_epoch-' + str(args.resume_epoch - 1) + '.pth'),
                           map_location=lambda storage, loc: storage))

        # optimizer
        self.train_params = [{'params': self.net.get_1x_lr_params(), 'lr': args.lr},
                             {'params': self.net.get_10x_lr_params(), 'lr': args.lr * 10}]
        self.optimizer = optim.SGD(self.train_params, lr=args.lr, momentum=args.momentum, weight_decay=args.wd)

        # transforms
        if self.cm is True:
            self.composed_transforms_tr = transforms.Compose([
                tr.CropFromMask(crop_elems=('image', 'gt_f', 'gts'), mask_elem='gt_f', relax=args.relax_crop,
                                zero_pad=args.zero_pad_crop),
                tr.FixedResize(resolutions={'crop_image': self.resolution, 'crop_gt_f': self.resolution,
                                            'crop_gts': self.resolution}),
                tr.SDT(elem='crop_gt_f', dt_max=args.dt_max),
                tr.ConfidenceMap(sigma=10, pert=5, elem1='crop_image', elem2='crop_gt_f'),
                tr.ConcatInputs(elems=('crop_image', 'confidence_map')),
                tr.ToTensor()])
            self.composed_transforms_ts = transforms.Compose([
                tr.CropFromMask(crop_elems=('image', 'gt_f', 'gts'), mask_elem='gt_f', relax=args.relax_crop,
                                zero_pad=args.zero_pad_crop),
                tr.FixedResize(resolutions={'void_pixels': None, 'gt_f': None, 'crop_image': self.resolution,
                                            'crop_gt_f': self.resolution, 'crop_gts': self.resolution}),
                tr.SDT(elem='crop_gt_f', dt_max=args.dt_max),
                tr.ConfidenceMap(sigma=10, pert=0, elem1='crop_image', elem2='crop_gt_f'),
                tr.ConcatInputs(elems=('crop_image', 'confidence_map')),
                tr.ToTensor()])
        else:
            self.composed_transforms_tr = transforms.Compose([
                #tr.RandomHorizontalFlip(),
                #tr.ScaleNRotate(rots=(-20, 20), scales=(.75, 1.25)),
                tr.CropFromMask(crop_elems=('image', 'gt_f', 'gts'), mask_elem='gt_f', relax=args.relax_crop, zero_pad=args.zero_pad_crop),
                tr.FixedResize(resolutions={'crop_image': self.resolution, 'crop_gt_f': self.resolution, 'crop_gts': self.resolution}),
                tr.SDT(elem='crop_gt_f', dt_max=args.dt_max),
                tr.ExtremePoints(sigma=10, pert=5, elem='crop_gt_f'),
                tr.ToImage(norm_elem='extreme_points'),
                tr.ConcatInputs(elems=('crop_image', 'extreme_points')),
                tr.ToTensor()])
            self.composed_transforms_ts = transforms.Compose([
                tr.CropFromMask(crop_elems=('image', 'gt_f', 'gts'), mask_elem='gt_f', relax=args.relax_crop,
                                zero_pad=args.zero_pad_crop),
                tr.FixedResize(resolutions={'void_pixels': None, 'gt_f': None, 'crop_image': self.resolution,
                                            'crop_gt_f': self.resolution, 'crop_gts': self.resolution}),
                tr.SDT(elem='crop_gt_f', dt_max=args.dt_max),
                tr.ExtremePoints(sigma=10, pert=0, elem='crop_gt_f'),
                tr.ToImage(norm_elem='extreme_points'),
                tr.ConcatInputs(elems=('crop_image', 'extreme_points')),
                tr.ToTensor()])

        # dataset
        if self.args.dataset == 'prostate':
            if self.mode == 'train':
                self.trainset = prostate.ProstateWithBD(train=True, split='train', transform=self.composed_transforms_tr)
                self.valset = prostate.ProstateWithBD(train=False, split='test', transform=self.composed_transforms_ts, retname=True)
            elif self.mode == 'test':
                self.valset = prostate.ProstateWithBD(train=False, split='test', transform=self.composed_transforms_ts, retname=True)
        elif self.args.dataset == 'refuge':
            if self.mode == 'train':
                self.trainset = refuge.RefugeWithBD(train=True, split='train', transform=self.composed_transforms_tr)
                self.valset = refuge.RefugeWithBD(train=False, split='val', transform=self.composed_transforms_ts, retname=True)
            elif self.mode == 'test':
                self.valset = refuge.RefugeWithBD(train=False, split='test', transform=self.composed_transforms_ts, retname=True)
        elif self.args.dataset == 'dgs':
            if self.mode == 'train':
                self.trainset = dgs.Dgs1WithBD(train=True, split='train', transform=self.composed_transforms_tr)
                self.valset = dgs.Dgs1WithBD(train=False, split='val', transform=self.composed_transforms_ts, retname=True)
            elif self.mode == 'test':
                self.valset = dgs.Dgs1WithBD(train=False, split='test', transform=self.composed_transforms_ts, retname=True)

        # dataloader
        if self.mode == 'train':
            self.trainloader = DataLoader(self.trainset, batch_size=args.batch, shuffle=True, num_workers=8)
            self.testloader = DataLoader(self.valset, batch_size=1, shuffle=False, num_workers=0)
            self.num_img_tr = len(self.trainloader)
            self.num_img_ts = len(self.testloader)
        elif self.mode == 'test':
            self.testloader = DataLoader(self.valset, batch_size=1, shuffle=False, num_workers=0)
            self.num_img_ts = len(self.testloader)

        # tensorboard
        log_dir = os.path.join(args.save_dir, 'models',
                               args.txt + '_' + datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
        self.writer = SummaryWriter(log_dir=log_dir)

        # move to gpu
        self.net = torch.nn.DataParallel(self.net).cuda()


    def __del__(self):
        if hasattr(self, 'writer'):
            self.writer.close()

    def train(self, epoch):
        # init
        torch.cuda.empty_cache()
        self.net.train()

        running_loss_tr = 0.0  # total loss
        sum_phi_loss = 0.0     # original ce loss
        #sum_bd_loss = 0.0      # level set loss
        sum_ac_loss = 0.0      # active contour loss
        #sum_sur_loss = 0.0     # surface loss

        aveGrad = 0
        start_time = timeit.default_timer()

        # main training loop
        for ii, sample_batched in enumerate(self.trainloader):
            if ii == self.num_img_tr - 1:
                break
            # data
            inputs, sdts = sample_batched['concat'], sample_batched['sdt']
            gts_f = sample_batched['crop_gt_f']
            gtss = sample_batched['crop_gts']

            inputs.requires_grad_()
            inputs, sdts = inputs.cuda(), sdts.cuda()

            gts_f = torch.ge(gts_f, 0.5).float().cuda()
            gtss = torch.ge(gtss, 0.5).float().cuda()

            dts = sample_batched['dt']
            dts = dts.cuda()  # (batch, 1, H, W)
            vfs = gradient_sobel(dts, split=False)

            # Forward of the mini-batch
            phi_0, energy, g, prob = self.net.forward(inputs)    # inputs: (batch, nchannels, 512, 512)

            # (batch, nclasses, 64, 64) -> (batch, nclasses, 512, 512)
            phi_0 = F.upsample(phi_0, size=self.resolution, mode='bilinear', align_corners=True)
            energy = F.upsample(energy, size=self.resolution, mode='bilinear', align_corners=True)
            g = F.sigmoid(F.upsample(g, size=self.resolution, mode='bilinear', align_corners=True))
            prob = F.upsample(prob, size=self.resolution, mode='bilinear', align_corners=True)

            # loss
            if not self.args.e2e and epoch < self.args.pretrain_epoch:
                # pre-train
                phi_0_loss = level_map_loss(phi_0, sdts, self.args.alpha)
                rand_shift = 10 * np.random.rand() - 5
                phi_T = levelset_evolution(sdts + rand_shift, energy, g, T=self.args.T, dt_max=self.args.dt_max)
                phi_T_loss = vector_field_loss(energy, vfs, sdts) \
                             + LSE_output_loss(phi_T, gts_f, sdts, self.args.epsilon, dt_max=self.args.dt_max)
                if self.loss == 'ac':
                    ac_loss = active_contour_loss(prob, gtss) * self.weight
                    loss = phi_0_loss + phi_T_loss + ac_loss
                    running_loss_tr += loss.item()
                    # sum loss
                    sum_phi_loss += phi_T_loss
                    sum_ac_loss += ac_loss 
                elif self.loss == 'ce':
                    loss = phi_0_loss + phi_T_loss
                    running_loss_tr += loss.item()
                    # sum loss
                    sum_phi_loss += phi_T_loss
            else:
                # joint-train
                rand_shift = 10 * np.random.rand() - 5
                phi_T = levelset_evolution(phi_0 + rand_shift, energy, g, T=self.args.T, dt_max=self.args.dt_max)
                phi_T_loss = LSE_output_loss(phi_T, gts_f, sdts, self.args.epsilon, dt_max=self.args.dt_max)
                if self.loss == 'ac':
                    ac_loss = active_contour_loss(prob, gtss) * self.weight
                    loss = phi_T_loss + ac_loss
                    running_loss_tr += loss.item()
                    # sum loss
                    sum_phi_loss += phi_T_loss
                    sum_ac_loss += ac_loss
                elif self.loss == 'ce':
                    loss = phi_T_loss
                    running_loss_tr += loss.item()
                    # sum loss
                    sum_phi_loss += phi_T_loss

            if self.loss == 'ac':
                aver_phi_loss = sum_phi_loss / self.num_img_tr
                aver_ac_loss = sum_ac_loss / self.num_img_tr
            elif self.loss == 'ce':
                aver_phi_loss = sum_phi_loss / self.num_img_tr

            # Backward the averaged gradient
            loss /= self.args.ave_grad
            loss.backward()
            aveGrad += 1

            # Update the weights once
            if aveGrad % self.args.ave_grad == 0:
                self.writer.add_scalar('data/total_loss_iter', loss.item(), ii + self.num_img_tr * epoch)
                if not self.args.e2e and epoch < self.args.pretrain_epoch:
                    self.writer.add_scalar('data/total_phi_0_loss_iter', phi_0_loss.item(), ii + self.num_img_tr * epoch)
                self.writer.add_scalar('data/total_phi_T_loss_iter', phi_T_loss.item(), ii + self.num_img_tr * epoch)
                #self.writer.add_scalar('data/total_ls_loss_iter', bd_loss.item(), ii + self.num_img_tr * epoch)
                clip_grad_norm(self.net.parameters(), 10)
                self.optimizer.step()
                self.optimizer.zero_grad()
                aveGrad = 0

        # print
        running_loss_tr = running_loss_tr / self.num_img_tr
        self.writer.add_scalar('data/total_loss_epoch', running_loss_tr, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, ii * self.args.batch + inputs.data.shape[0]))
        print('Total loss: %f' % running_loss_tr)
        if self.loss == 'ac':
            print('aver_phi_T_loss: %f' % aver_phi_loss)
            print('aver_ac_loss: %f' % aver_ac_loss)
        elif self.loss == 'ce':
            print('aver_phi_T_loss: %f' % aver_phi_loss)
        stop_time = timeit.default_timer()
        print("Execution time: " + str(stop_time - start_time) + "\n")


    def test(self, epoch, _save=False):
        # init
        torch.cuda.empty_cache()
        self.net.eval()

        # main testing loop
        with torch.no_grad():
            IoU = 0.0

            running_loss_ts = 0.0
            start_time = timeit.default_timer()

            for ii, sample_batched in enumerate(self.testloader):
                inputs, sdts, metas = sample_batched['concat'], sample_batched['sdt'], sample_batched['meta']
                gts_f = sample_batched['crop_gt_f']
                gtss = sample_batched['crop_gts']

                # Forward of the mini-batch
                inputs, sdts = inputs.cuda(), sdts.cuda()
                gts_f = torch.ge(gts_f, 0.5).float().cuda()
                gtss = torch.ge(gtss, 0.5).float().cuda()

                phi_0, energy, g, prob = self.net.forward(inputs)
                phi_0 = F.upsample(phi_0, size=self.resolution, mode='bilinear', align_corners=True)
                energy = F.upsample(energy, size=self.resolution, mode='bilinear', align_corners=True)
                g = F.sigmoid(F.upsample(g, size=self.resolution, mode='bilinear', align_corners=True))
                prob = F.upsample(prob, size=self.resolution, mode='bilinear', align_corners=True)

                phi_T = levelset_evolution(phi_0, energy, g, T=self.args.T, dt_max=self.args.dt_max, _test=True)
                if self.loss == 'ac':
                    loss = LSE_output_loss(phi_T, gts_f, sdts, self.args.epsilon, dt_max=self.args.dt_max) \
                            + active_contour_loss(prob, gtss) * self.weight
                elif self.loss == 'ce':
                    loss = LSE_output_loss(phi_T, gts_f, sdts, self.args.epsilon, dt_max=self.args.dt_max)

                running_loss_ts += loss.item()

                # cal & save!
                format = lambda x: np.squeeze(np.transpose(x.cpu().data.numpy()[0, :, :, :], (1, 2, 0)))
                phi_T = format(phi_T)

                gt_f = tens2image(sample_batched['gt_f'][0, :, :, :])
                bbox = get_bbox(gt_f, pad=self.args.relax_crop, zero_pad=self.args.zero_pad_crop)
                result = crop2fullmask(phi_T, bbox, gt_f, zero_pad=self.args.zero_pad_crop, relax=self.args.relax_crop, bg_value=20)

                # IoU
                if 'pascal' in self.args.dataset:
                    void_pixels = tens2image(sample_batched['void_pixels'][0, :, :, :])
                    void_pixels = (void_pixels >= 0.5)
                else:
                    void_pixels = None
                IoU += jaccard(gt_f, (result <= self.args.mask_threshold), void_pixels)

                # save outputs
                if _save:
                    save_dir_res = os.path.join(self.args.save_dir, 'results_ep' + str(epoch))
                    if not os.path.exists(save_dir_res):
                        os.makedirs(save_dir_res)
                    # round() is used to save space
                    np.save(os.path.join(save_dir_res, metas['image'][0] + '-' + metas['object'][0] + '.npy'),
                            result.round().astype(np.int8))

            # Print stuff at end of testing
            running_loss_ts = running_loss_ts / self.num_img_ts
            print('[Epoch: %d, numImages: %5d]' % (epoch, ii + inputs.data.shape[0]))
            self.writer.add_scalar('data/test_loss_epoch', running_loss_ts, epoch)
            print('Loss: %f' % running_loss_ts)

            mIoU = IoU / self.num_img_ts
            self.writer.add_scalar('data/test_mean_IoU', mIoU, epoch)
            print('Mean IoU: %f' % mIoU)

            stop_time = timeit.default_timer()
            print("Test time: " + str(stop_time - start_time) + "\n")

    def save_ckpt(self, epoch):
        torch.save(self.net.module.state_dict(),
                   os.path.join(self.args.save_dir, 'models', self.model_name + '_epoch-' + str(epoch) + '.pth'))

    def adjust_learning_rate(self, epoch, adjust_epoch=None, ratio=0.1):
        """Sets the learning rate to the initial LR decayed at selected epochs"""
        if epoch in adjust_epoch:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * ratio

